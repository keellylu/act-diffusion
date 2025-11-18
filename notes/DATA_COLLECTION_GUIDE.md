# Spot Robot Data Collection Guide

Complete guide for collecting pick-and-place demonstration data from your Spot robot for ACT training.

## Overview

The data collection pipeline records synchronized:
- **Robot state** (9 DOF): 6 arm joints + 3 base velocities
- **Camera images**: Hand camera + Scene camera
- **Actions**: Target joint positions and base velocities

Data is saved in HDF5 format compatible with the ACT training pipeline.

## Quick Start

### 1. Setup Data Collection Script

The provided `record_spot_episodes.py` is a template. You need to implement:

```python
# TODO items in the script:
1. _init_spot_robot() - Connect to your Spot robot
2. get_robot_state() - Read current joint positions and velocities  
3. get_camera_images() - Capture images from both cameras
4. execute_action() - Send commands to robot (if using autonomous policy)
```

### 2. Collect Data

```bash
python record_spot_episodes.py \
    --dataset_dir ./data/spot_pick_place \
    --num_episodes 50 \
    --episode_len 400 \
    --start_idx 0
```

### 3. Verify Data

```bash
# Verify single episode
python verify_spot_data.py \
    --dataset_dir ./data/spot_pick_place \
    --episode_idx 0

# Verify entire dataset
python verify_spot_data.py \
    --dataset_dir ./data/spot_pick_place
```

### 4. Train ACT Policy

```bash
python imitate_episodes.py \
    --task_name spot_mobile_manipulation \
    --ckpt_dir ./checkpoints/spot_pick_place \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0
```

## Data Format Specification

### HDF5 File Structure

Each episode is saved as `episode_N.hdf5` with the following structure:

```
episode_0.hdf5
├── attributes
│   └── sim: False (indicates real robot data)
├── observations/
│   ├── qpos: (T, 9) float64
│   ├── qvel: (T, 9) float64
│   └── images/
│       ├── hand_rgb: (T, H, W, 3) uint8
│       └── head_rgb: (T, H, W, 3) uint8
└── action: (T, 9) float64
```

Where:
- `T` = number of timesteps in episode (e.g., 400)
- `H, W` = image height and width (default: 480x640)

### State Representation (9 DOF)

```python
qpos = [
    arm_sh0,        # Arm joint 0 (shoulder)
    arm_sh1,        # Arm joint 1 (shoulder)
    arm_el0,        # Arm joint 2 (elbow)
    arm_el1,        # Arm joint 3 (elbow)
    arm_wr0,        # Arm joint 4 (wrist)
    arm_wr1,        # Arm joint 5 (wrist)
    velocity_x,     # Base forward velocity (m/s)
    velocity_y,     # Base lateral velocity (m/s)
    angular_vel,    # Base angular velocity (rad/s)
]
```

### Camera Setup

**hand_rgb** (Hand-mounted camera):
- Purpose: Close-up manipulation view
- Mounted on: Gripper/wrist
- Shows: Object being manipulated, gripper interactions
- Resolution: 640x480 (adjustable)

**head_rgb** (Scene camera):
- Purpose: Spatial reference and context
- Mounted on: Robot head/body
- Shows: Overall scene, navigation context, object locations
- Resolution: 640x480 (adjustable)

### Action Space

Actions represent **target states** (not velocities):
```python
action = [
    target_arm_sh0,     # Target position for arm joint 0
    target_arm_sh1,     # Target position for arm joint 1
    target_arm_el0,     # Target position for arm joint 2
    target_arm_el1,     # Target position for arm joint 3
    target_arm_wr0,     # Target position for arm joint 4
    target_arm_wr1,     # Target position for arm joint 5
    target_velocity_x,  # Target forward velocity
    target_velocity_y,  # Target lateral velocity
    target_angular_vel, # Target angular velocity
]
```

## Implementation Guide

### Step 1: Spot Robot Connection

```python
from bosdyn.client import create_standard_sdk
from bosdyn.client.lease import LeaseClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient

def _init_spot_robot(self):
    # Create SDK
    sdk = create_standard_sdk('spot-data-collector')
    
    # Connect to robot
    robot = sdk.create_robot('ROBOT_IP')
    robot.authenticate('USERNAME', 'PASSWORD')
    robot.time_sync.wait_for_sync()
    
    # Acquire lease
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease = lease_client.acquire()
    
    # Get clients
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    return robot, lease, robot_state_client, robot_command_client
```

### Step 2: Read Robot State

```python
def get_robot_state(self):
    # Get robot state
    robot_state = self.robot_state_client.get_robot_state()
    
    # Extract arm joint positions (radians)
    arm_joints = []
    joint_names = ['arm_sh0', 'arm_sh1', 'arm_el0', 'arm_el1', 'arm_wr0', 'arm_wr1']
    for joint_name in joint_names:
        joint_state = robot_state.kinematic_state.joint_states[joint_name]
        arm_joints.append(joint_state.position.value)
    
    # Extract base velocities
    velocity_state = robot_state.kinematic_state.velocity_of_body_in_odom
    base_velocity = [
        velocity_state.linear.x,   # Forward velocity
        velocity_state.linear.y,   # Lateral velocity
        velocity_state.angular.z,  # Angular velocity
    ]
    
    # Combine into qpos
    qpos = np.array(arm_joints + base_velocity, dtype=np.float64)
    
    # Get velocities (joint velocities + base accelerations)
    arm_velocities = []
    for joint_name in joint_names:
        joint_state = robot_state.kinematic_state.joint_states[joint_name]
        arm_velocities.append(joint_state.velocity.value)
    
    # Base acceleration (could compute from velocity history)
    base_accel = [0.0, 0.0, 0.0]  # Or implement proper differentiation
    
    qvel = np.array(arm_velocities + base_accel, dtype=np.float64)
    
    return qpos, qvel
```

### Step 3: Capture Camera Images

```python
from bosdyn.client.image import ImageClient, build_image_request

def get_camera_images(self):
    image_client = self.robot.ensure_client(ImageClient.default_service_name)
    
    # Request images
    image_requests = [
        build_image_request('hand_color_image', quality_percent=100),
        build_image_request('frontleft_fisheye_image', quality_percent=100),  # or appropriate scene cam
    ]
    
    image_responses = image_client.get_image(image_requests)
    
    # Decode hand camera
    hand_image = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8)
    hand_image = cv2.imdecode(hand_image, cv2.IMREAD_COLOR)
    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
    hand_image = cv2.resize(hand_image, (640, 480))
    
    # Decode scene camera
    head_image = np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8)
    head_image = cv2.imdecode(head_image, cv2.IMREAD_COLOR)
    head_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB)
    head_image = cv2.resize(head_image, (640, 480))
    
    return {
        'hand_rgb': hand_image,
        'head_rgb': head_image,
    }
```

## Data Collection Best Practices

### 1. Episode Quality
- **Successful episodes**: Only save episodes that successfully complete the task
- **Diverse starting conditions**: Vary object positions, robot orientations
- **Smooth trajectories**: Avoid jerky movements during teleoperation
- **Appropriate speed**: Maintain consistent control rate (50Hz recommended)

### 2. Dataset Size
- **Minimum**: 50 episodes for simple tasks
- **Recommended**: 100-200 episodes for robust policies
- **Complex tasks**: 300+ episodes

### 3. Data Diversity
Vary:
- Object positions (within reachable workspace)
- Object orientations
- Starting robot configuration
- Lighting conditions
- Background clutter

### 4. Camera Considerations
- **Hand camera**: Ensure gripper/object interactions are visible
- **Scene camera**: Capture enough spatial context for navigation
- **Lighting**: Consistent lighting across episodes
- **Resolution**: 640x480 is a good balance (can adjust if needed)
- **Synchronization**: Ensure cameras are time-synchronized

### 5. Control Rate
- **Recommended**: 50 Hz (0.02s per timestep)
- **Minimum**: 20 Hz
- **Higher rates**: Better for fast movements but larger data files

### 6. Episode Length
- Should cover entire task from start to finish
- Add buffer time at start/end (robot stationary)
- Typical: 200-400 timesteps for pick-and-place

## Troubleshooting

### Issue: Images are blurry
- Check camera focus settings
- Reduce motion blur by slowing down movements
- Increase shutter speed if possible

### Issue: State readings are noisy
- Add low-pass filtering to joint readings
- Average over small time windows (2-3 samples)
- Check encoder calibration

### Issue: Large file sizes
- Reduce image resolution (e.g., 320x240)
- Use fewer timesteps per episode
- Enable HDF5 compression (with trade-off in loading speed)

### Issue: Cameras not synchronized
- Use hardware-triggered cameras if available
- Record timestamps and interpolate during post-processing
- Ensure consistent image capture rate

### Issue: Incomplete episodes
- Always save episode length metadata
- Implement graceful handling of early termination
- Pad with zeros if necessary (and use `is_pad` mask during training)

## Post-Collection Checklist

- [ ] Verify all episodes with `verify_spot_data.py`
- [ ] Check that state dimensions are correct (9 DOF)
- [ ] Confirm both cameras are present in all episodes
- [ ] Review sample videos for quality
- [ ] Check for static episodes (no movement)
- [ ] Ensure actions are in correct format (target states)
- [ ] Verify dataset statistics (mean, std, ranges)
- [ ] Back up raw data before training

## Next Steps

After data collection:

1. **Verify data**: Use `verify_spot_data.py`
2. **Update config**: Set `DATA_DIR` in `constants.py`
3. **Train policy**: Run `imitate_episodes.py`
4. **Evaluate**: Test learned policy on robot
5. **Iterate**: Collect more data if needed

## Example: Complete Workflow

```bash
# 1. Collect 50 episodes
python record_spot_episodes.py \
    --dataset_dir ./data/spot_pick_place \
    --num_episodes 50 \
    --episode_len 400

# 2. Verify dataset
python verify_spot_data.py --dataset_dir ./data/spot_pick_place

# 3. Update constants.py
# Set DATA_DIR = './data'

# 4. Train ACT policy
python imitate_episodes.py \
    --task_name spot_mobile_manipulation \
    --ckpt_dir ./checkpoints/spot_v1 \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0

# 5. Evaluate
python imitate_episodes.py --eval \
    --task_name spot_mobile_manipulation \
    --ckpt_dir ./checkpoints/spot_v1 \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0 \
    --temporal_agg
```

## Additional Resources

- **Spot SDK Documentation**: https://dev.bostondynamics.com/
- **ACT Paper**: https://arxiv.org/abs/2304.13705
- **Original ACT Repo**: https://github.com/tonyzhaozh/act

## Support

For issues specific to:
- **Spot Robot API**: Consult Boston Dynamics documentation
- **ACT Training**: See `SPOT_ADAPTATION.md` and original ACT docs
- **Data Format**: Check `verify_spot_data.py` output

