# Spot Robot Adaptation Guide

## Overview
The ACT policy has been adapted to support the Spot robot with:
- **9 DOF**: 6 arm joints + 3 base velocities (velocity_x, velocity_y, angular_velocity)
- **2 RGB cameras**: 
  - `hand_rgb`: Hand-mounted camera for close-up manipulation
  - `head_rgb`: Scene camera for spatial reference/context

## Changes Made

### 1. Architecture Changes (`detr/models/detr_vae.py`)
- **Removed hardcoded state dimension**: All hardcoded `14` values replaced with `state_dim` parameter
- **Made state_dim configurable**: Both `DETRVAE` and `CNNMLP` models now accept and use `state_dim` dynamically
- **Updated all projections**:
  - `input_proj_robot_state`: `Linear(state_dim, hidden_dim)` 
  - `encoder_action_proj`: `Linear(state_dim, hidden_dim)`
  - `encoder_joint_proj`: `Linear(state_dim, hidden_dim)`
  - `action_head`: `Linear(hidden_dim, state_dim)`
  - CNNMLP `mlp`: Input/output adjusted for `state_dim`

### 2. Training Script Changes (`imitate_episodes.py`)
- **Added `--state_dim` argument**: Command-line parameter with default value of 14
- **Priority order for state_dim**:
  1. Command-line argument (`--state_dim`)
  2. Task config (`task_config['state_dim']`)
  3. Default value (14 for backward compatibility)
- **Passes state_dim to policy_config**: Both ACT and CNNMLP policies receive state_dim

### 3. Constants and Configuration (`constants.py`)
- **New task config**: `spot_mobile_manipulation` with:
  - 2 cameras: `['hand_rgb', 'head_rgb']`
  - `state_dim`: 9
- **Added Spot constants**:
  - `SPOT_JOINT_NAMES`: 6 arm joint names
  - `SPOT_BASE_VELOCITY_NAMES`: 3 base velocity names

## Architecture Summary

### Image Processing Flow
```
2 RGB Images:
  - hand_rgb: Hand-mounted camera (close-up manipulation view)
  - head_rgb: Scene camera (spatial context/reference)
    ↓
ResNet18 Backbone (shared or separate per camera)
    ↓
Feature Maps: 512 × H' × W' each camera
    ↓
Concatenate along width: (2×W')×H'×512 or 600×H'×512
    ↓
Project to hidden_dim (e.g., 512): Conv2d
    ↓
Image Embeddings for Transformer
```

### State Processing Flow
```
Joint Positions (9D: 6 arm joints + 3 base velocities)
    ↓
Project to hidden_dim: Linear(9, 512)
    ↓
Proprioception Embeddings
```

### Decoder Output Flow
```
Transformer Decoder Output (hidden_dim=512)
    ↓
Action Head: Linear(512, 9)
    ↓
Predicted Actions (k×9: k timesteps, 9 DOF)
```

## Usage

### Training Command
```bash
python3 imitate_episodes.py \
    --task_name spot_mobile_manipulation \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0 \
    --state_dim 9
```

**Note**: If using the `spot_mobile_manipulation` task config, `--state_dim 9` is optional since it's specified in the config.

### Evaluation Command
```bash
python3 imitate_episodes.py \
    --eval \
    --task_name spot_mobile_manipulation \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --batch_size 8 \
    --dim_feedforward 3200 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --seed 0 \
    --state_dim 9 \
    --temporal_agg  # optional: for smoother execution
```

## Data Format

Your HDF5 episode files should contain:
- `/observations/qpos`: Shape `(T, 9)` - 6 arm joints + 3 base velocities
- `/observations/images/hand_rgb`: Shape `(T, H, W, 3)` - Hand-mounted camera (close-up view)
- `/observations/images/head_rgb`: Shape `(T, H, W, 3)` - Scene camera (spatial reference)
- `/action`: Shape `(T, 9)` - Target actions (same format as qpos)

Where T is the episode length.

## Action Space Definition

The 9-dimensional action/state vector for Spot:
1. **Arm Joints (6)**: `[arm_sh0, arm_sh1, arm_el0, arm_el1, arm_wr0, arm_wr1]`
2. **Base Velocities (3)**: `[velocity_x, velocity_y, angular_velocity]`

## Backward Compatibility

All existing configurations (14 DOF bimanual robot) continue to work without modification:
- Default `state_dim=14` maintained
- Original task configs unchanged
- No breaking changes to existing code

## Custom State Dimensions

To adapt for other robots, simply:
1. Add a new task config in `constants.py` with your `state_dim`
2. Specify camera names in the config
3. Run training with your task name

