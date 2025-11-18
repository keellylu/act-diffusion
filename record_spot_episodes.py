"""
Data collection script for Spot robot pick-and-place tasks.
Saves episodes in HDF5 format compatible with ACT training pipeline.

Usage:
    python record_spot_episodes.py \
        --dataset_dir ./data/spot_pick_place \
        --num_episodes 50 \
        --episode_len 400
"""

import time
import os
import numpy as np
import argparse
import h5py
from collections import deque
import cv2

# TODO: Import your Spot SDK and camera interfaces
# from bosdyn.client import create_standard_sdk
# from your_spot_interface import SpotRobotInterface


class SpotDataCollector:
    def __init__(self, dataset_dir, episode_len=400, camera_height=480, camera_width=640):
        """
        Initialize Spot data collector.
        
        Args:
            dataset_dir: Directory to save episodes
            episode_len: Maximum timesteps per episode
            camera_height: Image height
            camera_width: Image width
        """
        self.dataset_dir = dataset_dir
        self.episode_len = episode_len
        self.camera_height = camera_height
        self.camera_width = camera_width
        
        # State dimension: 6 arm joints + 3 base velocities
        self.state_dim = 9
        
        # Camera names
        self.camera_names = ['hand_rgb', 'head_rgb']
        
        # Create dataset directory
        os.makedirs(dataset_dir, exist_ok=True)
        
        # TODO: Initialize Spot robot connection
        # self.robot = self._init_spot_robot()
        
        print(f"Data collector initialized")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Episode length: {episode_len}")
        print(f"State dimension: {self.state_dim}")
        print(f"Cameras: {self.camera_names}")
    
    def _init_spot_robot(self):
        """
        Initialize connection to Spot robot.
        TODO: Implement based on your Spot SDK setup.
        
        Returns:
            robot: Spot robot interface
        """
        # Example pseudo-code:
        # sdk = create_standard_sdk('spot-data-collector')
        # robot = sdk.create_robot('192.168.80.3')
        # robot.authenticate(username, password)
        # robot.time_sync.wait_for_sync()
        # return robot
        pass
    
    def get_robot_state(self):
        """
        Get current robot state (9 DOF).
        
        Returns:
            qpos: np.array of shape (9,) - [arm joints (6), base velocities (3)]
            qvel: np.array of shape (9,) - velocities corresponding to qpos
        
        TODO: Implement based on your Spot API.
        """
        # Example structure:
        # robot_state = self.robot.get_robot_state()
        
        # Arm joint positions (6 DOF)
        # arm_joints = [
        #     robot_state.arm_joint_states['arm_sh0'].position,
        #     robot_state.arm_joint_states['arm_sh1'].position,
        #     robot_state.arm_joint_states['arm_el0'].position,
        #     robot_state.arm_joint_states['arm_el1'].position,
        #     robot_state.arm_joint_states['arm_wr0'].position,
        #     robot_state.arm_joint_states['arm_wr1'].position,
        # ]
        
        # Base velocities (3 DOF: vx, vy, angular_velocity)
        # base_velocity = [
        #     robot_state.kinematic_state.velocity_of_body_in_odom.linear.x,
        #     robot_state.kinematic_state.velocity_of_body_in_odom.linear.y,
        #     robot_state.kinematic_state.velocity_of_body_in_odom.angular.z,
        # ]
        
        # Arm joint velocities (6 DOF)
        # arm_velocities = [
        #     robot_state.arm_joint_states['arm_sh0'].velocity,
        #     robot_state.arm_joint_states['arm_sh1'].velocity,
        #     robot_state.arm_joint_states['arm_el0'].velocity,
        #     robot_state.arm_joint_states['arm_el1'].velocity,
        #     robot_state.arm_joint_states['arm_wr0'].velocity,
        #     robot_state.arm_joint_states['arm_wr1'].velocity,
        # ]
        
        # Base acceleration as velocity derivative (3 DOF)
        # base_vel_derivatives = [0.0, 0.0, 0.0]  # Or compute from history
        
        # qpos = np.array(arm_joints + base_velocity)
        # qvel = np.array(arm_velocities + base_vel_derivatives)
        
        # Placeholder for now
        qpos = np.zeros(self.state_dim, dtype=np.float64)
        qvel = np.zeros(self.state_dim, dtype=np.float64)
        
        return qpos, qvel
    
    def get_camera_images(self):
        """
        Get images from both cameras.
        
        Returns:
            images: dict with keys ['hand_rgb', 'head_rgb']
                    each value is np.array of shape (H, W, 3) dtype uint8
        
        TODO: Implement based on your Spot camera API.
        """
        # Example pseudo-code:
        # image_client = self.robot.ensure_client('image')
        
        # Request images from both cameras
        # image_requests = [
        #     build_image_request('hand_color_image', quality_percent=100),
        #     build_image_request('frontleft_fisheye_image', quality_percent=100),  # or appropriate scene camera
        # ]
        
        # image_responses = image_client.get_image(image_requests)
        
        # hand_image = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8)
        # hand_image = cv2.imdecode(hand_image, cv2.IMREAD_COLOR)
        # hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
        # hand_image = cv2.resize(hand_image, (self.camera_width, self.camera_height))
        
        # head_image = np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8)
        # head_image = cv2.imdecode(head_image, cv2.IMREAD_COLOR)
        # head_image = cv2.cvtColor(head_image, cv2.COLOR_BGR2RGB)
        # head_image = cv2.resize(head_image, (self.camera_width, self.camera_height))
        
        # Placeholder: return dummy images
        images = {
            'hand_rgb': np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8),
            'head_rgb': np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8),
        }
        
        return images
    
    def record_episode(self, episode_idx, use_teleoperation=True):
        """
        Record one episode of pick-and-place task.
        
        Args:
            episode_idx: Episode number
            use_teleoperation: If True, collect via teleoperation. If False, use autonomous policy.
        
        Returns:
            success: Whether episode was successful
        """
        print(f"\n{'='*60}")
        print(f"Recording Episode {episode_idx}")
        print(f"{'='*60}")
        
        # Data buffers
        qpos_list = []
        qvel_list = []
        action_list = []
        image_dict = {cam_name: [] for cam_name in self.camera_names}
        
        # TODO: Reset robot to start position
        # self.reset_robot()
        
        print("Press ENTER to start recording...")
        input()
        
        print("Recording... Press Ctrl+C to stop early")
        start_time = time.time()
        
        try:
            for t in range(self.episode_len):
                timestep_start = time.time()
                
                # Get current state
                qpos, qvel = self.get_robot_state()
                images = self.get_camera_images()
                
                # Store observations
                qpos_list.append(qpos)
                qvel_list.append(qvel)
                for cam_name in self.camera_names:
                    image_dict[cam_name].append(images[cam_name])
                
                # Get action (either from teleoperation or autonomous policy)
                if use_teleoperation:
                    # TODO: Get action from teleop interface
                    # action = self.get_teleop_action()
                    action = qpos.copy()  # Placeholder
                else:
                    # TODO: Get action from your policy
                    # action = self.policy(qpos, images)
                    action = qpos.copy()  # Placeholder
                
                action_list.append(action)
                
                # TODO: Execute action on robot
                # self.execute_action(action)
                
                # Maintain control rate (e.g., 50 Hz = 0.02s)
                elapsed = time.time() - timestep_start
                sleep_time = max(0, 0.02 - elapsed)
                time.sleep(sleep_time)
                
                if t % 50 == 0:
                    print(f"  Timestep {t}/{self.episode_len}")
        
        except KeyboardInterrupt:
            print("\nEpisode stopped early by user")
        
        total_time = time.time() - start_time
        actual_timesteps = len(qpos_list)
        
        print(f"Episode completed: {actual_timesteps} timesteps in {total_time:.1f}s")
        
        # Ask if episode should be saved
        print("\nWas this episode successful? (y/n): ", end='')
        response = input().strip().lower()
        success = response == 'y'
        
        if not success:
            print("Episode marked as failed, not saving.")
            return False
        
        # Save episode
        self._save_episode(episode_idx, qpos_list, qvel_list, action_list, image_dict)
        
        return True
    
    def _save_episode(self, episode_idx, qpos_list, qvel_list, action_list, image_dict):
        """
        Save episode data to HDF5 file.
        
        Args:
            episode_idx: Episode number
            qpos_list: List of qpos arrays
            qvel_list: List of qvel arrays
            action_list: List of action arrays
            image_dict: Dict of camera_name -> list of images
        """
        # Convert lists to arrays
        qpos_arr = np.array(qpos_list, dtype=np.float64)
        qvel_arr = np.array(qvel_list, dtype=np.float64)
        action_arr = np.array(action_list, dtype=np.float64)
        
        max_timesteps = len(qpos_list)
        
        # Create HDF5 file
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_idx}.hdf5')
        
        print(f"Saving episode to {dataset_path}...")
        t0 = time.time()
        
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024**2*2) as root:
            # Set attributes
            root.attrs['sim'] = False  # Real robot data
            root.attrs['compress'] = False
            
            # Create observation group
            obs = root.create_group('observations')
            
            # Save robot state
            obs.create_dataset('qpos', data=qpos_arr)
            obs.create_dataset('qvel', data=qvel_arr)
            
            # Create images group and save camera data
            image_group = obs.create_group('images')
            for cam_name in self.camera_names:
                images = np.array(image_dict[cam_name], dtype=np.uint8)
                image_group.create_dataset(
                    cam_name, 
                    data=images,
                    dtype='uint8',
                    chunks=(1, self.camera_height, self.camera_width, 3),
                )
            
            # Save actions
            root.create_dataset('action', data=action_arr)
        
        save_time = time.time() - t0
        file_size_mb = os.path.getsize(dataset_path) / (1024**2)
        
        print(f"✓ Saved successfully!")
        print(f"  Time: {save_time:.1f}s")
        print(f"  Size: {file_size_mb:.1f} MB")
        print(f"  Timesteps: {max_timesteps}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Cameras: {len(self.camera_names)}")


def main(args):
    """Main data collection loop."""
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    episode_len = args['episode_len']
    start_idx = args['start_idx']
    
    # Initialize collector
    collector = SpotDataCollector(
        dataset_dir=dataset_dir,
        episode_len=episode_len,
    )
    
    # Collect episodes
    successes = []
    episode_idx = start_idx
    
    print("\n" + "="*60)
    print("SPOT ROBOT DATA COLLECTION")
    print("="*60)
    print(f"Target episodes: {num_episodes}")
    print(f"Starting from: episode_{start_idx}")
    print(f"Dataset dir: {dataset_dir}")
    print("="*60 + "\n")
    
    while len(successes) < num_episodes:
        try:
            success = collector.record_episode(episode_idx, use_teleoperation=True)
            
            if success:
                successes.append(1)
                print(f"✓ Episode {episode_idx} saved successfully")
                print(f"Progress: {len(successes)}/{num_episodes} successful episodes")
                episode_idx += 1
            else:
                print(f"✗ Episode {episode_idx} discarded")
            
            print("\nPress ENTER to continue to next episode, or Ctrl+C to quit...")
            input()
            
        except KeyboardInterrupt:
            print("\n\nData collection interrupted by user")
            break
    
    # Summary
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Total successful episodes: {len(successes)}")
    print(f"Success rate: {np.mean(successes)*100:.1f}%")
    print(f"Dataset saved to: {dataset_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect Spot robot demonstration data')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory to save episodes')
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='Number of episodes to collect')
    parser.add_argument('--episode_len', type=int, default=400,
                        help='Maximum timesteps per episode')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting episode index')
    
    main(vars(parser.parse_args()))

