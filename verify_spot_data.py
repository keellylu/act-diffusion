"""
Verification script for Spot robot data.
Checks data format, visualizes episodes, and ensures compatibility with ACT training.

Usage:
    python verify_spot_data.py --dataset_dir ./data/spot_pick_place --episode_idx 0
"""

import os
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from visualize_episodes import save_videos
from constants import DT

def verify_episode(dataset_dir, episode_idx):
    """
    Verify and visualize a single episode.
    
    Args:
        dataset_dir: Directory containing episodes
        episode_idx: Episode index to verify
    """
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Episode file not found: {dataset_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Verifying Episode {episode_idx}")
    print(f"{'='*60}\n")
    
    try:
        with h5py.File(dataset_path, 'r') as root:
            # Check attributes
            is_sim = root.attrs.get('sim', False)
            print(f"Source: {'Simulation' if is_sim else 'Real Robot'}")
            
            # Check qpos
            qpos = root['/observations/qpos'][()]
            print(f"\n✓ qpos shape: {qpos.shape}")
            print(f"  Expected: (T, 9) for Spot robot")
            if qpos.shape[1] != 9:
                print(f"  ⚠️  WARNING: Expected 9 DOF, got {qpos.shape[1]}")
            
            # Check qvel
            qvel = root['/observations/qvel'][()]
            print(f"\n✓ qvel shape: {qvel.shape}")
            if qvel.shape != qpos.shape:
                print(f"  ⚠️  WARNING: qvel shape doesn't match qpos")
            
            # Check actions
            action = root['/action'][()]
            print(f"\n✓ action shape: {action.shape}")
            if action.shape != qpos.shape:
                print(f"  ⚠️  WARNING: action shape doesn't match qpos")
            
            # Check images
            print(f"\n✓ Images:")
            image_dict = {}
            for cam_name in root[f'/observations/images/'].keys():
                images = root[f'/observations/images/{cam_name}'][()]
                image_dict[cam_name] = images
                print(f"  - {cam_name}: {images.shape}")
                print(f"    dtype: {images.dtype}, range: [{images.min()}, {images.max()}]")
            
            expected_cameras = ['hand_rgb', 'head_rgb']
            for cam in expected_cameras:
                if cam not in image_dict:
                    print(f"  ⚠️  WARNING: Missing camera '{cam}'")
            
            # Check temporal consistency
            T = qpos.shape[0]
            print(f"\n✓ Episode length: {T} timesteps ({T*DT:.1f} seconds @ {1/DT:.0f}Hz)")
            
            # Statistics
            print(f"\n{'='*60}")
            print("State Statistics (qpos)")
            print(f"{'='*60}")
            joint_names = ["arm_sh0", "arm_sh1", "arm_el0", "arm_el1", "arm_wr0", "arm_wr1",
                          "vel_x", "vel_y", "ang_vel"]
            for i, name in enumerate(joint_names):
                print(f"{name:12s}: mean={qpos[:, i].mean():7.3f}, "
                      f"std={qpos[:, i].std():7.3f}, "
                      f"range=[{qpos[:, i].min():7.3f}, {qpos[:, i].max():7.3f}]")
            
            # Action statistics
            print(f"\n{'='*60}")
            print("Action Statistics")
            print(f"{'='*60}")
            for i, name in enumerate(joint_names):
                print(f"{name:12s}: mean={action[:, i].mean():7.3f}, "
                      f"std={action[:, i].std():7.3f}, "
                      f"range=[{action[:, i].min():7.3f}, {action[:, i].max():7.3f}]")
            
            # Check for static episodes (no movement)
            qpos_std = qpos.std(axis=0)
            if np.all(qpos_std < 0.01):
                print(f"\n⚠️  WARNING: Episode appears to be static (no movement)")
            
            # Visualize
            print(f"\n{'='*60}")
            print("Generating Visualizations")
            print(f"{'='*60}")
            
            # Save video
            video_path = os.path.join(dataset_dir, f'episode_{episode_idx}_video.mp4')
            save_videos(image_dict, DT, video_path=video_path)
            
            # Plot joint trajectories
            plot_path = os.path.join(dataset_dir, f'episode_{episode_idx}_joints.png')
            plot_joint_trajectories(qpos, action, joint_names, plot_path)
            
            print(f"\n✅ Episode {episode_idx} verification complete!")
            return True
            
    except Exception as e:
        print(f"\n❌ Error reading episode: {e}")
        import traceback
        traceback.print_exc()
        return False


def plot_joint_trajectories(qpos, action, joint_names, plot_path):
    """Plot joint position and action trajectories."""
    num_joints = qpos.shape[1]
    fig, axs = plt.subplots(num_joints, 1, figsize=(12, 2*num_joints))
    
    for i in range(num_joints):
        ax = axs[i] if num_joints > 1 else axs
        ax.plot(qpos[:, i], label='State (qpos)', alpha=0.7)
        ax.plot(action[:, i], label='Action', alpha=0.7, linestyle='--')
        ax.set_ylabel(joint_names[i])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axs[-1].set_xlabel('Timestep')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Saved joint plot to: {plot_path}")
    plt.close()


def verify_dataset(dataset_dir, max_episodes=None):
    """
    Verify entire dataset.
    
    Args:
        dataset_dir: Directory containing episodes
        max_episodes: Maximum number of episodes to check (None = all)
    """
    print(f"\n{'='*60}")
    print(f"DATASET VERIFICATION")
    print(f"{'='*60}")
    print(f"Dataset directory: {dataset_dir}\n")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Find all episodes
    episode_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')])
    num_episodes = len(episode_files)
    
    print(f"Found {num_episodes} episodes\n")
    
    if num_episodes == 0:
        print("❌ No episodes found in dataset directory")
        return
    
    # Verify each episode
    if max_episodes:
        episode_files = episode_files[:max_episodes]
    
    valid_episodes = []
    episode_lengths = []
    
    for episode_file in episode_files:
        episode_idx = int(episode_file.split('_')[1].split('.')[0])
        
        try:
            dataset_path = os.path.join(dataset_dir, episode_file)
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                T = qpos.shape[0]
                state_dim = qpos.shape[1]
                
                # Quick check
                if state_dim != 9:
                    print(f"⚠️  Episode {episode_idx}: Incorrect state_dim={state_dim} (expected 9)")
                else:
                    valid_episodes.append(episode_idx)
                    episode_lengths.append(T)
                    print(f"✓ Episode {episode_idx}: {T} timesteps, {state_dim} DOF")
        
        except Exception as e:
            print(f"❌ Episode {episode_idx}: Error - {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total episodes: {num_episodes}")
    print(f"Valid episodes: {len(valid_episodes)}")
    print(f"Average length: {np.mean(episode_lengths):.1f} timesteps")
    print(f"Min/Max length: {np.min(episode_lengths)}/{np.max(episode_lengths)} timesteps")
    
    if len(valid_episodes) == num_episodes:
        print(f"\n✅ All episodes are valid and ready for training!")
    else:
        print(f"\n⚠️  Some episodes have issues. Check individual episodes for details.")


def main(args):
    dataset_dir = args['dataset_dir']
    
    if args['episode_idx'] is not None:
        # Verify single episode
        verify_episode(dataset_dir, args['episode_idx'])
    else:
        # Verify entire dataset
        verify_dataset(dataset_dir, max_episodes=args['max_episodes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify Spot robot data')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Dataset directory')
    parser.add_argument('--episode_idx', type=int, default=None,
                        help='Specific episode to verify (None = verify all)')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum episodes to check when verifying all')
    
    main(vars(parser.parse_args()))

