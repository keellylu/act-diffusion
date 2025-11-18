from numpy import np

def record_episode(self, episode_idx):
    # Initialize empty lists for this episode
    qpos_list = []          # Will store (T, 9) - one array per timestep
    qvel_list = []          # Will store (T, 9) - one array per timestep
    action_list = []        # Will store (T, 9) - one array per timestep
    
    # Dictionary for images (one list per camera)
    image_dict = {
        'hand_rgb': [],     # Will store (T, H, W, 3) - one image per timestep
        'head_rgb': [],     # Will store (T, H, W, 3) - one image per timestep
    }

    '''
    # qpos (9 values) - THE STATE
    qpos = [
        arm_sh0,        # arm joint position
        arm_sh1,        # arm joint position
        arm_el0,        # arm joint position
        arm_el1,        # arm joint position
        arm_wr0,        # arm joint position
        arm_wr1,        # arm joint position
        velocity_x,     # base velocity (yes, velocity is in "qpos"!)
        velocity_y,     # base velocity
        angular_vel,    # base velocity
    ]

    # qvel (9 values) - RATE OF CHANGE OF QPOS
    qvel = [
        arm_sh0_vel,    # d/dt of arm_sh0 (velocity)
        arm_sh1_vel,    # d/dt of arm_sh1 (velocity)
        arm_el0_vel,    # d/dt of arm_el0 (velocity)
        arm_el1_vel,    # d/dt of arm_el1 (velocity)
        arm_wr0_vel,    # d/dt of arm_wr0 (velocity)
        arm_wr1_vel,    # d/dt of arm_wr1 (velocity)
        accel_x,        # d/dt of velocity_x (acceleration)
        accel_y,        # d/dt of velocity_y (acceleration)
        angular_accel,  # d/dt of angular_vel (acceleration)
    ]
    '''

    # Loop for 400 timesteps (or until task complete)
    for t in range(400):
        # ===== GET CURRENT STATE =====
        qpos, qvel = self.get_robot_state()
        # qpos shape: (9,) - single timestep
        # qvel shape: (9,) - single timestep
        
        images = self.get_camera_images()
        # images = {
        #     'hand_rgb': (480, 640, 3),  # single frame
        #     'head_rgb': (480, 640, 3),  # single frame
        # }
        
        # ===== APPEND TO LISTS =====
        qpos_list.append(qpos)                      # Add this timestep's qpos
        qvel_list.append(qvel)                      # Add this timestep's qvel
        
        image_dict['hand_rgb'].append(images['hand_rgb'])  # Add this frame
        image_dict['head_rgb'].append(images['head_rgb'])  # Add this frame
        
        # ===== GET/EXECUTE ACTION =====
        action = self.get_action()  # From teleoperation or policy
        # action shape: (9,) - single timestep
        
        action_list.append(action)                  # Add this timestep's action
        
        # Execute action on robot...
        self.robot.execute(action)
        
        # Maintain control rate
        time.sleep(0.02)  # 50 Hz

    
    # ===== CONVERT LISTS TO ARRAYS =====
    qpos_arr = np.array(qpos_list, dtype=np.float64)
    # Shape: (T, 9) where T is number of timesteps
    
    qvel_arr = np.array(qvel_list, dtype=np.float64)
    # Shape: (T, 9)
    
    action_arr = np.array(action_list, dtype=np.float64)
    # Shape: (T, 9)
    
    # Images are already in correct format as list of arrays
    # Will convert when saving to HDF5

    dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_idx}.hdf5')
    
    with h5py.File(dataset_path, 'w') as root:
        # Set metadata
        root.attrs['sim'] = False  # Real robot data
        
        # Create observations group
        obs = root.create_group('observations')
        
        # Save qpos and qvel
        obs.create_dataset('qpos', data=qpos_arr)      # (T, 9)
        obs.create_dataset('qvel', data=qvel_arr)      # (T, 9)
        
        # Create images group
        image_group = obs.create_group('images')
        
        # Convert image lists to arrays and save
        for cam_name in ['hand_rgb', 'head_rgb']:
            images = np.array(image_dict[cam_name], dtype=np.uint8)
            # Shape: (T, 480, 640, 3)
            
            image_group.create_dataset(
                cam_name, 
                data=images,
                dtype='uint8',
                chunks=(1, 480, 640, 3),  # Chunk by frame for efficient access
            )
        
        # Save actions
        root.create_dataset('action', data=action_arr)  # (T, 9)