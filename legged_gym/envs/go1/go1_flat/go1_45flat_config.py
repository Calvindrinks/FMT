from legged_gym.envs.go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO

class Go1WoLVelCfg( Go1RoughCfg ):
    class env( Go1RoughCfg.env ):
        num_observations = 45
        num_privileged_obs = 48
        without_linear_vel = True
        without_angular_vel = False
  
    class terrain( Go1RoughCfg.terrain ):
        mesh_type = 'trimesh'
        measure_heights = False

        terrain_proportions = [0.5, 0.5, 0, 0, 0]
        # dynamic_friction = 0.
  
    class asset( Go1RoughCfg.asset ):
        save_error = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( Go1RoughCfg.rewards ):
        max_contact_force = 350.
        base_height_target = 0.3
        class scales ( Go1RoughCfg.rewards.scales ):
            orientation = -1.0
            # torques = -2e-4
            energy = -1e-4
            feet_air_time = 0.
            base_height = -10. # -1.5
            action_rate = -0.1
            dof_pos_limits = -0.0
            hip_pos = -0.5
            # action_bias = -0.0005
            default_pos = -0.04
            feet_contact_forces = -0.01
            
            alive = 1.0
    
    class commands( Go1RoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( Go1RoughCfg.commands.ranges ):
            lin_vel_x = [0.4, 1.0] # min max [m/s]
            ang_vel_yaw = [-1, 1]

class Go1WoLVelCfgPPO( Go1RoughCfgPPO ):
    class policy( Go1RoughCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class runner ( Go1RoughCfgPPO.runner):
        run_name = 'trimesh_go1'
        experiment_name = 'flat_go1_WL_saver' if(Go1WoLVelCfg.asset.save_error) else 'flat_go1_WL'
        save_error = False
        load_run = -1
        max_iterations = 1500