from legged_gym.envs import Go1RoughCfg, Go1RoughCfgPPO

class Go1WoHighCfg( Go1RoughCfg ):
    class env( Go1RoughCfg.env ):
        num_observations = 42
        num_privileged_obs = 48
        without_linear_vel = True
        without_angular_vel = True
  
    class terrain( Go1RoughCfg.terrain ):
        mesh_type = 'trimesh'
        measure_heights = False
        terrain_proportions = [0.5, 0.5, 0, 0, 0]
  
    class asset( Go1RoughCfg.asset ):
        save_error = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( Go1RoughCfg.rewards ):
        max_contact_force = 350.
        base_height_target = 0.35
        class scales ( Go1RoughCfg.rewards.scales ):
            orientation = -5.0
            torques = -0.000025
            base_height = -0.5 
            # feet_air_time = 2.
            feet_contact_forces = -0.01
            action_rate = -0.05
            dof_pos_limits = -0.0
            hip_pos = -0.5
            action_bias = -0.0005
            feet_contact_forces = -0.01
    
    class commands( Go1RoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( Go1RoughCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]
class Go1WoHighCfgPPO( Go1RoughCfgPPO ):
    class policy( Go1RoughCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class runner ( Go1RoughCfgPPO.runner):
        run_name = 'trimesh_go1_wo_base'
        experiment_name = 'flat_go1_WH_saver' if(Go1WoHighCfg.asset.save_error) else 'flat_go1_WH'
        save_error = False
        load_run = -1
        max_iterations = 1500