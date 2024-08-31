from legged_gym.envs.go1.go1_config  import Go1RoughCfg, Go1RoughCfgPPO

class Go1FlatCfg( Go1RoughCfg ):
    class env( Go1RoughCfg.env ):
        num_observations = 48
  
    class terrain( Go1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
  
    class asset( Go1RoughCfg.asset ):
        save_error = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( Go1RoughCfg.rewards ):
        max_contact_force = 350.
        base_height_target = 0.35
        class scales ( Go1RoughCfg.rewards.scales ):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.
            # feet_contact_forces = -0.01
    
    class commands( Go1RoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( Go1RoughCfg.commands.ranges ):
            ang_vel_yaw = [-1.5, 1.5]


class Go1FlatCfgPPO( Go1RoughCfgPPO ):
    class policy( Go1RoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]

    class runner ( Go1RoughCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_go1_saver' if(Go1FlatCfg.asset.save_error) else 'flat_go1'
        save_error = False
        load_run = -1
        max_iterations = 500