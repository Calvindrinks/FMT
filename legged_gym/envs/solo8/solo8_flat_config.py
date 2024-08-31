from .solo8_config import Solo8RoughCfg, Solo8RoughCfgPPO

class Solo8FlatCfg( Solo8RoughCfg ):
    class env( Solo8RoughCfg.env ):
        num_observations = 30 # 6(base angular vel, projected gravity) + 3 + 8*3 + 0(base height)
        with_angular_vel = False
        num_actions = 8
        episode_length_s = 8 # episode length in seconds
        num_envs = 4096

    # class domain_rand:
        # reflection_rate = 0.
    #     randomize_base_mass = True
    #     added_mass_range = [-0.1, 0.1]
    #     push_robots = True
    
    class terrain( Solo8RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class asset( Solo8RoughCfg.asset ):
        weak_test = False
        weak_train = False

        limit_test = False # by timing
        limit_train = False # by environment
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards(Solo8RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.232
        max_contact_force = 80.

        class scales(Solo8RoughCfg.rewards.scales):
            dof_pos_limits = -10.0
            orientation = -2.5
            base_height = -10.0
            collision = -5.0
            torques = -1e-3
            tracking_ang_vel = 1.5
            tracking_lin_vel = 1.5
            action_bias = -0.01
            lin_vel_z = -1.0
            feet_air_time = 1.5
            foot_clearance = -0.5
            foot_slip = -2.5
            feet_contact_forces = -0.01

    class commands( Solo8RoughCfg.commands ):
        survive_exp = False
        heading_command = False
        resampling_time = 8.
        # num_commands = 3 # default: lin_vel_x, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        class ranges:
            lin_vel_x = [-0.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]


class Solo8FlatCfgPPO(Solo8RoughCfgPPO):
    class policy( Solo8RoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]

    class runner( Solo8RoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'solo8_flat'
        max_iterations = 1000
