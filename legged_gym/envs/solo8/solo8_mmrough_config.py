from .solo8_config import Solo8RoughCfg, Solo8RoughCfgPPO

class Solo8MMRoughCfg( Solo8RoughCfg ):
    class env( Solo8RoughCfg.env ):
        episode_length_s = 15 # episode length in seconds
    
    class terrain( Solo8RoughCfg.terrain ):
        measure_heights = True
    
    class asset( Solo8RoughCfg.asset ):
        weak_test = False
        weak_train = True
        limit_test = False # by timing
        limit_train = True # by environment

    class rewards(Solo8RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.232
        max_contact_force = 80.

        class scales(Solo8RoughCfg.rewards.scales):
            dof_pos_limits = -10.0
            orientation = -0.5 # less orientation
            base_height = -1.0 # less base height
            collision = -5.0
            torques = -1e-3
            tracking_ang_vel = 1.5
            tracking_lin_vel = 1.5
            action_bias = -0.  # disable action bias
            lin_vel_z = -1.0
            feet_air_time = 1.5
            foot_clearance = -0. # disable clearance
            foot_slip = -2.5
            feet_contact_forces = -0.01

    class commands( Solo8RoughCfg.commands ):
        heading_command = False
        resampling_time = 8.
        # num_commands = 3 # default: lin_vel_x, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        class ranges:
            lin_vel_x = [-0.0, 1.0] # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]


class Solo8MMRoughCfgPPO(Solo8RoughCfgPPO):
    runner_class_name = 'MultiPolicyRunner' #
    class policy():                         #
        init_noise_std = 1.0
        pre_actor_hidden_dims = [512]
        pre_critic_hidden_dims = [512]
        task_actor_hidden_dims = [256, 128]
        task_critic_hidden_dims = [256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    
    class runner( Solo8RoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'solo8_mmrough'
        max_iterations = 2000
        save_interval = 500 # check for potential saves every this many iterations

        policy_class_name = 'MMActorCritic' #
        algorithm_class_name = 'MMPPO'      #