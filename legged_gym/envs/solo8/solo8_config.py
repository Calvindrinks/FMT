from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Solo8RoughCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_observations = 217 # 6(base angular vel, projected gravity) + 3 + 8*3 + 0(base height)
        with_angular_vel = False
        num_actions = 8
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.25] # x,y,z [m]
        
        footZoffset = 0.016
        maxfh = 0.073

        joint_index = {
            # symmetrical
            "FL_HFE": 0,   
            "FL_KFE": 1,
            "FR_HFE": 2,
            "FR_KFE": 3,

            "HL_HFE": 4,
            "HL_KFE": 5,
            "HR_HFE": 6,
            "HR_KFE": 7
        }

        default_joint_angles = { # = target angles [rad] when action = 0.0
            # symmetrical
            "FL_HFE": 0.7854,
            "HL_HFE": -0.7854,
            "FR_HFE": 0.7854,
            "HR_HFE": -0.7854,

            "FL_KFE": -1.5708,
            "HL_KFE": 1.5708,
            "FR_KFE": -1.5708,
            "HR_KFE": 1.5708,
        }

    class commands( LeggedRobotCfg.commands ):
        num_commands = 3
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'HFE': 3, 'KFE': 3}  # [N*m/rad]
        damping = {'HFE': 0.3, 'KFE': 0.3}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class domain_rand( LeggedRobotCfg.domain_rand ):
        reflection_rate = 0.
        randomize_friction = True
        friction_range = [0.6, 1.5]
        randomize_base_mass = True
        added_mass_range = [-0.2, 1]

    class asset( LeggedRobotCfg.asset ):
        save_error = False
        colors = True
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/solo/urdf/solo8.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/solo8_v7/solo8.urdf'
        name = "solo_robot"  # actor name
        foot_name = "FOOT"
        penalize_contacts_on = ["LEG"]
        terminate_after_contacts_on = ["torso"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.232
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

class Solo8RoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'solo8'
        max_iterations = 1500
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01 # ?