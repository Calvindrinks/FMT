from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_onnx, export_policy_as_jit, export_mmpolicy_as_jit, task_registry, mm_task_registry, Logger
import time

import numpy as np
import torch


def secondtrain(args):
    
    if args.mm:
        env_cfg, train_cfg = mm_task_registry.get_cfgs(name=args.task)
    else:
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 12)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Random time fault settings
    env_cfg.asset.weak_train = False
    env_cfg.asset.limit_train = False
    if FAULT_TYPE == "weak":
        env_cfg.asset.weak_test = True
    if FAULT_TYPE == "limit":
        env_cfg.asset.limit_test = True
    
    # load policy
    train_cfg.runner.resume = True
    
    # prepare environment
    if args.mm:
        env, _ = mm_task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        ppo_runner, train_cfg = mm_task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    else:
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    obs = env.get_observations()
    
    policy = ppo_runner.get_inference_policy(device=env.device)

    ppo_runner.second_learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        time.sleep(0.015)
        actions = policy(obs.detach())
        # print(actions.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        print(env.commands)
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

if __name__ == '__main__':
    MOVE_CAMERA = False
    # args.load_run = "newbest"
    # args.load_run = "mix"
    
    DRAW_SURVIVE = True
    FAULT_TYPE = "limit"
    DISCRIMINATOR = True
    
    args = get_args()
    secondtrain(args)
  
# from legged_gym import LEGGED_GYM_ROOT_DIR
# import os

# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import  get_args, export_policy_as_onnx, export_policy_as_jit, export_mmpolicy_as_jit, task_registry, mm_task_registry, Logger
# import time
# from MulticoreTSNE import MulticoreTSNE as TSNE
# from matplotlib import pyplot as plt

# import numpy as np
# import torch


# def play(args):
    
#     sleep_time = 0.015
#     if args.mm:
#         env_cfg, train_cfg = mm_task_registry.get_cfgs(name=args.task)
#     else:
#         env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#         print(f"View with sleep time: {sleep_time} seconds")
#     # override some parameters for testing
#     if args.headless:
#         env_cfg.env.num_envs = min(env_cfg.env.num_envs, 3200)
#     else:
#         env_cfg.env.num_envs = min(env_cfg.env.num_envs, 32)
#     env_cfg.terrain.num_rows = 5
#     env_cfg.terrain.num_cols = 5
#     env_cfg.terrain.curriculum = False
#     env_cfg.noise.add_noise = False
#     env_cfg.domain_rand.randomize_friction = False
#     env_cfg.domain_rand.push_robots = False

#     # load policy
#     train_cfg.runner.resume = True
    
#     # high/low speed commands groups for survive experiment.
#     if DRAW_SURVIVE == True:
#         env_cfg.commands.survive_exp = True
#     else:
#         print("Setting no command stairs, all random")
    
    
#     # prepare environment
#     if args.mm:
#         env, _ = mm_task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#         ppo_runner, train_cfg = mm_task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#         policy = ppo_runner.get_selected_inference_policy(device=env.device)
#     else:
#         env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#         ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#         policy = ppo_runner.get_inference_policy(device=env.device)
#     obs = env.get_observations()
    
#     # export policy as a jit module (used to run it from C++)
#     if EXPORT_POLICY:
#         path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
#         if args.mm:
#             export_mmpolicy_as_jit(ppo_runner.alg.actor_critic, path)
#         else:
#             export_policy_as_jit(ppo_runner.alg.actor_critic, path)
#             export_policy_as_onnx(ppo_runner.alg.actor_critic, path, env_cfg.env.num_observations)
#         print('Exported policy as jit script to: ', path)

#     logger = Logger(env.dt)
#     robot_index = 0 # which robot is used for logging
#     joint_index = 1 # which joint is used for logging
#     stop_state_log = 100 # number of steps before plotting states
#     stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
#     img_idx = 0

#     env.camera_actor = 3
#     actions = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device, requires_grad=False)

#     for i in range(10*int(env.max_episode_length)):

#         if not args.headless:
#             time.sleep(sleep_time)

#         if args.mm and DISCRIMINATOR: 
#             actions[~env.weak_fail_env_ids] = policy(
#                 obs[~env.weak_fail_env_ids].detach(), labels="health")
#             actions[env.weak_fail_env_ids] = policy(
#                 obs[env.weak_fail_env_ids].detach(), labels=FAULT_TYPE)
#         elif args.mm and not DISCRIMINATOR:
#             actions = policy(obs.detach(), labels=FAULT_TYPE)
#         else:
#             actions = policy(obs.detach())
        
#         # stand with initial state
#         # actions.fill_(0)
#         obs, _, rews, dones, infos = env.step(actions.detach())
#         if RECORD_FRAMES:
#             if i % 2:
#                 filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
#                 env.gym.write_viewer_image_to_file(env.viewer, filename)
#                 img_idx += 1 
#         if MOVE_CAMERA:
#             env.camera_tracking()

#         if i < stop_state_log:
#             logger.log_states(
#                 {
#                     'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
#                     'dof_pos': env.dof_pos[robot_index, joint_index].item(),
#                     'dof_vel': env.dof_vel[robot_index, joint_index].item(),
#                     'dof_torque': env.torques[robot_index, joint_index].item(),
#                     'command_x': env.commands[robot_index, 0].item(),
#                     'command_y': env.commands[robot_index, 1].item(),
#                     'command_yaw': env.commands[robot_index, 2].item(),
#                     'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
#                     'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
#                     'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
#                     'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
#                     'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
#                 }
#             )
#         elif i==stop_state_log:
#             logger.plot_states()
#         if  0 < i < stop_rew_log:
#             if infos["episode"]:
#                 num_episodes = torch.sum(env.reset_buf).item()
#                 if num_episodes>0:
#                     logger.log_rewards(infos["episode"], num_episodes)
#         elif i==stop_rew_log:
#             logger.print_rewards()
    
#     if DRAW_SURVIVE:
#         commands_group_num = 2
#         groups_num = 32
#         commands_group = torch.div(torch.arange(env.num_envs, device=env.device), (env.num_envs//groups_num), rounding_mode='floor').to(torch.long)
#         survive_rates_buf = torch.zeros(groups_num)
        
#         for i in range(groups_num // commands_group_num):
#             survive_rate = torch.sum(env.time_out_buf[commands_group==i]) / (env.num_envs // groups_num)
#             print(f"low speed{i}'s survive rate: {survive_rate}")
#             survive_rates_buf[i].copy_(survive_rate)
#         print(f"low speed mean survive rate: {torch.mean(survive_rates_buf[:groups_num // commands_group_num])}")
        
#         for i in range(groups_num // commands_group_num, groups_num):
#             survive_rate = torch.sum(env.time_out_buf[commands_group==i]) / (env.num_envs // groups_num)
#             print(f"high speed{i}'s survive rate: {survive_rate}")
#             survive_rates_buf[i].copy_(survive_rate)
#         print(f"high speed mean survive rate: {torch.mean(survive_rates_buf[groups_num // commands_group_num:])}")
        
#         if args.mm and DISCRIMINATOR:
#             if args.load_run == "no_Reflect":
#                 survive_data_name = f"./legged_gym/tests/pictures/survive/sur_{FAULT_TYPE}_wo_reflect.pt"
#             else:
#                 survive_data_name = f"./legged_gym/tests/pictures/survive/sur_{FAULT_TYPE}_ours.pt"
#         elif args.mm and not DISCRIMINATOR:
#             survive_data_name = f"./legged_gym/tests/pictures/survive/sur_{FAULT_TYPE}_taskpolicy.pt"
#         else:
#             if args.load_run == "mix":
#                 survive_data_name = f"./legged_gym/tests/pictures/survive/sur_{FAULT_TYPE}_mixTrain_normal.pt"
#             else:
#                 survive_data_name = f"./legged_gym/tests/pictures/survive/sur_{FAULT_TYPE}_normal.pt"
        
#         # save data in .pt
#         torch.save(survive_rates_buf, survive_data_name)

# if __name__ == '__main__':
#     EXPORT_POLICY = False
#     RECORD_FRAMES = False
#     MOVE_CAMERA = True
#     args = get_args()
    
#     DRAW_SURVIVE = True
#     FAULT_TYPE = "limit"
#     DISCRIMINATOR = True
#     args.load_run = "newbest"
#     # args.load_run = "mix"

#     play(args)