from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_onnx, export_policy_as_jit, export_mmpolicy_as_jit, task_registry, mm_task_registry, Logger
import time

import numpy as np
import torch

def _reward_action_bias(actions):
    # Penalize changes in actions
    return torch.sum(torch.square(actions), dim=10)

def play(args):
    
    if args.mm:
        env_cfg, train_cfg = mm_task_registry.get_cfgs(name=args.task)
    else:
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    if TRUN_TO == "left": 
        env_cfg.commands.ranges.lin_vel_x = [0.8, 0.8] # min max [m/s]
        env_cfg.commands.ranges.ang_vel_yaw = [-0.3, -0.3]   # min max [m/s]
    elif TRUN_TO == "right":
        env_cfg.commands.ranges.lin_vel_x = [0.8, 0.8] # min max [m/s]
        env_cfg.commands.ranges.ang_vel_yaw = [0.3, 0.3]   # min max [m/s]
    else:
        env_cfg.commands.ranges.lin_vel_x = [0.45, 0.45] # min max [m/s]
        env_cfg.commands.ranges.ang_vel_yaw = [0, 0]   # min max [m/s]

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
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        if args.mm:
            export_mmpolicy_as_jit(ppo_runner.alg.actor_critic, path)
        else:
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            export_policy_as_onnx(ppo_runner.alg.actor_critic, path, env_cfg.env.num_observations)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    command_x_buf = torch.zeros(stop_state_log, env.num_envs)
    command_yaw_buf = torch.zeros(stop_state_log, env.num_envs)
    base_vel_x_buf = torch.zeros(stop_state_log, env.num_envs)
    base_vel_yaw_buf = torch.zeros(stop_state_log, env.num_envs)
    
    PIC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pictures", "velocity")
    name_list = ["command_x", "command_yaw", "base_vel_x", "base_vel_yaw"]
    if (TRUN_TO == "left" or TRUN_TO == "right"):
        for i in range(len(name_list)):
            name_list[i] += f"_{TRUN_TO}"

    # env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    for i in range(1*int(env.max_episode_length)):
        
        if torch.sum(env.time_out_buf):
            print(env.episode_length_buf, "episode")
        time.sleep(0.015)
        actions = policy(obs.detach())
        # actions.fill_(0)
        # print(actions.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # print(env.commands)
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            env.camera_tracking()

        if i < stop_state_log:
            command_x_buf[i].copy_(env.commands[:, 0])
            command_yaw_buf[i].copy_(env.commands[:, 1])
            base_vel_x_buf[i].copy_(env.base_lin_vel[:, 0])
            base_vel_yaw_buf[i].copy_(env.base_ang_vel[:, 2])

            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_yaw': env.commands[robot_index, 1].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
            
        elif i==stop_state_log:
            logger.plot_states()
            torch.save(command_x_buf, os.path.join(PIC_DIR, name_list[0] + ".pt"))
            torch.save(command_yaw_buf, os.path.join(PIC_DIR, name_list[1] + ".pt"))
            torch.save(base_vel_x_buf, os.path.join(PIC_DIR, name_list[2] + ".pt"))
            torch.save(base_vel_yaw_buf, os.path.join(PIC_DIR, name_list[3] + ".pt"))
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    TRUN_TO = "right"
    TRUN_TO = "forward"
    args = get_args()
    args.load_run = "newbest"
    play(args)