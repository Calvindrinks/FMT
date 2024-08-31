from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_onnx, export_policy_as_jit, export_mmpolicy_as_jit, task_registry, mm_task_registry, Logger
import time
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt

import numpy as np
import torch


def play(args):
    
    sleep_time = 0.015
    if args.mm:
        env_cfg, train_cfg = mm_task_registry.get_cfgs(name=args.task)
    else:
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        print(f"View with sleep time: {sleep_time} seconds")
    # override some parameters for testing
    if args.headless:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 24)
    else:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 12)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # load policy
    train_cfg.runner.resume = True
    
    # high/low speed commands groups for survive experiment.
        # env_cfg.commands.survive_exp = True
    env_cfg.commands.ranges.lin_vel_x = [0.7, 0.9] # min max [m/s]
    env_cfg.commands.ranges.ang_vel_yaw = [0, 0]   # min max [m/s]
    
    # prepare environment
    if args.mm:
        env, _ = mm_task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        ppo_runner, train_cfg = mm_task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)
    else:
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)
    obs = env.get_observations()
    
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
    img_idx = 0

    actions = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device, requires_grad=False)
    start_steps = 430
    sampling_steps = 120
    actions_buf = torch.zeros(sampling_steps, env.num_envs, 8)
    states_buf = torch.zeros(sampling_steps, env.num_envs, 16)

    for i in range(2 * int(env.max_episode_length)):

        if not args.headless:
            time.sleep(sleep_time)

        actions = policy(obs.detach())
        
        # stand with initial state
        # actions.fill_(0)
        if i >= start_steps and i < start_steps + sampling_steps and DRAW_TSNE:
            actions_buf[i-start_steps].copy_(actions.detach() * 0.25)
            joint_states = obs[:, 6:22].detach() # 4, 16
            states_buf[i-start_steps].copy_(joint_states)
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_direction = np.array([0., 2., 1.3])
            z_direction = np.array([0., 0., 0.1])
            root_state = env.root_states[5, :3].cpu().numpy()
            camera_position = root_state + 0.25 * camera_direction - z_direction
            env.set_camera(camera_position, camera_position - camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
    
    if DRAW_TSNE:
        tsne = TSNE(n_jobs=4)
        fault_group_num = 4
        fault_group = torch.div(torch.arange(env.num_envs, device=env.device), (env.num_envs/fault_group_num), rounding_mode='floor').to(torch.long)
        print("group", fault_group)
        colors_group = ["lightblue", "lightblue", "pink", "orange"]
        
        actions_buf = actions_buf.transpose(0, 1) # num_envs, sampling_steps, num_actions
        states_buf = states_buf.transpose(0, 1) 
        
        concate_steps = 1
        PIC_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pictures", "tsne")
        name_list = ["tsne-actions", "tsne-faultLeg-actions", "tsne-states", "tsne-faultLeg-states"]
        if concate_steps > 1:
            for i in range(len(name_list)):
                name_list[i] += f"-{concate_steps}"
        
        actions_name = name_list[0]
        concate_actions_buf = actions_buf.view(env.num_envs, sampling_steps//concate_steps, concate_steps, 8)
        torch.save(concate_actions_buf, os.path.join(PIC_DIR, actions_name + ".pt"))
        print(f"actions_buf shape: {actions_buf.shape}")
        for i in range(fault_group_num):
            actions_group = concate_actions_buf[fault_group==i]
            embeddings = tsne.fit_transform(actions_group.flatten(0, 1).flatten(-2, -1))
            vis_x = embeddings[:, 0]
            vis_y = embeddings[:, 1]
            plt.scatter(vis_x, vis_y, c=colors_group[i], cmap=plt.cm.get_cmap("jet", 10), marker='.')
            # plt.colorbar(ticks=range(10))
            plt.clim(-0.5, 9.5)
        plt.savefig(os.path.join(PIC_DIR, actions_name + ".png"))
        plt.clf()
        
        faultleg_actions_name = name_list[1]
        concate_leg_actions_buf = actions_buf[..., :2].view(env.num_envs, sampling_steps//concate_steps, concate_steps, 2)
        torch.save(concate_leg_actions_buf, os.path.join(PIC_DIR, faultleg_actions_name + ".pt"))
        for i in range(fault_group_num):
            actions_group = concate_leg_actions_buf[fault_group==i]
            if concate_steps==1:
                embeddings = actions_group.flatten(0, 1).flatten(-2, -1)
            else:
                embeddings = tsne.fit_transform(actions_group.flatten(0, 1).flatten(-2, -1))
            vis_x = embeddings[:, 0]
            vis_y = embeddings[:, 1]
            plt.scatter(vis_x, vis_y, c=colors_group[i], cmap=plt.cm.get_cmap("jet", 10), marker='.')
            # plt.colorbar(ticks=range(10))
            plt.clim(-0.5, 9.5)
        plt.savefig(os.path.join(PIC_DIR, faultleg_actions_name + ".png"))
        plt.clf()
        
        states_name = name_list[2]
        states_buf = states_buf.view(env.num_envs, sampling_steps//concate_steps, concate_steps, 16)
        torch.save(states_buf, os.path.join(PIC_DIR, states_name + ".pt"))
        print(f"states_buf shape: {states_buf.shape}")
        for i in range(1, fault_group_num):
            states_group = states_buf[fault_group==i]
            embeddings = tsne.fit_transform(states_group.flatten(0, 1).flatten(-2, -1))
            vis_x = embeddings[:, 0]
            vis_y = embeddings[:, 1]
            plt.scatter(vis_x, vis_y, c=colors_group[i], cmap=plt.cm.get_cmap("jet", 10), marker='.')
            plt.clim(-0.5, 9.5)
        plt.savefig(os.path.join(PIC_DIR, states_name + ".png"))
        plt.clf()
        
        faultleg_states_name = name_list[3]
        concate_states_buf = states_buf[..., :2].view(env.num_envs, sampling_steps//concate_steps, concate_steps, 2)
        torch.save(concate_states_buf, os.path.join(PIC_DIR, faultleg_states_name + ".pt"))
        for i in range(1, fault_group_num):
            states_group = concate_states_buf[fault_group==i]
            if concate_steps==1:
                embeddings = states_group.flatten(0, 1).flatten(-2, -1)
            else:
                embeddings = tsne.fit_transform(states_group.flatten(0, 1).flatten(-2, -1))
            vis_x = embeddings[:, 0]
            vis_y = embeddings[:, 1]
            plt.scatter(vis_x, vis_y, c=colors_group[i], cmap=plt.cm.get_cmap("jet", 10), marker='.')
            plt.clim(-0.5, 9.5)
        plt.savefig(os.path.join(PIC_DIR, faultleg_states_name + ".png"))
        plt.clf()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    
    DRAW_TSNE = True
    args.load_run = "newbest"
    play(args)