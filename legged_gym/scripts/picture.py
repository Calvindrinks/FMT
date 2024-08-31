from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.num_envs = 4096
    # env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/solo9/urdf/solo9.urdf'
    env_cfg.sim.gravity = [0., 0., 0.]
    env_cfg.env.episode_length_s = 10 # episode length in seconds
    env_cfg.env.env_spacing = 1
    env_cfg.control.stiffness = {'spin':0., 'FE':0.}  # [N*m/rad]
    env_cfg.control.damping = {'spin':0., 'FE':0.}     # [N*m*s/rad]
    env_cfg.terrain.num_rows = 50
    env_cfg.terrain.num_cols = 5
    env_cfg.viewer.pos = np.array([30, -1, 1.2], dtype=np.float64)
    env_cfg.viewer.lookat = np.array([30, 30, -15], dtype=np.float64)
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    img_idx = 0
    actions = torch.zeros((env.num_envs, 9), device=env.device, requires_grad=False)

    for i in range(600*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    args = get_args()
    play(args)