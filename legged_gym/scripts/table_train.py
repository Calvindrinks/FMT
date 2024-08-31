import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

import time

env = None
env_cfg = None
ppo_runner = None
train_cfg = None

# def grid_search(args, x_values, iterations, embedd=False):
#     best_checkpoint = None
#     best_reward = -float('inf')

#     for x in x_values:
#         print(f"Training with x={x}")
#         train(args, iterations, x, embedd)
#         # Assuming that your train function returns the final reward and checkpoint
#         reward, checkpoint = evaluate(args)
#         if reward > best_reward:
#             best_reward = reward
#             best_checkpoint = checkpoint

#     return best_checkpoint

# x_values = [[-1, 1], [0, 1], [-1, 0], [-1, 1]]
# best_checkpoint = grid_search(args, x_values, 800, True)

# print(f"Best checkpoint: {best_checkpoint}")
# print("模型的最优参数：",model.best_params_)
# print("最优模型分数：",model.best_score_)
# print("最优模型对象：",model.best_estimator_)

# params = [
# 	{'kernel':['linear'],'C':[1,10,100,1000]},
# 	{'kernel':['poly'],'C':[1,10],'degree':[2,3]},
# 	{'kernel':['rbf'],'C':[1,10,100,1000], 
# 	 'gamma':[1,0.1, 0.01, 0.001]}]

def train(args, iterations, x, embedd=False):
    global env, env_cfg, ppo_runner, train_cfg
    if embedd:
        env, env_cfg = task_registry.make_env(name=args.task, args=args)
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.env.command_ranges["lin_vel_x"] = x
    train_cfg.runner.max_iterations = iterations
    dev = torch.tensor([0,1], dtype=torch.long).to(device="cuda:0")

    if embedd == False:
        ppo_runner.alg.actor_critic.froze_embedding()
    else:
        ppo_runner.alg.actor_critic.unfroze_embedding()
    print(ppo_runner.alg.actor_critic.embedding_layer(dev))
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    args.task = "solo9"
    args.run_name = 'e12'
    args.headless = True

    x = [-1, 1]
    train(args, 100, x, True)
    
    print("\n\nend")
    time.sleep(1)

    args.resume = True
    x = [0, 1]
    train(args, 500, x)

    x = [-1, 0]
    train(args, 500, x)

    x = [-1, 1]
    train(args, 600, x)