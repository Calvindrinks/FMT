# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2023 Fudan Univercity, Calvin Hou

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class MMActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        pre_actor_hidden_dims=[128],
                        pre_critic_hidden_dims=[128],
                        task_actor_hidden_dims=[64, 32],
                        task_critic_hidden_dims=[64, 32],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("MMActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(MMActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        pre_actor_layers = []
        pre_actor_layers.append(nn.Linear(mlp_input_dim_a, pre_actor_hidden_dims[0]))
        pre_actor_layers.append(activation)
        task_actor_layers = []
        task_actor_layers.append(nn.Linear(pre_actor_hidden_dims[-1], task_actor_hidden_dims[0]))
        task_actor_layers.append(activation)
        for l in range(len(task_actor_hidden_dims)):
            if l == len(task_actor_hidden_dims) - 1:
                task_actor_layers.append(nn.Linear(task_actor_hidden_dims[l], num_actions))
            else:
                task_actor_layers.append(nn.Linear(task_actor_hidden_dims[l], task_actor_hidden_dims[l + 1]))
                task_actor_layers.append(activation)
        self.pre_actor = nn.Sequential(*pre_actor_layers)
        self.task1_actor = nn.Sequential(*task_actor_layers)
        self.task2_actor = nn.Sequential(*task_actor_layers)
        self.task3_actor = nn.Sequential(*task_actor_layers)
        # self.actor = self.task_actor(self.pre_actor)
        # Value function
        pre_critic_layers = []
        pre_critic_layers.append(nn.Linear(mlp_input_dim_c, pre_critic_hidden_dims[0]))
        pre_critic_layers.append(activation)
        task_critic_layers = []
        task_critic_layers.append(nn.Linear(pre_critic_hidden_dims[-1], task_critic_hidden_dims[0]))
        task_critic_layers.append(activation)
        for l in range(len(task_critic_hidden_dims)):
            if l == len(task_critic_hidden_dims) - 1:
                task_critic_layers.append(nn.Linear(task_critic_hidden_dims[l], 1))
            else:
                task_critic_layers.append(nn.Linear(task_critic_hidden_dims[l], task_critic_hidden_dims[l + 1]))
                task_critic_layers.append(activation)
        self.pre_critic = nn.Sequential(*pre_critic_layers)
        self.task1_critic = nn.Sequential(*task_critic_layers)
        self.task2_critic = nn.Sequential(*task_critic_layers)
        self.task3_critic = nn.Sequential(*task_critic_layers)

        print(f"Pre_Actor MLP: {self.pre_actor}")
        print(f"Task_Actor MLP: {self.task1_actor}")
        print(f"Pre_Critic MLP: {self.pre_critic}")
        print(f"Task_Critic MLP: {self.task1_critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # sklf.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

        task_discriminator_layers = []
        task_discriminator_layers.append(nn.Linear(mlp_input_dim_a, 1))
        task_discriminator_layers.append(nn.Tanh())
        self.task_discriminator = nn.Sequential(*task_discriminator_layers)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.act_inference(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        hiddens = self.pre_actor(observations)
        task1_hidden = hiddens[0:int(0.5*len(observations))]
        task2_hidden = hiddens[int(0.5*(len(observations))):int(0.75*len(observations))]
        task3_hidden = hiddens[int(0.75*len(observations)):]
        print(task1_hidden.shape)
        task1_mean = self.task1_actor(task1_hidden)
        task2_mean = self.task2_actor(task2_hidden)
        task3_mean = self.task3_actor(task3_hidden)
        actions_mean = torch.cat((task1_mean, task2_mean, task3_mean), dim=0)
        return actions_mean

    def discriminator_inference(self, observations):
        hiddens = self.pre_actor(observations)
        if (self.task_discriminator(hiddens) < -0.33):
            actions_mean = self.task1_actor(hiddens)
        elif (self.task_discriminator(hiddens) < 0.33):
            actions_mean = self.task2_actor(hiddens)
        else:
            actions_mean = self.task3_actor(hiddens)
        return actions_mean
    
    def evaluate(self, critic_observations, **kwargs):
        hiddens = self.pre_critic(critic_observations)
        task1_hidden = hiddens[0:int(0.5*len(critic_observations))]
        task2_hidden = hiddens[int(0.5*(len(critic_observations))):int(0.75*len(critic_observations))]
        task3_hidden = hiddens[int(0.75*len(critic_observations)):]
        task1_value = self.task1_critic(task1_hidden)
        task2_value = self.task2_critic(task2_hidden)
        task3_value = self.task3_critic(task3_hidden)
        value = torch.cat((task1_value, task2_value, task3_value), dim=0)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None