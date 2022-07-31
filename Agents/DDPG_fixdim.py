# DDPG
import os
import time
import random
import logging
from typing import List, Tuple
from multiprocessing.pool import Pool

import gym
import numpy as np
import matplotlib.pyplot as plot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

from utils.ReplayBuffer import ReplayBuffer

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("RLmain")


def linear_layer_init(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)  # build-in random method
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class Actor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Actor, self).__init__()
        self.is_test = False
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(input_size-1, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes, bias=False)
        self.fc1.weight = init.xavier_normal_(self.fc1.weight, gain=1)
        self.fc2.weight = init.xavier_normal_(self.fc2.weight, gain=1)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.Sigmoid()

    def forward(self, x):
        # Set initial states
        if self.is_test:
            x = torch.unsqueeze(x,dim=0)

        ws_tag = x[:, :, -1]
        x = x[:, :, :-1]
        pred = []
        for i in range(x.size(1)):
            out = self.activation1(self.fc1(x[:, i, :]))
            fc_out = -self.activation2(self.fc2(out))
            fc_out = fc_out * ws_tag[:, [i]]
            pred.append(torch.squeeze(fc_out))
        return torch.stack(pred, dim=-1)


class Critic(nn.Module):  # Q net
    def __init__(self, input_size, hidden_size, action_dim):
        super(Critic, self).__init__()
        self.is_test = False
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.action_dim = action_dim

        self.state_emb = nn.Linear(input_size-1, hidden_size)
        self.action_emb = nn.Linear(action_dim, hidden_size)
        self.hidden = nn.Linear(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

        self.state_emb = linear_layer_init(self.state_emb)
        self.action_emb = linear_layer_init(self.action_emb)
        self.hidden = linear_layer_init(self.hidden)
        self.out = linear_layer_init(self.out)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_emb(action)
        # TODO: broadcast
        ws_tag = state[:, :, [-1]]
        state = state[:, :, :-1]*ws_tag
        state_emb = self.state_emb(torch.mean(state, 1))
        hidden = F.relu(self.hidden(torch.cat([state_emb, action_emb], dim=-1)))
        out = self.out(hidden)
        return out


class DDPGAgent:
    def __init__(self, env,
                 memory_size: int,
                 batch_size: int,
                 hidden_dim: int,
                 num_classes: int,
                 noise_mu: float,
                 noise_gamma: float,
                 gamma: float = 0.99,
                 tau: float = 5e-3,  # soft target update
                 soft_update_freq: int = 2
                 ):
        self.device = device
        self.env = env
        self.action_dim, self.state_dim = self.env.get_dim()
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.soft_update_freq = soft_update_freq

        self.actor = Actor(self.state_dim, hidden_dim, num_classes).to(self.device)
        self.actor_target = Actor(self.state_dim, hidden_dim, num_classes).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(self.state_dim, hidden_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, hidden_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # TODO: 规范化noise，以及讲清楚代码原理
        # self.noise = OUNoise(
        # 	action_dim,
        # 	theta=ou_noise_theta,
        # 	gamma=ou_noise_gamma,
        # 	)

        def get_optim_param(optim):  # optim = torch.optim.Adam(network_param, learning_rate)
            params_list = list()
            for params_dict in optim.state_dict()['state'].values():
                params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
            return params_list

        from types import MethodType
        self.actor_optimizer.parameters = MethodType(get_optim_param, self.actor_optimizer)
        self.critic_optimizer.parameters = MethodType(get_optim_param, self.critic_optimizer)

        # TODO: 适合离散选择的noise；或者不同action维度的noise
        self.noise = Normal(noise_mu * torch.ones(self.action_dim), noise_gamma * torch.ones(self.action_dim))

        self.transition = dict()

        self.total_step = 0
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.actor.is_test = True
        action = self.actor(torch.FloatTensor(state).to(self.device))
        selected_action = action.cpu().detach().numpy()
        if not self.is_test:
            # attention: explore
            noise = self.noise.sample().cpu().detach().numpy()
            # if np.random.rand() < 0.05:
            #     selected_action = np.clip(selected_action + 1, 0, state.shape[0]-1)
            selected_action = np.clip(selected_action + noise, 0, 1)
            self.transition['state'] = state
            self.transition['action'] = selected_action
            # sum of errors对TD的解释
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        nstate, reward, done, _ = self.env.step(action)
        if not self.is_test:
            self.transition['nstate'] = nstate
            self.transition['reward'] = reward
            self.transition['done'] = done
            self.memory.add(**self.transition)
            self.transition = dict()
        return nstate, reward, done

    def update(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.actor.is_test = False
        a_lossls, q_lossls = [], []
        for i in range((self.memory.count() // self.batch_size) + 1):
            cur_transition = self.memory.sample_batch(self.batch_size)
            state, action, reward, nstate, mask = self.collate_fn(cur_transition)
            naction = self.actor_target(nstate)
            # TODO: update critic slate-wise!
            q_pred = self.critic(state, action)
            q_target = reward + self.gamma * self.critic_target(nstate, naction) * mask
            q_loss = F.mse_loss(q_pred, q_target.detach())
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()

            # update actor: nabla_{\theta}\pi_{\theta}nabla_{a}Q_{\pi}(s,a)
            a_loss = -self.critic(state, self.actor(state)).mean()
            if self.total_step % 4 == 0:
                self.actor_optimizer.zero_grad()
                a_loss.backward()
                self.actor_optimizer.step()

            # update target networks
            self._target_soft_update(self.tau)
            self.total_step += 1

            a_lossls.append(a_loss.item())
            q_lossls.append(q_loss.item())
        return np.mean(q_lossls), np.mean(a_lossls)

    def collate_fn(self, data):
        data.sort(key=lambda x: x[0].shape[0], reverse=True)
        state = torch.stack([torch.FloatTensor(s[0]).to(self.device) for s in data], dim=0)
        action = torch.FloatTensor([s[1] for s in data]).reshape(-1, self.action_dim).to(self.device)
        reward = torch.FloatTensor([s[2] for s in data]).reshape(-1, 1).to(self.device)
        nstate = torch.stack([torch.FloatTensor(s[3]).to(self.device) for s in data], dim=0)
        done = torch.FloatTensor([1 - s[4] for s in data]).reshape(-1, 1).to(self.device)
        return state, action, reward, nstate, done

    def _target_soft_update(self, tau: float):
        if self.total_step % self.soft_update_freq == 0:
            for t_param, l_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

            for t_param, l_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.actor), ('act_optim', self.actor_optimizer),
                         ('critic', self.critic), ('cri_optim', self.critic_optimizer), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None
