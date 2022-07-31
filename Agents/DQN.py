# DDPG
import os
import time
import random
import logging
from copy import deepcopy
from typing import List, Tuple
from types import MethodType
import gym
import numpy as np
import matplotlib.pyplot as plot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from utils.ReplayBuffer import ReplayBuffer

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("RLmain")


def linear_layer_init(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)  # build-in random method
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


def init_proc():
    log.debug(f"initializing {os.getpid()}")
    os.environ["OMP_NUM_THREADS"] = "1"


class Qnet(nn.Module):  # Q net, cannot handle different seqlen batch
    def __init__(self, state_dim, hid_dim):
        super(Qnet, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hid_dim), nn.ReLU(),
                                 nn.Linear(hid_dim, hid_dim), nn.ReLU(),
                                 nn.Linear(hid_dim, hid_dim), nn.Hardswish(),
                                 nn.Linear(hid_dim, 1))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                linear_layer_init(layer)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.size())==2:
            state = torch.unsqueeze(state, 0)
        m = state.size(1)
        actions = []
        for i in range(m):
            actions.append(self.net(state[:, i, :]))
        return torch.stack(actions, -1).argmax(dim=-1)


class DQNAgent:
    def __init__(self, env,
                 memory_size: int,
                 batch_size: int,
                 hidden_dim: int,
                 gamma: float = 0.99,
                 tau: float = 5e-3,  # soft target update
                 soft_update_freq: int = 2
                 ):
        self.device = device
        self.env = env
        self.explore_rate = 0.125
        self.action_dim, self.state_dim = self.env.get_dim()
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.soft_update_freq = soft_update_freq

        self.critic = Qnet(self.state_dim, hidden_dim).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        def get_optim_param(optim):  # optim = torch.optim.Adam(network_param, learning_rate)
            params_list = list()
            for params_dict in optim.state_dict()['state'].values():
                params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
            return params_list

        self.critic_optimizer.parameters = MethodType(get_optim_param, self.critic_optimizer)

        self.transition = dict()

        self.total_step = 0
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        ten_state = torch.as_tensor(state, dtype=torch.float32)
        m = state.shape[0]
        with torch.no_grad():
            if random.random() < self.explore_rate:  # epsilon-greedy
                a_ints = torch.randint(m, size=(1,))  # choosing action randomly
            else:

        selected_action = a_ints.detach().cpu().numpy()
        if not self.is_test:
            self.transition['state'] = state
            self.transition['action'] = selected_action
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
        a_lossls, q_lossls = [], []
        for i in range((self.memory.count() // self.batch_size) + 1):
            cur_transition = self.memory.sample_batch(self.batch_size)
            state, action, reward, nstate, mask = self.collate_fn(cur_transition)
            print(state.size(), action.size())
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

        name_obj_list = [('critic', self.critic), ('cri_optim', self.critic_optimizer), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None
