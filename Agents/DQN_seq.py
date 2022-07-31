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


def init_proc():
    log.debug(f"initializing {os.getpid()}")
    os.environ["OMP_NUM_THREADS"] = "1"


class Critic(nn.Module):  # Q net
    def __init__(self, input_size, hidden_size, num_layers):
        super(Critic, self).__init__()
        self.is_test = False
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.05,
                                  bidirectional=True)

        self.hidden = nn.Linear(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

        self.hidden = linear_layer_init(self.hidden)
        self.out = linear_layer_init(self.out)

    def forward(self, state: torch.Tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.is_test:
            h0 = torch.zeros(self.num_layers * 2, state.batch_sizes.numpy()[0], self.hidden_size).to(
                device)  # 2 for bidirection
            c0 = torch.zeros(self.num_layers * 2, state.batch_sizes.numpy()[0], self.hidden_size).to(device)

            out, _ = self.lstm(state, (h0, c0))  # out: tensor of shape (batch_size, seq_id, hidden_size*2)
            out, _ = pad_packed_sequence(out, batch_first=True)
        else:
            state = torch.unsqueeze(state, dim=0)
            h0 = torch.zeros(self.num_layers * 2, 1, self.hidden_size).to(device)  # 2 for bidirection
            c0 = torch.zeros(self.num_layers * 2, 1, self.hidden_size).to(device)
            out, _ = self.lstm(state, (h0, c0))
        # qs = []  # [b, seq, emb] --> q: [b,seq,1] --> q: [b,1],arg: [b,1]
        # for i in range(state.size(1)):
        #     hidden = F.relu(self.hidden(out[:, i, :]))
        #     qs.append(torch.squeeze(self.out(hidden)))
        # return qs  # cannot deal with dynamic seqlen
        print(out)
        print(out.size())

class DDPGAgent:
    def __init__(self, env,
                 memory_size: int,
                 batch_size: int,
                 hidden_dim: int,
                 num_layers: int,
                 gamma: float = 0.99,
                 tau: float = 5e-3,  # soft target update
                 soft_update_freq: int = 2
                 ):
        self.device = device
        self.env = env
        state_dim = self.env.get_statedim()
        action_dim = 1  # Question: 不等长action评价？
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.soft_update_freq = soft_update_freq

        self.critic = Critic(state_dim, hidden_dim, num_layers, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, hidden_dim, num_layers, action_dim).to(self.device)
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
        self.noise = Normal(ou_noise_theta * torch.ones(action_dim), ou_noise_gamma * torch.ones(action_dim))

        self.transition = dict()

        self.total_step = 0
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.actor.is_test = True
        action = self.actor(torch.FloatTensor(state).to(self.device))
        selected_action = action.cpu().detach().numpy()
        if not self.is_test:
            # attention: explore
            # noise = self.noise.sample().cpu().detach().numpy()
            # if np.random.rand() < 0.05:
            #     selected_action = np.clip(selected_action + 1, 0, state.shape[0]-1)
            selected_action = np.clip(selected_action + np.random.randint(-1, 2), 0, state.shape[0] - 1)
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
        for i in range((self.memory.count() // self.batch_size)*2 + 1):
            cur_transition = self.memory.sample_batch(self.batch_size)
            state, action, reward, nstate, mask = self.collate_fn(cur_transition)
            naction = torch.unsqueeze(self.actor_target(nstate), 1).to(torch.float32)
            # TODO: update critic slate-wise!
            q_pred = self.critic(state, action)
            q_target = reward + self.gamma * self.critic_target(nstate, naction) * mask
            q_loss = F.mse_loss(q_pred, q_target.detach())
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()

            # update actor: nabla_{\theta}\pi_{\theta}nabla_{a}Q_{\pi}(s,a)
            a_loss = -self.critic(state, torch.unsqueeze(self.actor(state), 1).to(torch.float32)).mean()
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
        state = [torch.FloatTensor(s[0]).to(self.device) for s in data]
        action = torch.FloatTensor([s[1] for s in data]).reshape(-1, 1).to(self.device)
        reward = torch.FloatTensor([s[2] for s in data]).reshape(-1, 1).to(self.device)
        nstate = [torch.FloatTensor(s[3]).to(self.device) for s in data]
        done = torch.FloatTensor([1 - s[4] for s in data]).reshape(-1, 1).to(self.device)
        seq_len, nseq_len = torch.LongTensor([f.shape[0] for f in state]), \
                            torch.LongTensor([f.shape[0] for f in nstate])
        state, nstate = pad_sequence(state, batch_first=True).float(), \
                        pad_sequence(nstate, batch_first=True).float()
        # features,labels = features.unsqueeze(-1),labels.unsqueeze(-1)
        state, nstate = pack_padded_sequence(state, seq_len, batch_first=True), \
                        pack_padded_sequence(nstate, nseq_len, batch_first=True, enforce_sorted=False)
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