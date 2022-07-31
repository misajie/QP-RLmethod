# DQN
import os
import logging
from typing import List, Tuple
import numpy as np
from copy import deepcopy
from types import MethodType
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import init
from Agents.layers import MultiLayerPerceptron
from utils.ReplayBuffer import ReplayBuffer

# from layers import MultiLayerPerceptron
# from ReplayBuffer import ReplayBuffer

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("RLmain")


def get_optim_param(optim):  # optim = torch.optim.Adam(network_param, learning_rate)
    params_list = list()
    for params_dict in optim.state_dict()['state'].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


def linear_layer_init(layer: nn.Linear, init_w: float = 3e-1) -> nn.Linear:
    layer.weight.data.uniform_(-init_w, init_w)  # build-in random method
    layer.bias.data.uniform_(-init_w, init_w)
    return layer


class Qnet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, action_dim):
        super(Qnet, self).__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.net = MultiLayerPerceptron(input_size - 1, (hidden_size, hidden_size, hidden_size // 2), dropout=0.2,
                                        output_layer=True)

    def forward(self, x):
        # Set initial states
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, dim=0)
        ws_tag = x[:, :, -1]
        x = x[:, :, :-1]
        # pred = []
        # for i in range(self.action_dim):
        #     fc_out = self.net(x[:, i, :])
        #     fc_out = fc_out * ws_tag[:, [i]]
        #     pred.append(torch.squeeze(fc_out))
        # return torch.stack(pred, dim=-1)
        out = (torch.squeeze(self.net(x), -1) + 1) * ws_tag
        return torch.squeeze(out, dim=-1)


class DQNAgent:
    def __init__(self, env,
                 memory_size: int,
                 batch_size: int,
                 hidden_dim: int,
                 explore_rate: float,
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
        self.explore_rate = explore_rate
        self.tau = tau
        self.soft_update_freq = soft_update_freq

        self.critic = Qnet(self.state_dim, hidden_dim, self.action_dim).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-3)
        self.critic_optimizer.parameters = MethodType(get_optim_param, self.critic_optimizer)

        self.transition = dict()
        self.total_step = 0
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        actions = self.critic(torch.FloatTensor(state).to(self.device))
        selected_action = torch.argmax(actions, dim=-1).cpu().detach().numpy()[0]
        if not self.is_test:
            if np.random.rand() < self.explore_rate:
                selected_action = np.random.choice(np.where(state[:, -1] > 0)[0])
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
        q_lossls = []
        for i in range((self.memory.count() // self.batch_size) + 1):
            cur_transition = self.memory.sample_batch(self.batch_size)
            state, action, reward, nstate, mask = self.collate_fn(cur_transition)
            q_pred = self.critic(state).gather(1, action.long())
            # Test: actions are all in dropset, checked!
            # with torch.no_grad():
            #     state = state.cpu().numpy()
            #     action = action.long().cpu().numpy()
            #     for i in range(self.batch_size):
            #         drop_set = np.where(state[i, :, -1] > 0)
            #         print(drop_set,action[i])
            # break
            # print(self.critic(state).size(),self.critic_target(nstate).max(dim=1).values.size(),self.critic_target(nstate).max(dim=1).values)
            # TODO: why all 0 prediction!跟最大行为带入有关，为何有大量的0预测，明明都在drop set里面了，因为数据不是agent探索的？
            q_target = reward + self.gamma * torch.unsqueeze(self.critic_target(nstate).max(dim=1).values, -1) * mask
            q_loss = F.mse_loss(q_pred, q_target.detach())
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()
            # update target networks
            self._target_soft_update(self.tau)
            self.total_step += 1
            q_lossls.append(q_loss.item())
        return np.mean(q_lossls), np.array([0])

    def collate_fn(self, data):
        state = torch.FloatTensor([s[0] for s in data]).reshape(-1, self.action_dim, self.state_dim).to(self.device)
        action = torch.FloatTensor([s[1] for s in data]).reshape(-1, 1).to(self.device)
        reward = torch.FloatTensor([s[2] for s in data]).reshape(-1, 1).to(self.device)
        nstate = torch.FloatTensor([s[3] for s in data]).reshape(-1, self.action_dim, self.state_dim).to(self.device)
        done = torch.FloatTensor([1 - s[4] for s in data]).reshape(-1, 1).to(self.device)
        return state, action, reward, nstate, done

    def _target_soft_update(self, tau: float):
        if self.total_step % self.soft_update_freq == 0:
            for t_param, l_param in zip(self.critic_target.parameters(), self.critic.parameters()):
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


if __name__ == '__main__':
    from qpenv.ASenv_fixed_dqn import ASenv

    env = ASenv(buffer_size=100)
    agent = DQNAgent(env,
                     memory_size=100,
                     batch_size=32,
                     hidden_dim=16,
                     explore_rate=0.15,
                     gamma=0.99,
                     tau=5e-1,  # soft target update
                     soft_update_freq=2)
    for i in range(100):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            # action = env.random_action(state)  # test score:-6.61, average niter:13.37
            # action = env.as_action()    # AS policy:test score:-5.94, average niter:12.7
            nstate, reward, done, _ = env.step(action)
            state = nstate
            if done:
                print("done", env.cur_episode.niter)
                break
