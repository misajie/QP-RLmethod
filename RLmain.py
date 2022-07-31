import os
import random
import numpy as np
import torch
from train.run import train, test
from train.config import Arguments

SEED = 2022
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def train_and_test(train_size=1000, test_size=1000):
    from qpenv.ASenv import ASenv
    from Agents.DDPG import DDPGAgent
    args = Arguments()
    args.hidden_dim = 256
    args.memory_size = 10000
    args.batch_size = 512
    args.init_episode = 2000
    args.buffer_size = train_size
    env = ASenv(mpc_path="./benchmarks/normal/", buffer_size=args.buffer_size)
    # 注意test env专门的数据集
    test_env = ASenv("./benchmarks/normal/", buffer_size=test_size)
    save_dir = "./chkpt/mpcnorm_neg_{}M_{}B".format(args.memory_size, args.batch_size)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    agent = DDPGAgent(env,
                      memory_size=args.memory_size,
                      batch_size=args.batch_size,
                      hidden_dim=args.hidden_dim,
                      num_layers=args.num_layers,
                      num_classes=args.num_classes,
                      noise_mu=args.noise_mu,
                      noise_gamma=args.noise_gamma,
                      gamma=args.gamma,
                      tau=args.tau,
                      soft_update_freq=args.soft_update_freq
                      )
    print(agent.actor)
    print(agent.critic)
    # agent.save_or_load_agent(save_dir, if_save=False)
    # train(env, env, agent, args.epoch, args.init_episode, save_dir)
    agent.save_or_load_agent(save_dir, if_save=False)
    test(test_env, agent, test_size)


def train_and_test_fix(train_size=100, test_size=100):
    from qpenv.ASenv_fixed import ASenv
    from Agents.DDPG_fixdim import DDPGAgent
    args = Arguments()
    args.buffer_size = train_size
    args.init_episode = 100
    args.batch_size = 32
    env = ASenv(mpc_path="./benchmarks/normal/", buffer_size=args.buffer_size)
    # 注意test env专门的数据集
    test_env = ASenv("./benchmarks/normal/", buffer_size=test_size)
    save_dir = "./chkpt/mpcfixdim_{}M_{}B".format(args.memory_size, args.batch_size)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    agent = DDPGAgent(env,
                      memory_size=args.memory_size,
                      batch_size=args.batch_size,
                      hidden_dim=args.hidden_dim,
                      num_classes=args.num_classes,
                      noise_mu=args.noise_mu,
                      noise_gamma=args.noise_gamma,
                      gamma=args.gamma,
                      tau=args.tau,
                      soft_update_freq=args.soft_update_freq
                      )
    print(agent.actor)
    print(agent.critic)
    # train(env, env, agent, args.epoch, args.init_episode, save_dir)
    agent.save_or_load_agent(save_dir, if_save=False)
    test(test_env, agent, test_size)


def train_and_test_fix2(train_size=100, test_size=100):
    from qpenv.ASenv_fixed_case2 import ASenv
    from Agents.DDPG_fix2 import DDPGAgent
    args = Arguments()
    args.buffer_size = train_size
    env = ASenv(qp_type=1, mpc_path="./benchmarks/normal/", buffer_size=args.buffer_size)
    # 注意test env专门的数据集
    test_env = ASenv(1, "./benchmarks/normal/", buffer_size=test_size)
    save_dir = "./chkpt/mpcfix2_{}M_{}B".format(args.memory_size, args.batch_size)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    agent = DDPGAgent(env,
                      memory_size=args.memory_size,
                      batch_size=args.batch_size,
                      hidden_dim=args.hidden_dim,
                      # num_layers,
                      # num_classes,
                      noise_mu=args.noise_mu,
                      noise_gamma=args.noise_gamma,
                      gamma=args.gamma,
                      tau=args.tau,
                      soft_update_freq=args.soft_update_freq
                      )
    print(agent.actor)
    print(agent.critic)
    # train(env, env, agent, args.epoch, args.init_episode, save_dir)
    # agent.save_or_load_agent(save_dir, if_save=False)
    test(test_env, agent, test_size)


def train_and_test_DQN(train_size=1000, test_size=1000):
    from qpenv.ASenv_fixed_dqn import ASenv
    from Agents.DQN_fixdim import DQNAgent
    args = Arguments()
    args.hidden_dim = 64
    args.memory_size = 5000
    args.batch_size = 512
    args.init_episode = 500
    args.buffer_size = train_size
    env = ASenv(mpc_path="./benchmarks/normal/", buffer_size=args.buffer_size)
    # 注意test env专门的数据集
    test_env = ASenv(mpc_path="./benchmarks/normal/", buffer_size=test_size)
    save_dir = "./chkpt/mpcdqn_{}M_{}B".format(args.memory_size, args.batch_size)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    agent = DQNAgent(env,
                     memory_size=args.memory_size,
                     batch_size=args.batch_size,
                     hidden_dim=args.hidden_dim,
                     explore_rate=0.125,
                     gamma=args.gamma,
                     tau=args.tau,
                     soft_update_freq=args.soft_update_freq
                     )
    print(agent.critic)
    # agent.save_or_load_agent(save_dir, if_save=False)
    # train(env, env, agent, args.epoch, args.init_episode, save_dir)
    agent.save_or_load_agent(save_dir, if_save=False)
    test(test_env, agent, test_size)


if __name__ == '__main__':
    # train_and_test(1000, 500)
    train_and_test_DQN(10,500)
    # train_and_test_fix(1000, 500)
    # train_and_test_fix2(1000, 500)
