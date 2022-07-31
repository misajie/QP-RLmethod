import sys
sys.path.append("/usr/project/rlqp_train/rlqp_benchmarks")
sys.path.append("/usr/project/rlqp_train/rlqp_train")

from rlqp_train.qp_env import QPEnv
import numpy as np
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--save_dir", metavar="DIR", required=True, type=str, help="Directory to save/load checkpoints")
# hparams = parser.parse_args()

# define hyper params
# save_dir = hparams.save_dir
save_dir = "./tmp/"
qp_env = ["Random QP:10:50"]
qp_eps = 1e-6            # terminal epsilon
qp_step_reward = -1      # reward for each step
qp_iters_per_step = 200  # Number of QP ADMM (internal) iterations per adaptation #

env = QPEnv(
    eps=qp_eps,
    step_reward=qp_step_reward,
    iterations_per_step=qp_iters_per_step)

for e in qp_env:
    name, min_dim, max_dim = e.split(':')
    env.add_benchmark_problem_class(name, int(min_dim), int(max_dim))
print(env.problems)
if __name__ == '__main__':
    print(env.action_space)
    print(env.observation_space)
    ep_no = 1  # iterative random seed
    rng = np.random.default_rng(ep_no)

    # Generate a new episode
    episode = env.new_episode(ep_no, rng=rng)
    obs, done, ep_log, ep_ret = episode.get_obs(), False, [], 0
    print(obs)
