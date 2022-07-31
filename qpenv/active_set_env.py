# active set environment
# TODO:
# 1. deal with inf constraints
# 2. 处理无约束情况
# 3. osqp dual 和 active 的关系推导

import sys
sys.path.append("./")
sys.path.append("../loas/")

import numpy as np
import scipy.sparse as spa
import warnings
from solvers.LSsolver import LPSolver
import logging
# from benchmarks.benchmark_problems.control_example import MPC0QPExample
from benchmarks.problem_classes.random_qp import RandomQPExample

log = logging.getLogger("active_set_env")
warnings.filterwarnings("ignore")

Y_MIN, Y_MAX = -1e6, 1e6
AX_MIN, AX_MAX = -1e6, 1e6


class Box:
    def __init__(self, low, high, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = self.low.shape
        self.size = self.low.size
        self.dtype = dtype

    def sample(self, rng):
        return rng.uniform(low=self.low, high=self.high, size=self.shape).astype(self.dtype)


class DiscreteBox:
    def __init__(self, low, high, dtype=np.int):
        self.low = low
        self.high = high
        self.shape = self.low.shape
        self.size = self.low.size
        self.dtype = dtype

    def sample(self, rng):
        return rng.randint(low=self.low, high=self.high, size=self.shape).astype(self.dtype)


# maybe completed!
class QPEpisode:
    def __init__(self, P, q, A, lx, ux, step_reward):
        self.m = lx.shape[0]
        initial_a = [2 for i in range(self.m)]  # TODO: configurable.

        # print(sum(initial_a)/2)
        # 1: left active; 0: inactive; 2: right active;
        # !! how should the agent deal with case 2? rule based action according to observation?
        # print("initial as:",initial_a)
        # print("Problem dimension:", P.shape, q.shape, A.shape, lx.shape, ux.shape)
        self.P, self.q, self.A, self.lx, self.ux = P, q, A, lx, ux

        # process 1
        self.cur_A, self.cur_l, self.cur_u, keep_ls = self.process_as1(initial_a)
        self.step_reward = step_reward

        # use LP solver to solve the equalQP problem
        self.solver = LPSolver()
        self.solver.cscsetup(self.P, self.q, self.cur_A, self.cur_l, self.cur_u)
        self.x, y = self.solver.cscsolve()
        self.y = np.array([0 for i in range(self.m)])
        self.y[keep_ls] = y

        # !! TODO: check primal/dual results
        # print(self.result.x,self.result.y)
        # Ax = self.cur_A.dot(self.result.x)
        # print(sum(Ax-self.l>1e-6), sum(self.u-Ax>1e-6))  # result is strange!
        # Ax = self.cur_A.dot(self.result.x)
        # print(Ax)
        # PX_q = self.P.dot(self.result.x)+self.q  # actually 0, only float error
        # print(PX_q)

    def process_as1(self, activeset):
        cur_A, cur_lx, cur_ux = [], [], []
        tmpA = self.A.toarray()
        keep_ls = []  # in method 2 to delete inactive constraints in A
        for i in range(self.m):
            active_flag = activeset[i]
            if active_flag == 1:
                cur_A.append(tmpA[i, :])
                cur_ux.append(self.lx[i])
                cur_lx.append(self.lx[i])
                keep_ls.append(i)
            elif active_flag == 2:
                cur_A.append(tmpA[i, :])
                cur_ux.append(self.ux[i])
                cur_lx.append(self.ux[i])
                keep_ls.append(i)
            elif self.ux[i] == self.lx[i]:  # deal with equal
                cur_A.append(tmpA[i, :])
                cur_ux.append(self.ux[i])
                cur_lx.append(self.lx[i])
                keep_ls.append(i)
            else:
                pass
        return spa.csc_matrix(np.array(cur_A)), np.array(cur_lx), np.array(cur_ux), keep_ls

    def process_as2(self, activeset):
        cur_A, cur_lx, cur_ux = self.A.copy(), self.lx.copy(), self.ux.copy()
        for i in range(self.m):
            active_flag = activeset[i]
            if active_flag == 1:
                cur_ux[i] = self.lx[i]
            elif active_flag == 2:
                cur_lx[i] = self.ux[i]
            elif active_flag == 0:
                cur_ux[i] = 0
                cur_lx[i] = 0
                cur_A[i, :] = 0
            else:
                pass
        return cur_A, cur_lx, cur_ux

    def get_obs(self):
        x = self.x.copy()  # primal solution
        y = self.y.copy()  # dual solution
        Ax = self.A.dot(x)
        # Px = self.P.dot(x)
        # Aty = self.A.T.dot(y)
        # dual_res = Px + q + Aty
        return np.stack([
            np.clip(Ax - self.lx, -1e6, 1e6),  # primal gaps,allow negative for inactive
            np.clip(self.ux - Ax, -1e6, 1e6),
            np.clip(y, -1e6, 1e6)  # dual variables
            # np.clip(dual_res,-1e6,1e6)
        ], axis=-1)

    # def get_obs_orig(self):
    #     lo = self.solver._model.z - self.lower_bound
    #     hi = self.upper_bound - self.solver._model.z
    #     return np.stack([
    #         np.log10(np.clip(np.minimum(lo, hi), 1e-8, 1e6)),
    #         np.clip(self.solver._model.y, Y_MIN, Y_MAX),
    #         np.clip(self.solver._model.z - self.solver._model.Ax, AX_MIN, AX_MAX),
    #         np.log10(self.solver._model.rho_vec)
    #     ], axis=-1)

    def done(self):
        return sum(self.y < 0) == 0

    def step(self, action):
        # process 1:
        self.cur_A, self.cur_l, self.cur_u, keep_ls = self.process_as1(action)
        # !! how to reuse setup? as a sequential qp
        self.solver.cscsetup(self.P, self.q, self.cur_A, self.cur_l, self.cur_u)
        self.x, y = self.solver.cscsolve()
        self.y = np.array([0 for i in range(self.m)])
        self.y[keep_ls] = y

        # TODO: process 2 distinguish

        # !! how to append the deleted constraint? maybe not necessary
        next_obs = self.get_obs()
        done = self.done()
        reward = self.step_reward * (not done)
        return next_obs, reward, done, {}


class BenchmarkGen:
    def __init__(self, problem_class, min_dim, max_dim):
        self.problem_class = problem_class
        self.min_dim = min_dim
        self.max_dim = max_dim

    def __call__(self, rng, step_reward):
        prob_dim = rng.integers(self.min_dim, self.max_dim, endpoint=True)
        log.debug(f"Generating QP {self.problem_class.__name__}, dim={prob_dim}")
        # qp = self.problem_class(prob_dim).qp_problem
        qp = self.problem_class(prob_dim).qp_problem
        return QPEpisode(qp['P'], qp['q'], qp['A'], qp['l'], qp['u'], step_reward)


class RandomMPCGen:
    def __init__(self):
        pass


class QPEnv:
    def __init__(self, step_reward):
        self.step_reward = step_reward
        self.problems = []
        self.observation_space = Box(
            # Ax-l,u-Ax,y
            low=np.array([-1e6, -1e6, -1e6], dtype=np.float32),
            high=np.array([1e6, 1e6, 1e6], dtype=np.float32))
        self.action_space = DiscreteBox(
            low=np.array([0], dtype=np.int),
            high=np.array([2], dtype=np.int))

        self.no = 0
        self.episode = self._random_episode()

    # TODO: different benchmark problems for training
    # def add_benchmark_problem_class(self, name, min_dim, max_dim):
    #     log.info(f"Adding {name} problem class with dimension range=[{min_dim}, {max_dim}]")
    #     problem_class = EXAMPLES_MAP[name]
    #     self.problems.append(
    #         BenchmarkGen(problem_class, min_dim, max_dim))

    def _random_episode(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(seed=self.no)

        # TODO: change qp classes
        # idx = rng.integers(0, 10, 1)
        # mpceg = MPC0QPExample(idx).qp_problem
        mpceg = RandomQPExample(5).qp_problem
        P, q, A, l, u = mpceg['P'], mpceg['q'], mpceg['A'], mpceg['l'], mpceg['u']
        return QPEpisode(P, q, A, l, u, step_reward=self.step_reward)

    def reset(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(seed=self.no)
        self.episode = self._random_episode(rng)
        self.no += 1
        # while episode.done():
        #     log.debug("  already solved")
        #     episode = self._random_episode(no, rng)
        return self.episode.get_obs()

    def step(self, action):
        next_obs, reward, done, info = self.episode.step(action)
        return next_obs, reward, done, info


if __name__ == '__main__':
    # from QP_solver import OSQPSolver
    from solvers.qpsolver import QPOASES_solver

    mpceg = RandomQPExample(2).qp_problem
    P, q, A, l, u = mpceg['P'], mpceg['q'], mpceg['A'], mpceg['l'], mpceg['u']

    # 1. get solution from osqp
    # osqp_solver = OSQPSolver()
    # osqp_solver.setup(P, q, A, l, u)
    # res = osqp_solver.solve()
    # print(res.status)
    # resx = res.x
    # resy = res.y
    # print(resx)
    # print(np.where(resy != 0, 2, 0))
    # print(u-A.dot(resx))

    # 2. get solution from qpOASES
    qpoases_solver = QPOASES_solver()
    qpoases_solver.setup(**mpceg)
    qpoases_solver.solve()
    # print("number of iteration:",qpoases_solver.solver.get_niter())  # 5 //8 if warm by rl
    resx = qpoases_solver.result()
    # print("active set:",spa.csc_matrix(np.array(qpoases_solver.get_as())))

    # 3. setup env
    qpeps = QPEpisode(P, q, A, l, u, step_reward=-1)

    # though only one step is needed to solve LP via OSQP
    rng = np.random.default_rng(123)
    for step in range(5000):
        # rng_action = [rng.integers(low=0,high=3) for i in range(l.shape[0])]
        rng_action = [rng.choice([0, 2]) for i in range(l.shape[0])]
        next_obs, reward, done, _ = qpeps.step(rng_action)
        if done:
            print(step)
            print(spa.csc_matrix(np.array(rng_action)))
            print(qpeps.x)
            print("residual:", np.mean(resx - qpeps.x))
            break
        if step % 100 == 0:
            print("<", end=" ")

    # print(next_obs.shape)

"""OSQP结果
optimal
[ 0.99025707 -0.46913575 -3.34821661  0.04662148 -0.83853259]
[0 1 0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0
 0 0 0 0 0 0 0 0 0 0 0 0 1]
 #qpOASES结果
 [0.9879302117640902, -0.4684773317453783, -3.3506565255245038, 0.04657517256928717, -0.8378670197181424]
[0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Problem dimension: (5, 5) (5,) (50, 5) (50,) (50,)
< < < < < < < < < < < < < < < < < < < < < < < < < < < 2672
  (0, 3)	2
  (0, 4)	2
  (0, 5)	2
  (0, 7)	2
  (0, 13)	2
  (0, 15)	2
  (0, 19)	2
  (0, 24)	2
  (0, 27)	2
  (0, 29)	2
  (0, 31)	2
  (0, 32)	2
  (0, 33)	2
  (0, 36)	2
  (0, 40)	2
  (0, 41)	2
  (0, 42)	2
  (0, 44)	2
  (0, 46)	2
  (0, 47)	2
  (0, 48)	2
[ 4.59524187e-07 -8.16749563e-06  9.12456605e-05  6.59526305e-06
  3.26346744e-06]
residual: -0.7245177778148311

"""
