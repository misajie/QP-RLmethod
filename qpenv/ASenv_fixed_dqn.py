# drop constraint environment
import numpy as np
from solvers.ASMsolver import ASM
from benchmarks.problem_classes.random_qp import RandomQPExample
from benchmarks.benchmark_problems.control_example import MPC0QPExample
from solvers.qpsolver import QPOASES_solver


class QPEpisode:
    def __init__(self, P, q, A, l, u, settings=None):
        if settings is None:
            settings = {
                "eps": 1e-6,
            }
        self.settings = settings
        self.labels = {"inactive": 0, "left_active": 1, "right_active": 2, "active": 3}
        self.solver = None
        self.working_set = None
        self.drop_set = None  # key operator for this env
        self.done = False
        self.score = 0
        self.niter = 0

        m, n = A.shape
        self.setup(P, q, A, l, u, m, n)

    def setup(self, P, q, A, l, u, m, n):
        # Ax>=B format
        self.n = n
        self.m = m
        self.P = P.toarray()
        self.q = q
        self.A = A.toarray()
        self.l = l
        self.u = u

        self.AA = np.array([-self.A, self.A]).reshape(2 * self.m, self.n)
        self.B = np.array([-self.u, self.l]).reshape(2 * self.m, 1)
        self.solver = ASM(self.P, self.q, self.AA, self.B, self.settings['eps'])
        self.solver.init()
        nstep = 0
        while not self.solver.solve_step():
            nstep += 1
        self.niter += nstep
        # print("init reward:",-1*nstep)
        if self.solver.status != 0:
            self.done = True  # if init done, then reset problem
        else:
            self.done = False
            self.drop_set = self.solver.drop_set
            self.state = np.stack([self.get_obs(i) for i in range(2*self.m)], 0)

    def get_as(self):
        assert self.done, "qp not solved"
        tag = [self.labels["inactive"]] * self.m
        self.x = self.solver.x
        cur_cons = self.AA.dot(self.x.reshape(-1, 1))
        astag = np.where(np.abs(cur_cons - self.B) < self.settings['eps'])[0]
        for i in astag.tolist():  # sorted
            if i < self.m:
                tag[i] = self.labels['right_active']
            else:
                if tag[i - self.m] == self.labels['right_active']:
                    tag[i - self.m] = self.labels['active']
                else:
                    tag[i - self.m] = self.labels['left_active']
        return tag

    def get_obj(self):
        assert self.done, "qp not solved"
        x = self.x
        obj = 1 / 2 * np.dot(x.T, self.P).dot(x) + np.dot(self.q.T, x)
        return obj

    def step(self, action):  # action is a rating for all m constraints
        # print([self.solver.WorkingSet[i] for i in self.drop_set])
        # print(self.solver.WorkingSet)
        self.solver.WorkingSet.remove(action)
        nstep = 0
        while True:
            nstep += 1
            if self.solver.solve_step():
                if self.solver.status > 0:
                    self.done = True
                    self.nstate = self.state  # nstate 补前
                    break
                else:
                    self.done = False
                    self.drop_set = self.solver.drop_set
                    self.nstate = np.stack([self.get_obs(i) for i in range(2*self.m)], 0)
                    self.state = self.nstate
                    break
        step_reward = -1 * nstep
        self.score += step_reward
        self.niter += nstep
        return self.nstate, step_reward, self.done, len(self.drop_set)

    def get_obs(self, id):
        self.x = self.solver.x.copy()  # primal solution
        drop_sets = [self.solver.WorkingSet[j] for j in self.drop_set]

        ws_tag = 1 if id in drop_sets else 0
        if ws_tag > 0:
            idx = drop_sets.index(id)
            self.y = self.solver.y[idx]
        else:
            self.y = 0
        constraint = id%self.m
        Ax = self.A.dot(self.x)[constraint]
        lx = self.l[constraint]
        ux = self.u[constraint]
        # Px = self.P.dot(x)
        # Aty = self.A.T.dot(y)
        # dual_res = Px + q + Aty

        H_trace = np.trace(self.P)
        Ag_trace = self.A.dot(self.q)[constraint]
        g_f = self.q

        # for fixed dim
        # + self.A[constraint].tolist()
        res = np.array([H_trace, Ag_trace, ux] + [lx] + [lx - Ax, ux - Ax, self.y])
        # res = 1 / (1 + np.exp(-np.array(res)))  # map to 0-1,avoid inf
        res = np.r_[res, np.array([ws_tag])]
        return res


class ASenv:
    def __init__(self, mpc_path="../benchmarks/normal/", buffer_size=None):
        # examples = [RandomQPExample,
        #             MPC0QPExample]
        # EXAMPLES_MAP = {example.name(): example for example in examples}
        if buffer_size is not None:
            self.problem_class = "mpc"
            self.problem_sets = MPC0QPExample(mpc_path, buffer_size=buffer_size)
        else:
            self.problem_class = "rand"
        self.no = 0
        self.state = self.reset()

    def _random_episode(self, probelm_class=None, seed=None):
        if seed is None:
            seed = self.no
        if probelm_class is None:
            probelm_class = "mpc"
        if self.problem_class == "mpc":
            mpceg = self.problem_sets.random_mpc_problem(seed)
        else:
            mpceg = RandomQPExample(self.no % 6 + 5, seed=self.no).qp_problem

        self.qp_problem = dict(
            P=mpceg['P'],
            q=mpceg['q'],
            A=mpceg['A'],
            l=mpceg['l'],
            u=mpceg['u']
        )
        self.no += 1
        return QPEpisode(**self.qp_problem)

    # TODO: 如何保存初始解决的QP结果?暂时忽略
    def reset(self):
        while True:
            self.cur_episode = self._random_episode()
            if not self.cur_episode.done:
                break
        state = self.cur_episode.state
        return state

    def random_action(self,state):
        return self.cur_episode.solver.WorkingSet[np.random.choice(self.cur_episode.drop_set)]

    def as_action(self,state):
        return self.cur_episode.solver.WorkingSet[np.argmin(self.cur_episode.solver.y)]

    def get_dim(self):
        return self.state.shape

    def step(self, action):
        nstate, step_reward, done, epilen = self.cur_episode.step(action)
        return nstate, step_reward, done, epilen

    def render(self):
        return self.qp_problem

    # def close(self):
    #     del self.problem_sets

def run_episode(env):
    state = env.reset()
    while True:
        # action = agent.select_action(state)
        action = env.random_action(state)  # test score:-6.61, average niter:13.37
        # action = env.as_action()    # AS policy:test score:-5.94, average niter:12.7
        nstate, reward, done, _ = env.step(action)
        print(action)
        state = nstate
        if done:
            print("done")
            break
    return env.cur_episode.score, env.cur_episode.niter

if __name__ == '__main__':
    env = ASenv(buffer_size=100)
    run_episode(env)