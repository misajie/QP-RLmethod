# drop constraint environment
import numpy as np
from solvers.ASMsolver import ASM
from benchmarks.problem_classes.random_qp import RandomQPExample
from benchmarks.benchmark_problems.control_example import MPC0QPExample


# from solvers.qpsolver import QPOASES_solver


class QPEpisode:
    def __init__(self, P, q, A, l, u, settings=None):
        if settings is None:
            settings = {
                "eps": 1e-6,
                "max_step": 10000,
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
            self.state = np.stack([self.get_obs(i) for i in range(2 * self.m)], 0).reshape(-1)

    def step(self, action):  # action is a rating for all m constraints
        action = np.argmin(action[self.drop_set])
        del self.solver.WorkingSet[self.drop_set[action]]
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
                    self.nstate = np.stack([self.get_obs(i) for i in range(2 * self.m)], 0).reshape(-1)
                    self.state = self.nstate
                    break
        step_reward = 1 * nstep
        self.score += step_reward
        self.niter += nstep
        return self.nstate, step_reward, self.done, len(self.drop_set)

    def get_obs(self, id):
        self.x = self.solver.x.copy()  # primal solution
        constraint = id % self.m
        Ax = self.A.dot(self.x)[constraint]
        lx = self.l[constraint]
        ux = self.u[constraint]

        H_trace = np.trace(self.P)
        Ag_trace = self.A.dot(self.q)[constraint]

        # for fixed dim
        # + self.A[constraint].tolist()
        res = np.array([H_trace, Ag_trace, ux] + [lx] + [Ax - lx, ux - Ax])
        res = 1 / (1 + np.exp(-np.array(res)))  # map to 0-1,avoid inf
        return res


class ASenv:
    def __init__(self, qp_type=0, mpc_path="../benchmarks/normal/", buffer_size=None):
        self.env_num = 1
        self.env_name = "rlas-v1"
        self.max_step = 1000
        self.state_dim = 6 * 400
        self.action_dim = 400
        self.if_discrete = False
        self.target_return = -79
        examples = [RandomQPExample,
                    MPC0QPExample]
        self.problem_set = examples[qp_type]
        self.problem_class = self.problem_set.name()

        if buffer_size is not None:
            self.problem_class = "mpc"
            mpc_path = mpc_path
            self.problem = self.problem_set(mpc_path, buffer_size=buffer_size)
        else:
            self.problem = self.problem_set
        self.no = 0
        self.state = self.reset()

    def _random_episode(self, seed=None):
        if seed is None:
            seed = self.no
        if self.problem_class == "mpc":
            mpceg = self.problem.random_mpc_problem(seed)
        else:
            mpceg = self.problem(self.no % 6 + 5, seed=self.no).qp_problem

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

    def random_action(self):
        return np.random.rand(self.action_dim)

    def as_action(self):
        action = np.zeros(2 * self.cur_episode.m)
        drop_set = self.cur_episode.drop_set
        action[drop_set] += self.cur_episode.solver.y[drop_set]
        return action

    def get_dim(self):
        return self.state.shape

    def step(self, action):
        nstate, step_reward, done, epilen = self.cur_episode.step(action)
        return nstate, step_reward, done, epilen

    def render(self):
        return self.qp_problem

    # def close(self):
    #     del self.problem_sets


if __name__ == '__main__':
    # import torch
    # from Agents.layers import BiRNN_nopack
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = BiRNN_nopack(244,256,2,1)
    # model.to(device)
    # model.load_state_dict(torch.load("../pretrain/mpc_rnn.pth"))

    env = ASenv(qp_type=1, buffer_size=100)
    total_score = []
    # env = ASenv()
    for i in range(25, 26):
        print("QP:", i)
        state = env.reset()
        while True:
            # action = 0  # fixed action
            # action = env.random_action(state)  # random action 88平均
            # print(state.shape[0])
            action = env.as_action()  # active set action 86平均

            # NN action
            # state = 1/(1+np.exp(state[:,:-3]))
            # state = torch.from_numpy(state).reshape(-1, state.shape[0], 244).to(device)
            # state = state.permute(1, 0, 2).to(torch.float32)
            # pred = model(state) # 全1pred待解决
            # action = torch.argmax(pred).cpu().detach().numpy()
            nstate, reward, done, info = env.step(action)
            state = nstate
            print(state.shape)
            if done:
                # TODO: check the obj value and as with qpoases
                # print(env.cur_episode.get_as())
                # qpeg = env.render()
                # qpeg['m'], qpeg['n'] = qpeg['A'].shape
                # qpoases = QPOASES_solver({})  # all P is pd.
                # qpoases.setup(**qpeg)
                # qpoases.solve()
                # x1, ws1 = qpoases.result(), qpoases.get_as()
                # idx = np.where(np.array(ws1)>0)[0][0]
                #
                # obj = 1 / 2 * np.dot(np.array(x1).T, qpeg['P'].toarray()).dot(x1) + np.dot(qpeg['q'].T, x1)

                # print(ws1) # check active set
                # print(qpeg['l'][idx], qpeg['u'][idx])
                # print(env.cur_episode.get_obj()-obj)
                break
        total_score.append(env.cur_episode.score)
        # if i%100 == 0:
        #     print(np.mean(total_score))
    print(total_score)
