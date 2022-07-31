from scipy.optimize import linprog
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from benchmarks.benchmark_problems.control_example import MPC0QPExample

warnings.filterwarnings("ignore")


class ASM:
    def __init__(self, H, C, A, B, eps=1e-6):
        self.H = H
        self.C = C.reshape(-1, 1)
        self.A = A
        self.B = B.reshape(-1, 1)
        # A_ub = self.A
        # b_ub = self.B.reshape(-1)
        # lb = np.array([-np.inf]*self.A.shape[1])
        # ub = np.array([np.inf]*self.A.shape[1])
        # tol = 1e-6
        #
        # singleton_row = np.array(np.sum(A_ub != 0, axis=1) == 1).flatten()
        # cols = np.where(A_ub[singleton_row, :])[1]
        # rows = np.where(singleton_row)[0]
        # if len(rows) > 0:
        #     complete = False
        #     for row, col in zip(rows, cols):
        #         val = b_ub[row] / A_ub[row, col]
        #         print(A_ub[row, col] > 0,val, lb[col],ub[col])
        #         if A_ub[row, col] > 0:  # upper bound
        #             if val < lb[col] - tol:  # infeasible
        #                 complete = True
        #             elif val < ub[col]:  # new upper bound
        #                 ub[col] = val
        #         else:  # lower bound
        #             if val > ub[col] + tol:  # infeasible
        #                 complete = True
        #             elif val > lb[col]:  # new lower bound
        #                 lb[col] = val
        #         if complete:
        #             print("The problem is (trivially) infeasible because a "
        #                        "singleton row in the upper bound constraints is "
        #                        "inconsistent with the bounds.")
        # print("end check")
        mask = np.isinf(self.B)
        self.B[mask] = 0
        mask = np.repeat(mask,self.A.shape[1],1)
        self.A[mask] = 0

        self.eps = eps
        self.x = None
        self.y = None
        self.WorkingSet = []

        self.nWSR = 10000
        self.total_step = 0
        self.status = 0  # 0: not solved; 1: solved; -1: not solvable;2: reach maximum steps;
        self.path = []
        self.drop_set = []

        # self.init()


    def derivative(self, x):
        de = self.H.dot(x) + self.C.T
        return de[0]

    def KKT(self, H, C, A, B):
        n = self.H.shape[0]
        wsc = len(self.WorkingSet)
        kkt_A = np.zeros((n + wsc, n + wsc))
        kkt_B = np.zeros((n + wsc))

        kkt_A[:n, :n] = H
        kkt_A[:n, n:] = -A.T
        kkt_A[n:, :n] = -A

        kkt_B[:n] = -C
        kkt_B[n:] = B[:, 0]
        return np.linalg.inv(kkt_A).dot(kkt_B)
        # try:
        #     res = np.linalg.inv(kkt_A).dot(kkt_B)
        #     return res
        # except np.linalg.LinAlgError as e:
        #     print(kkt_A)
        #     return False

    @classmethod
    def null_kkt(A, atol=1e-13, rtol=0):
        A = np.atleast_2d(A)
        u, s, vh = np.linalg.svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns

    def Alpha(self, x, p):
        # print(x)
        # print("P:",p)
        min_alpha = 1
        new_constraint = 0
        decision_set = []
        for i in range(self.A.shape[0]):
            if i in self.WorkingSet:
                continue
            else:
                bi = self.B[i]
                ai = self.A[i]
                atp = ai.dot(p)
                res = bi - ai.dot(x)
                # print(atp, res)
                if atp >= 0 or np.abs(atp) < self.eps:
                    continue
                else:
                    alpha = res[0] / atp

                    decision_set.append(i)
                    if alpha < min_alpha:
                        # print(i,alpha, bi,ai,atp)
                        min_alpha = alpha
                        new_constraint = i
        return min_alpha, new_constraint, decision_set

    def solve(self):
        self.init()
        count = self.H.shape[1]
        self.add_set = []
        # 2. loop
        for i in range(self.nWSR):
            # self.path.append([self.x[0], self.x[1]])
            # KKT equation
            c = self.derivative(self.x)
            a = self.A[self.WorkingSet]
            b = np.zeros_like(self.B[self.WorkingSet])
            # print(self.WorkingSet)
            dlambda = self.KKT(self.H, c, a, b)
            _lambda = dlambda[count:]  # dual variable
            d = dlambda[0:count]
            # print(self.WorkingSet, _lambda, np.linalg.norm(d, ord=1))
            # get descending direction
            if np.linalg.norm(d, ord=1) < self.eps:
                # optimal condition
                # error: 处理空working set: ValueError: zero-size array to reduction operation minimum which has no identity
                # 下降到头都找不到一个block constraint，甚至把当前working set删空
                # 待解决，20220617

                if _lambda.shape[0] == 0 or _lambda.min() > 0:
                    self.nWSR = i
                    self.status = 1
                    break
                else:
                    # TODO: drop step
                    if _lambda.shape[0] != 0:
                        index = np.argmin(_lambda)
                        # index = np.random.choice(np.where(_lambda < 0)[0])
                        del self.WorkingSet[index]
                        self.WorkingSet.sort()

            else:
                # TODO: add step
                alpha, new_constraint, decision_set = self.Alpha(self.x, d)
                self.add_set.append(np.sum(np.array(decision_set) == alpha))
                # print("add", alpha, np.sum(self.A.dot(self.x.reshape(-1, 1)) >= self.B)) # 原因：初始解不可行！
                self.x += alpha * d
                if alpha < 1:
                    self.WorkingSet.append(new_constraint)
                    self.WorkingSet.sort()

    def init(self):
        n = self.H.shape[0]
        obj = [0] * n
        res = linprog(obj, A_ub=-self.A, b_ub=-self.B.reshape(-1))  # 此处为<=
        # assert res.get("success"), "Primal infeasible"
        if not res.get("success"):
            self.status = -1
        init_x = np.array(res.get("x"))
        init_step = 0
        alpha = 1
        cons = 0
        ps = [np.power(10*np.ones(n), k) for k in range(10)]
        for p in ps:
            alpha, cons, _ = self.Alpha(init_x, p)
            if alpha < 1:
                init_step = p
                break
        # print(alpha)
        init_x += init_step * alpha
        self.WorkingSet = [cons]
        if "determined in presolve" in res.message or np.sum(self.A[cons]) < self.eps:
            self.WorkingSet = []
        # init feasible point at init working set
        self.x = init_x
        self.drop_set = []


    def solve_step(self):
        self.total_step += 1
        if self.total_step > self.nWSR:
            self.status = 2
            return True
        c = self.derivative(self.x)
        a = self.A[self.WorkingSet]
        b = np.zeros_like(self.B[self.WorkingSet])
        dlambda = self.KKT(self.H, c, a, b)
        count = self.H.shape[1]
        _lambda = dlambda[count:]  # dual variable
        self.y = _lambda
        d = dlambda[0:count]
        # get descending direction
        if np.linalg.norm(d, ord=1) < self.eps:
            # optimal condition
            if _lambda.shape[0] == 0 or _lambda.min() > 0:
                self.status = 1
                return True
            else:
                # TODO: drop step
                if _lambda.shape[0] != 0:
                    self.drop_set = np.where(_lambda < 0)[0].tolist()
                    # # index = np.argmin(_lambda)
                    # del self.WorkingSet[index]
                    # self.WorkingSet.sort()
                    # print("decision cons:",np.where(_lambda < 0))
                    if len(self.drop_set)==1:
                        del self.WorkingSet[self.drop_set[0]]
                        return False
                    else:
                        return True  # return state here
        else:
            # TODO: add step
            alpha, new_constraint, _ = self.Alpha(self.x, d)
            self.x += alpha * d
            if alpha < 1:
                self.WorkingSet.append(new_constraint)
                self.WorkingSet.sort()
            return False

    def add_step(self, action):
        # state (gap, dual, cons+obj) of constraints in decision set: [batch,seqlen,emb_dim]
        # only one should be decided to add into working set
        # argmax problem in RNN: softmax output and choose the maximum?

        # return nstate, action, done
        pass

    def get_as(self):
        assert self.status == 1, "qp not solved"
        cur_cons = self.A.dot(self.x)
        res = np.where(np.abs(cur_cons - self.B) < self.eps)[0]
        return res


class ASMSolver:
    def __init__(self, settings={}):
        self.settings = settings
        self.labels = {"inactive": 0, "left_active": 1, "right_active": 2}
        self.WorkingSet = None

    def setup(self, P, q, A, l, u, m, n):
        self.n = n
        self.m = m
        self.P = P.toarray()
        self.q = q
        self.A = A.toarray()
        self.l = l
        self.u = u

        self.AA = np.array([-self.A, self.A]).reshape(2 * self.m, self.n)
        # TODO: wrong to deal with inf
        # self.B = np.array([self.u,-self.u+1]).reshape(2*self.m, 1)
        self.B = np.array([-self.u, self.l]).reshape(2 * self.m, 1)
        self.asmsolver = ASM(self.P, self.q, self.AA, self.B)

    def solve(self):
        self.asmsolver.solve()
        # print(self.asmsolver.x)
        # print("number of iteration:", self.asmsolver.nWSR)
        # print("active set:", self.asmsolver.WorkingSet)
        # print("path:",self.asmsolver.path)
        return self.asmsolver.x, self.asmsolver.WorkingSet


if __name__ == '__main__':
    mpc_examples = MPC0QPExample(MPC_PATH="../../../coursepros/RLAS/benchmarks/normal/", buffer_size=100)
    example = mpc_examples.random_mpc_problem()
    resarray = []
    drop_set = []
    for i in range(1):
        qpeg = mpc_examples.random_mpc_problem()
        asm_test = ASMSolver()
        asm_test.setup(**qpeg)
        x, ws = asm_test.solve()

