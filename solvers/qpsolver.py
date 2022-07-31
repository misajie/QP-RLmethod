import sys
sys.path.append("../loas/")
sys.path.append("../benchmarks/")
import warnings
warnings.filterwarnings("ignore")

import mpc_osqp as qp_solver


class QPOASES_solver():
    def __init__(self, settings={}):
        self.solver_name = qp_solver.QPOASES
        self.settings = settings
        # TODO: use _call to reset guess constraints
        # self.warm_start = settings["warm_start"]
        # self.init_guess_constraint = settings['init_guess_constraint']   # any list, such as [0]

    def setup(self,P,q,A,l,u,m,n):
        self.n = n
        self.m = m
        self.P = P.toarray().reshape(-1).tolist()
        self.q = q.tolist()
        self.A = A.toarray().reshape(-1).tolist()
        self.l = l.tolist()
        self.u = u.tolist()
        self.solver = qp_solver.QPsolvers(self.n, self.m,
                                          self.P,self.q,self.A,self.l,self.u,
                                          self.solver_name,
                                          0,[0 for i in range(self.m)])

    def warm_setup(self,P,q,A,l,u,m,n,warm_ls=None):
        self.n = n
        self.m = m
        self.P = P.toarray().reshape(-1).tolist()
        self.q = q.tolist()
        self.A = A.toarray().reshape(-1).tolist()
        self.l = l.tolist()
        self.u = u.tolist()

        if warm_ls == None:
            self.warm_ls = [0 for i in range(self.m)]
        else:
            self.warm_ls = warm_ls

        self.solver = qp_solver.QPsolvers(self.n, self.m,
                                          self.P, self.q, self.A, self.l, self.u,
                                          self.solver_name,
                                          1,self.warm_ls)

    def solve(self):
        assert self.solver is not None, "solver not setup!"
        self._result_info = self.solver.solve_qp()
        self._activeset = self.solver.get_oas()

    # TODO: update warm constraints
    def update_warm(self):
        pass

    def get_as(self):
        assert self._activeset is not None, "QP not solved"
        return self._activeset

    def result(self):
        assert self._result_info is not None, "QP not solved"
        return self._result_info

class OSQP_solver():
    def __init__(self, settings={}):
        self.solver_name = qp_solver.OSQP
        self.settings = settings
        # TODO: use _call to reset guess constraints
        # self.init_guess_constraint = settings['init_guess_constraint']   # any list, such as [0]

    def setup(self,P,q,A,l,u,m,n):
        self.n = n
        self.m = m
        self.P = P.toarray().reshape(-1).tolist()
        self.q = q.tolist()
        self.A = A.toarray().reshape(-1).tolist()
        self.l = l.tolist()
        self.u = u.tolist()
        self.solver = qp_solver.QPsolvers(self.n, self.m,
                                          self.P,self.q,self.A,self.l,self.u,
                                          self.solver_name,
                                          0,[0 for i in range(self.m)])

    def solve(self):
        assert self.solver is not None, "solver not setup!"
        self.result_info = self.solver.solve_qp()

    # TODO: update warm constraints
    def update_warm(self):
        pass

    def result(self):
        assert self.result_info is not None, "QP not solved"
        return self.result_info
