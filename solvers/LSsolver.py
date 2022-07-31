import numpy as np
import qdldl
import scipy.sparse as spa
from scipy import linalg as la


class LPSolver(object):
    def __init__(self, settings={}):
        # TODO: construct KKT matrix for LP; need to control the constraint dim, add 0 for inactive ones
        # TODO: solve KKT matrix and return dual/primal result
        pass
        self._settings = settings

    @property
    def settings(self):
        """Solver settings"""
        return self._settings

    def cscsetup(self, P, q, A, l, u):
        self.P, self.q, self.A, self.b = P.toarray(), q, A.toarray(), u
        PAt = np.hstack((self.P, self.A.T))
        A0 = np.hstack((self.A, np.zeros((self.A.shape[0], self.A.shape[0]))))
        self.KKT_matrix = np.row_stack((PAt, A0))
        print(self.KKT_matrix.shape)
        # + smallest eig +
        if la.det(self.KKT_matrix)==0:
            for i in range(self.KKT_matrix.shape[0]):
                if self.KKT_matrix[i][i] == 0:
                    self.KKT_matrix[i][i] = 1e-6
        self.KKT_matrix = spa.csc_matrix(self.KKT_matrix)
        self.KKT_b = np.r_[-self.q, self.b]
        # print(self.KKT_matrix)
        # print(self.KKT_matrix.shape,self.KKT_b.shape)

    def cscsolve(self):
        F = qdldl.Solver(self.KKT_matrix,upper=False)
        res = F.solve(self.KKT_b)
        x, y = res[:self.P.shape[1]], res[self.P.shape[1]:]
        return x,y

    def lasetup(self,P, q, A, l, u):
        self.P, self.q, self.A, self.b = P.toarray(), q, A.toarray(), u
        PAt = np.hstack((self.P, self.A.T))
        A0 = np.hstack((self.A, np.zeros((self.A.shape[0], self.A.shape[0]))))
        self.KKT_matrix = np.row_stack((PAt, A0))
        if la.det(self.KKT_matrix)==0:
            for i in range(self.KKT_matrix.shape[0]):
                if self.KKT_matrix[i][i] == 0:
                    self.KKT_matrix[i][i] = 1e-6
        self.KKT_b = np.r_[-self.q, self.b]

    def lasolve(self):
        res = la.solve(self.KKT_matrix,self.KKT_b)
        x,y = res[:self.P.shape[1]],res[self.P.shape[1]:]
        return x,y

def check_stack():
    P = np.random.randint(0,2,size=(4,4))
    A = np.random.randint(0,2,size=(5,4))
    PAt = np.hstack((P, A.T))
    A0 = np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))
    print(np.row_stack((PAt, A0)))
    KKT_matrix = spa.csc_matrix(np.row_stack((PAt, A0)))
    print(KKT_matrix)
    b = np.zeros((9,))
    F = qdldl.Solver(KKT_matrix, upper=False)
    x,y = F.solve(KKT_b)
    # RuntimeError: Error in computing elimination tree. Matrix not properly upper-triangular, sum_Lnz = -1
    # error, maybe something wrong about quasidefine/singular
    print(x)

def check_csc_solve():
    mpceg = control_example.MPC0QPExample(idx=1).qp_problem
    P, q, A, l, u = mpceg['P'], mpceg['q'], mpceg['A'], mpceg['l'], mpceg['u']
    s = LPSolver()
    s.cscsetup(P, q, A, l, u)
    x,y = s.cscsolve()

def check_la_solve():
    mpceg = control_example.MPC0QPExample(idx=1).qp_problem
    P, q, A, l, u = mpceg['P'], mpceg['q'], mpceg['A'], mpceg['l'], mpceg['u']
    s = LPSolver()
    s.lasetup(P, q, A, l, u)
    x,y = s.lasolve()

if __name__ == '__main__':
    import sys
    sys.path.append("../rlqp_benchmarks")
    from rlqp_benchmarks.benchmark_problems import control_example
    check_csc_solve()
    # check_la_solve()