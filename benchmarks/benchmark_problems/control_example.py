import random

import numpy as np
import scipy.sparse as spa
import cvxpy
import pandas as pd

class MPC0QPExample(object):
    '''
    Random QP example
    '''
    def __init__(self, MPC_PATH = "../normal/", buffer_size=100):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        # rng = np.random.default_rng(seed)
        # load QP data
        self.buffer_size = buffer_size
        self.hessian = pd.read_csv(MPC_PATH+"hessian.csv",nrows=buffer_size,header=None)
        self.constraint_ub_ = pd.read_csv(MPC_PATH+"a_ub.csv",nrows=buffer_size,header=None)
        self.constraint_lb_ = pd.read_csv(MPC_PATH+"a_lb.csv",nrows=buffer_size,header=None)
        self.constraint = pd.read_csv(MPC_PATH+"a_mat.csv", nrows=buffer_size,header=None)
        self.g_vector = pd.read_csv(MPC_PATH+"g_vec.csv",nrows=buffer_size,header=None)
        # print(hessian.values.dtype,constraint_lb_.values.dtype,constraint.values.dtype,g_vector.values.dtype)
        # print(hessian.shape,g_vector.shape,constraint.shape,constraint_lb_.shape)
        self.n = self.g_vector.shape[1]        # number of variables
        self.m = self.constraint_lb_.shape[1]  # number of constraints
        # Generate problem data

        # print(self.P.shape, self.A.shape, self.q.shape, self.l.shape, self.u.shape)
        # (120, 120) (200, 120) (120,) (200,) (200,)
        self.qp_problem = self.random_mpc_problem()
        # TODO: cvx problem
        # self.cvxpy_problem = self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'MPC0'

    def random_mpc_problem(self, seed=1):
        return self._generate_qp_problem(seed=seed)

    def _generate_qp_problem(self, seed=1):
        '''
        Generate QP problem
        '''
        rng = np.random.default_rng(seed)
        index = rng.integers(0,self.buffer_size-1)
        P = spa.csr_matrix(self.hessian.iloc[index].to_numpy(dtype=np.float64).reshape((self.n, self.n)))
        A = spa.csr_matrix(self.constraint.iloc[index].to_numpy(dtype=np.float64).reshape((self.m, self.n)))

        self.P = P.tocsc()
        self.q = self.g_vector.iloc[index].to_numpy(dtype=np.float64)
        self.A = A.tocsc()
        self.u = self.constraint_ub_.iloc[index].to_numpy(dtype=np.float64)
        # self.u = constraint_lb_.iloc[self.index].to_numpy()+np.random.randint(1,100,size=(self.m,))
        self.l = self.constraint_lb_.iloc[index].to_numpy(dtype=np.float64)
        problem = {}
        problem['P'] = self.P
        problem['q'] = self.q
        problem['A'] = self.A
        problem['l'] = self.l
        problem['u'] = self.u
        problem['m'] = self.A.shape[0]
        problem['n'] = self.A.shape[1]

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        x_var = cvxpy.Variable(self.n)
        objective = .5 * cvxpy.quad_form(x_var, self.P) + self.q * x_var
        constraints = [self.A * x_var <= self.u, self.A * x_var >= self.l]
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        return problem

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        variables = self.cvxpy_problem.variables()
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = variables[0].value

        # dual solution
        y = constraints[0].dual_value - constraints[1].dual_value

        return x, y

if __name__ == '__main__':
    a = MPC0QPExample(3)
    # print(a.l-a.u)
