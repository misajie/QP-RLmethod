import numpy as np
import osqp
from qpenv import statuses as s
from qpenv.results import Results
from qpenv.general import is_qp_solution_optimal
import time


class OSQPSolver(object):

    m = osqp.OSQP()
    STATUS_MAP = {osqp.constant('OSQP_SOLVED'): s.OPTIMAL,
                  osqp.constant('OSQP_MAX_ITER_REACHED'): s.MAX_ITER_REACHED,
                  osqp.constant('OSQP_PRIMAL_INFEASIBLE'): s.PRIMAL_INFEASIBLE,
                  osqp.constant('OSQP_DUAL_INFEASIBLE'): s.DUAL_INFEASIBLE}

    def __init__(self, settings=None):
        '''
        Initialize solver object by setting require settings
        '''
        if settings is None:
            settings = {
                'verbose': False,
                'adaptive_rho': False,
                'eps_rel': 1e-03,
                'eps_abs': 1e-03,
                'polish': False,
                'max_iter': int(1e05),
                'eps_prim_inf': 1e-15,  # disable infeasibility check
                'eps_dual_inf': 1e-15,  # disable infeasibility check
            }
        self._settings = settings

    @property
    def settings(self):
        """Solver settings"""
        return self._settings

    def setup(self,P,q,A,l,u):
        self.P,self.q,self.A,self.l,self.u = P,q,A,l,u
        settings = self._settings.copy()
        # Setup OSQP
        self.m = osqp.OSQP()
        self.m.setup(P,q,A,l,u,**settings)  # how to keep formal obj when constraints are reset

    def solve(self):
        '''
        Solve problem
        Returns:
            Results structure
        '''
        settings = self._settings.copy()
        # Solve
        start = time.time()
        results = self.m.solve()
        end = time.time()
        status = self.STATUS_MAP.get(results.info.status_val, s.SOLVER_ERROR)
        high_accuracy = settings.pop('high_accuracy', None)
        if status in s.SOLUTION_PRESENT:
            if not is_qp_solution_optimal(self.P,self.q,self.A,self.l,self.u,
                                          results.x,
                                          results.y,
                                          high_accuracy=high_accuracy):
                status = s.SOLVER_ERROR

        # Verify solver time
        if settings.get('time_limit') is not None:
            if results.info.run_time > settings.get('time_limit'):
                status = s.TIME_LIMIT

        # run_time = results.info.run_time
        run_time = end-start
        return_results = Results(status,
                                 results.info.obj_val,
                                 results.x,
                                 results.y,
                                 run_time,
                                 results.info.iter)

        return_results.status_polish = results.info.status_polish
        return_results.setup_time = results.info.setup_time
        return_results.solve_time = results.info.solve_time

        return return_results

if __name__ == '__main__':
    # from rlqp_benchmarks.benchmark_problems.control_example import MPC0QPExample
    import warnings
    warnings.filterwarnings("ignore")
    settings = {
        'verbose': False,
        'adaptive_rho': False,
        'eps_rel': 1e-03,
        'eps_abs': 1e-03,
        'polish': False,
        'max_iter': int(1e05),
        'eps_prim_inf': 1e-15,  # disable infeasibility check
        'eps_dual_inf': 1e-15,  # disable infeasibility check
    }
    mpceg = MPC0QPExample(idx=1)
    qp_eg = mpceg.qp_problem
    P,q,A,l,u = qp_eg['P'],qp_eg['q'],qp_eg['A'],qp_eg['l'],qp_eg['u']
    l[0] = 0
    A[0, :] = 0
    A.eliminate_zeros()
    print(A)
    u[0] = 0
    # print(l,u)
    osqp_s = OSQPSolver(settings=settings)
    osqp_s.setup(P,q,A,l,u)
    res = osqp_s.solve()
    print(res.y)