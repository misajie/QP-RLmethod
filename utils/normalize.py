import numpy as np


def normalize(x):
    n,m = x.shape[0],x.shape[1]
    center_matrix = np.identity(n)-np.ones((n,n))/n
    cov = x.T @ x
    scale_matrix = np.diag(np.power([cov[i][i]/n for i in range(m)],-1/2))
    res = center_matrix @ x @ scale_matrix
    return res