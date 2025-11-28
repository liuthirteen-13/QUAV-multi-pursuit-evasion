import numpy as np
from config import N_PURSUERS, STATE_DIM


def build_graph_laplacian():
    """
    4 个追击者，环形拓扑。
    b_2 = 1, 其余 0
    """
    n = N_PURSUERS
    A = np.zeros((n, n))

    for i in range(n):
        A[i, (i - 1) % n] = 1
        A[i, (i + 1) % n] = 1

    deg = np.sum(A, axis=1)
    L = np.diag(deg) - A

    B = np.diag([0, 1, 0, 0])
    L_tilde = L + B
    return A, L, B, L_tilde
