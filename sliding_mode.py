# sliding_mode.py
import numpy as np
from config import STATE_DIM, BETA, N_PURSUERS


def compute_sliding_and_Es(pursuer_states, evader_state, L_tilde):
    """
    pursuer_states: (N, 6)
    evader_state : (6,)
    返回：
        s  : (N, 3)
        Es : (N, 3)
    """

    n = N_PURSUERS

    # ---------- 必须是 3 维 ----------
    s = np.zeros((n, 3))
    Es = np.zeros((n, 3))

    # 逃避者角度与角速度
    phi_e = evader_state[0:3]
    dphi_e = evader_state[3:6]

    for k in range(n):
        phi_p = pursuer_states[k, 0:3]
        dphi_p = pursuer_states[k, 3:6]

        # ζ_p, ζ_v （都是 3 维）
        zeta_p = phi_p - phi_e
        zeta_v = dphi_p - dphi_e

        # 滑模面 s_k：3 维
        s[k, :] = BETA * zeta_p + zeta_v

    # E_s = (L_tilde ⊗ I_3) * s
    s_flat = s.reshape(-1)  # 长度 = 3N
    L_kron = np.kron(L_tilde, np.eye(3))
    Es_flat = L_kron @ s_flat
    Es = Es_flat.reshape((n, 3))

    return s, Es
