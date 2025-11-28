# simulation.py  —— 追击者 I–DAC + 逃避者 min–max 方案 C（含 Fig.1–7 绘图）

import numpy as np
import matplotlib.pyplot as plt

from config import (
    N_PURSUERS,
    STATE_DIM,
    DT,
    MU_P,
    MU_E,
)

from dynamics import QUAVParams, QUAVAttitudeDynamics, rk4_step
from graph import build_graph_laplacian
from sliding_mode import compute_sliding_and_Es

# 追击者 NN
from nn_identifier import PursuerIdentifier
from nn_critic import Critic as PursuerCritic
from nn_actor import Actor as PursuerActor

# 逃避者 NN
from ev_identifier import EvaderIdentifier
from ev_critic import EvaderCritic
from ev_actor import EvaderActor


# ============================================================
#                  Pursuer I–DAC Agent
# ============================================================

class PursuerAgent:
    def __init__(self, idx, lambda_m, mu_m,
                 P1=1.0, q1=15, q2=15, seed=None):

        self.idx = idx
        self.lambda_m = lambda_m
        self.mu_m = mu_m
        self.P1 = P1

        Es_dim = 3          # 只用前三维姿态误差
        state_dim_id = 6    # \bar x_m 维度

        if seed is not None:
            np.random.seed(seed)

        # Identifier
        self.identifier = PursuerIdentifier(
            state_dim=state_dim_id,
            out_dim=3,
            num_centers=q1,
            phi_fm=5.0,
            sigma_fm=0.01,
            seed=seed + 1 if seed is not None else None
        )

        # Critic
        self.critic = PursuerCritic(
            Es_dim=Es_dim,
            xbar_dim=state_dim_id,
            out_dim=3,
            num_centers=q2,
            P1=P1,
            lambda_m=lambda_m,
            mu_m=mu_m,
            identifier=self.identifier,
            kappa_cm=1.0,
            sigma_m=0.1,
            seed=seed + 2 if seed is not None else None
        )

        # Actor
        self.actor = PursuerActor(
            Es_dim=Es_dim,
            xbar_dim=state_dim_id,
            mu_p=mu_m,
            critic=self.critic,
            num_centers=q2,
            kappa_am=1.0,
            sigma_am=0.1,
            seed=seed + 3 if seed is not None else None
        )

    def build_xbar(self, state6):
        # 这里简单取 attitude + rates
        return np.concatenate([state6[0:3], state6[3:6]])

    def compute_control(self, Es_m_full, state_m):
        """
        Es_m_full: shape (6,) 或 (3,)
        只取前三维姿态误差送给 NN
        """
        Es_m = np.asarray(Es_m_full).reshape(-1)[0:3]
        x_bar = self.build_xbar(state_m)

        u = self.actor.policy(Es_m, x_bar)
        cache = dict(
            x_bar=x_bar,
            Es_old=Es_m,
        )
        return u, cache

    def update_networks(self, Es_new_full, cache, dt):
        Es_new = np.asarray(Es_new_full).reshape(-1)[0:3]
        x_bar = cache["x_bar"]

        self.identifier.update(x_bar, Es_new, dt)
        self.critic.update(cache["Es_old"], x_bar, dt)
        self.actor.update(dt)


# ============================================================
#                  Evader I–DAC Agent  (方案 C，弱化版)
# ============================================================

class EvaderAgent:
    def __init__(self, mu_s=MU_E):

        self.mu_s = mu_s

        # Identifier
        self.identifier = EvaderIdentifier(
            state_dim=6,
            out_dim=3,
            num_centers=10,
            phi_fs=0.3,
            sigma_fs=0.05,
            w_max=3.0,
            seed=200
        )

        # Critic
        self.critic = EvaderCritic(
            Es_dim=3,
            xbar_dim=6,
            out_dim=3,
            num_centers=10,
            P2=1.0,
            lambda_e=1.0,
            mu_e=mu_s,
            identifier=self.identifier,
            kappa_ce=0.001,
            sigma_e=0.5,
            w_max=3.0,
            seed=201
        )

        # Actor（方案 C：反向反馈 + 可选正弦扰动，这里扰动关掉以保持稳定）
        self.actor = EvaderActor(
            Es_dim=3,
            xbar_dim=6,
            mu_s=mu_s,
            critic=self.critic,
            num_centers=10,
            kappa_ae=0.01,
            sigma_ae=0.2,
            u_max=0.2,
            sin_amp=0.00,                  # 这里给 0，表示不加显式扰动
            sin_freqs=(0.6, 0.9, 1.2),
            seed=202
        )

    def build_xbar(self, state6):
        return np.concatenate([state6[0:3], state6[3:6]])

    def compute_control(self, Es_all_full, state_s, t):
        """
        Es_all_full: shape (N, 6) 或 (N, 3)
        取所有追击者的前三维姿态误差求和
        """
        Es_all_full = np.asarray(Es_all_full)
        if Es_all_full.ndim == 1:
            Es_all_full = Es_all_full.reshape(1, -1)

        Es_all = Es_all_full[:, 0:3] if Es_all_full.shape[1] >= 3 else Es_all_full
        Es_s = np.sum(Es_all, axis=0)  # (3,)

        x_bar_s = self.build_xbar(state_s)
        u_s = self.actor.policy(Es_s, x_bar_s, t)

        cache = {
            "Es_s_old": Es_s,
            "x_bar_s": x_bar_s,
        }
        return u_s, cache

    def update_networks(self, Es_all_next_full, cache, dt):
        # 逃避者学习更慢一些，保持“弱化”
        slow_dt = dt * 0.01

        Es_all_next_full = np.asarray(Es_all_next_full)
        if Es_all_next_full.ndim == 1:
            Es_all_next_full = Es_all_next_full.reshape(1, -1)

        Es_all_next = (Es_all_next_full[:, 0:3]
                       if Es_all_next_full.shape[1] >= 3
                       else Es_all_next_full)
        Es_s_new = np.sum(Es_all_next, axis=0)

        Es_s_old = cache["Es_s_old"]
        x_bar = cache["x_bar_s"]

        self.identifier.update(x_bar, Es_s_new, slow_dt)
        self.critic.update(Es_s_old, x_bar, slow_dt)
        self.actor.update(slow_dt)


# ============================================================
#                 主仿真类 QUAVGameSimulation
# ============================================================

class QUAVGameSimulation:
    def __init__(self):

        self.A, self.L, self.B, self.L_tilde = build_graph_laplacian()
        self.lambda_vec = np.diag(self.L_tilde)

        self.pursuer_params = [QUAVParams() for _ in range(N_PURSUERS)]
        self.evader_params = QUAVParams()

        self.pursuer_dyn = [
            QUAVAttitudeDynamics(self.pursuer_params[i])
            for i in range(N_PURSUERS)
        ]
        self.evader_dyn = QUAVAttitudeDynamics(self.evader_params)

        # 状态：角度 + 角速度
        self.pursuer_states = np.zeros((N_PURSUERS, STATE_DIM))
        self.evader_state = np.zeros(STATE_DIM)

        # 初始姿态（可以根据需要再调）
        self.pursuer_states[0, 0:3] = np.deg2rad([5, 2, -2])
        if N_PURSUERS > 1:
            self.pursuer_states[1, 0:3] = np.deg2rad([-4, 0, 4])
        if N_PURSUERS > 2:
            self.pursuer_states[2, 0:3] = np.deg2rad([3, -2, 1])
        if N_PURSUERS > 3:
            self.pursuer_states[3, 0:3] = np.deg2rad([-1, 3, -3])

        self.evader_state[0:3] = np.deg2rad([10, -4, 8])

        # agent 构造
        self.pursuer_agents = []
        for k in range(N_PURSUERS):
            lam = self.lambda_vec[k]
            ag = PursuerAgent(
                idx=k,
                lambda_m=lam,
                mu_m=MU_P,
                P1=1.0,
                q1=15,
                q2=15,
                seed=100 + k
            )
            self.pursuer_agents.append(ag)

        self.evader_agent = EvaderAgent(mu_s=MU_E)

        # 历史记录
        self.time = []
        self.err = []
        self.hji = []

        self.hist_phi_p = []   # 追击者姿态 (T, N, 3)
        self.hist_phi_s = []   # 逃避者姿态 (T, 3)

        # 追击者 NN 范数（每一项都是长度 N_PURSUERS 的向量）
        self.W_id_p = []       # identifier
        self.W_cr_p = []       # critic
        self.W_ac_p = []       # actor

        # 逃避者 NN 范数（单个网络的整体范数）
        self.W_id_s = []
        self.W_cr_s = []
        self.W_ac_s = []

        self.pursuer_att_history = []  # 所有追击者姿态角
        self.evader_att_history = []  # 逃避者姿态角

    # ---------------------------------------------------------
    def step(self, t, dt):

        # --- 计算滑模变量和 Es ---
        s, Es = compute_sliding_and_Es(
            self.pursuer_states,
            self.evader_state,
            self.L_tilde
        )
        # 只用前三维姿态误差
        Es_angles = Es[:, 0:3] if Es.shape[1] >= 3 else Es

        # =============== Pursuers 控制 ===============
        u_p_all = np.zeros((N_PURSUERS, 3))
        cache_p = []

        for k in range(N_PURSUERS):
            ag = self.pursuer_agents[k]
            u_k, c_k = ag.compute_control(Es_angles[k], self.pursuer_states[k])
            u_p_all[k] = u_k
            cache_p.append(c_k)

        # =============== Evader 控制 ===============
        u_s, cache_s = self.evader_agent.compute_control(
            Es_angles, self.evader_state, t
        )

        # =============== 状态推进 (RK4) ===============
        for k in range(N_PURSUERS):
            self.pursuer_states[k] = rk4_step(
                self.pursuer_dyn[k],
                self.pursuer_states[k],
                u_p_all[k],
                dt
            )

        self.evader_state = rk4_step(
            self.evader_dyn,
            self.evader_state,
            u_s,
            dt
        )

        # 逃避者状态钳制
        angle_lim = np.deg2rad(45)
        rate_lim = 2.0
        self.evader_state[0:3] = np.clip(
            self.evader_state[0:3], -angle_lim, angle_lim
        )
        self.evader_state[3:6] = np.clip(
            self.evader_state[3:6], -rate_lim, rate_lim
        )

        # =============== 重新计算 Es 用于 NN 更新 ===============
        _, Es_next = compute_sliding_and_Es(
            self.pursuer_states,
            self.evader_state,
            self.L_tilde
        )
        Es_next_angles = Es_next[:, 0:3] if Es_next.shape[1] >= 3 else Es_next

        # =============== 追击者 NN 更新 ===============
        for k in range(N_PURSUERS):
            self.pursuer_agents[k].update_networks(
                Es_next_angles[k],
                cache_p[k],
                dt
            )

        # =============== 逃避者 NN 更新 ===============
        self.evader_agent.update_networks(
            Es_next_angles,
            cache_s,
            dt
        )

        # =============== HJI 残差（基于追击者 critic） ===============
        H_list = []
        for k in range(N_PURSUERS):
            Es_k = Es_angles[k]
            Es_k_next = Es_next_angles[k]
            ag = self.pursuer_agents[k]

            gradJ = ag.critic.grad_J(Es_k, cache_p[k]["x_bar"])  # (3,)
            E_dot = (Es_k_next - Es_k) / dt                       # (3,)

            H_k = float(np.dot(gradJ, E_dot))
            H_list.append(H_k)

        hji_res = np.mean(np.abs(H_list))

        # =============== 追逃总姿态误差记录 ===============
        err_sum = 0.0
        phi_s = self.evader_state[0:3]
        for k in range(N_PURSUERS):
            err_sum += np.linalg.norm(self.pursuer_states[k, 0:3] - phi_s)

        # =============== 写入历史 ===============
        self.time.append(t)
        self.err.append(err_sum)
        self.hji.append(hji_res)

        self.hist_phi_p.append(self.pursuer_states[:, 0:3].copy())
        self.hist_phi_s.append(phi_s.copy())

        self.pursuer_att_history.append(self.pursuer_states[:, 0:3].copy())
        self.evader_att_history.append(self.evader_state[0:3].copy())

        # 追击者 NN 范数（一个追击者对应一条线）
        wid, wcr, wac = [], [], []
        for ag in self.pursuer_agents:
            wid.append(np.linalg.norm(ag.identifier.Phi_fm))
            wcr.append(np.linalg.norm(ag.critic.Phi_cm))
            wac.append(np.linalg.norm(ag.actor.Phi_am))

        self.W_id_p.append(wid)
        self.W_cr_p.append(wcr)
        self.W_ac_p.append(wac)

        # 逃避者 NN 范数（整体范数）
        self.W_id_s.append(np.linalg.norm(self.evader_agent.identifier.Phi_fs))
        self.W_cr_s.append(np.linalg.norm(self.evader_agent.critic.Phi_cs))
        self.W_ac_s.append(np.linalg.norm(self.evader_agent.actor.Phi_as))

    # ---------------------------------------------------------
    def run(self, T, dt):

        t = 0.0
        steps = int(T / dt)

        print("Starting QUAV pursuit-evasion simulation...")
        for _ in range(steps):
            self.step(t, dt)
            t += dt

        self.time = np.array(self.time)
        self.err = np.array(self.err)
        self.hji = np.array(self.hji)

        self.W_id_p = np.array(self.W_id_p)   # (T, N)
        self.W_cr_p = np.array(self.W_cr_p)
        self.W_ac_p = np.array(self.W_ac_p)

        self.W_id_s = np.array(self.W_id_s)   # (T,)
        self.W_cr_s = np.array(self.W_cr_s)
        self.W_ac_s = np.array(self.W_ac_s)

        self.hist_phi_p = np.array(self.hist_phi_p)
        self.hist_phi_s = np.array(self.hist_phi_s)

        self.pursuer_att_history = np.array(self.pursuer_att_history)
        self.evader_att_history = np.array(self.evader_att_history)

    # ---------------------------------------------------------
    #   Fig.1–3：姿态跟踪 + 误差 + 总误差 & HJI
    # ---------------------------------------------------------
    def plot(self):

        t = self.time
        phi_p = self.hist_phi_p[:, :, 0]    # pursuer roll
        theta_p = self.hist_phi_p[:, :, 1]  # pitch
        psi_p = self.hist_phi_p[:, :, 2]    # yaw

        phi_s = self.hist_phi_s[:, 0]
        theta_s = self.hist_phi_s[:, 1]
        psi_s = self.hist_phi_s[:, 2]

        # ===============================
        # 图 1：姿态跟踪性能（3 子图）
        # ===============================
        plt.figure(figsize=(10, 6))
        plt.suptitle("Tracking Performance of the Attitude Angles")

        # Roll
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(t, phi_p[:, 0], 'b-', label="Single-QUAV attitude")
        ax1.plot(t, phi_s, 'm--', label="Multi-QUAV attitude")
        ax1.set_ylabel(r'$\phi(t)$')
        ax1.grid(True)
        ax1.legend()

        # Pitch
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(t, theta_p[:, 0], 'b-')
        ax2.plot(t, theta_s, 'm--')
        ax2.set_ylabel(r'$\psi(t)$')
        ax2.grid(True)

        # Yaw
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(t, psi_p[:, 0], 'b-')
        ax3.plot(t, psi_s, 'm--')
        ax3.set_ylabel(r'$\theta(t)$')
        ax3.set_xlabel("Time (s)")
        ax3.grid(True)

        # ===============================
        # 图 2：姿态跟踪误差（3 子图）
        # ===============================
        roll_err = phi_p[:, 0] - phi_s
        pitch_err = theta_p[:, 0] - theta_s
        yaw_err = psi_p[:, 0] - psi_s

        plt.figure(figsize=(10, 6))
        plt.suptitle("Tracking Errors")

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(t, roll_err, 'b')
        ax1.set_ylabel(r'$\phi_{pk}(t)$')
        ax1.set_title("Roll tracking error")
        ax1.grid(True)

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(t, pitch_err, 'b')
        ax2.set_ylabel(r'$\psi_{pk}(t)$')
        ax2.set_title("Pitch tracking error")
        ax2.grid(True)

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(t, yaw_err, 'b')
        ax3.set_ylabel(r'$\theta_{pk}(t)$')
        ax3.set_xlabel("Time (s)")
        ax3.set_title("Yaw tracking error")
        ax3.grid(True)

        # ===============================
        # 图 3：追逃总姿态误差
        # ===============================
        plt.figure()
        plt.plot(t, self.err)
        plt.title("Pursuit-Evasion Attitude Error")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [rad]")
        plt.grid(True)

        # ===============================
        # 图 4：HJI 残差
        # ===============================
        plt.figure()
        plt.plot(t, self.hji)
        plt.title("Approximate HJI Residual")
        plt.xlabel("Time [s]")
        plt.ylabel("Residual")
        plt.grid(True)

        plt.show()

    # ---------------------------------------------------------
    #   Fig.4：追击者 Identifier NN 权重范数
    # ---------------------------------------------------------
    def plot_identifier_weights(self):
        Phi = self.W_id_p  # shape (T, N_PURSUERS)

        plt.figure(figsize=(8, 5))
        for k in range(N_PURSUERS):
            plt.plot(self.time, Phi[:, k], label=f"P{k+1}")
        plt.title("Figure 4  Identifier NN Weight Norm (||$\\hat{\\Phi}_{r m}$||)")
        plt.xlabel("Time (s)")
        plt.ylabel("Norm")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ---------------------------------------------------------
    #   Fig.5：追击者 Actor NN 权重范数
    # ---------------------------------------------------------
    def plot_actor_weights(self):
        Phi = self.W_ac_p  # shape (T, N_PURSUERS)

        plt.figure(figsize=(8, 5))
        for k in range(N_PURSUERS):
            plt.plot(self.time, Phi[:, k], label=f"P{k+1}")
        plt.title("Figure 5  Actor NN Weight Norm (||$\\hat{\\Phi}_{a m}$||)")
        plt.xlabel("Time (s)")
        plt.ylabel("Norm")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ---------------------------------------------------------
    #   Fig.6：追击者 Critic NN 权重范数
    # ---------------------------------------------------------
    def plot_critic_weights(self):
        Phi = self.W_cr_p  # shape (T, N_PURSUERS)

        plt.figure(figsize=(8, 5))
        for k in range(N_PURSUERS):
            plt.plot(self.time, Phi[:, k], label=f"P{k+1}")
        plt.title("Figure 6  Critic NN Weight Norm (||$\\hat{\\Phi}_{c m}$||)")
        plt.xlabel("Time (s)")
        plt.ylabel("Norm")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ---------------------------------------------------------
    #   Fig.7：逃避者 Critic / Actor NN 权重范数
    # ---------------------------------------------------------
    def plot_evader_weights(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.time, self.W_cr_s, label=r"$||\hat{\Phi}_{ce}||$")
        plt.plot(self.time, self.W_ac_s, label=r"$||\hat{\Phi}_{ae}||$")
        plt.title("Figure 7  Evader NN Weight Norms")
        plt.xlabel("Time (s)")
        plt.ylabel("Norm")
        plt.grid(True)
        plt.legend()
        plt.show()
