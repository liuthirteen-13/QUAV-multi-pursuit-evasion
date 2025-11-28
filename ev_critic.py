# ev_critic.py
import numpy as np


class EvaderCritic:
    """
    Evader Critic NN (max-HJI side)
    -------------------------------
    近似 dJ_s^*/dE_s，对应论文中逃避者一侧的 HJI 梯度。

    结构与 PursuerCritic 对称，只是参数命名换成 P2, lambda_e, mu_e。
    """

    def __init__(
        self,
        Es_dim: int = 3,      # 实际只用前三维姿态误差
        xbar_dim: int = 6,
        out_dim: int = 3,
        num_centers: int = 10,   # q2
        P2: float = 1.0,
        lambda_e: float = 1.0,
        mu_e: float = 3.0,
        identifier=None,         # EvaderIdentifier
        kappa_ce: float = 0.05,
        sigma_e: float = 0.5,
        w_max: float | None = None,
        seed: int | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.Es_dim = Es_dim
        self.xbar_dim = xbar_dim
        self.in_dim = Es_dim + xbar_dim
        self.out_dim = out_dim
        self.q2 = num_centers

        self.P2 = P2
        self.lambda_e = lambda_e
        self.mu_e = mu_e

        self.identifier = identifier
        self.kappa_ce = kappa_ce
        self.sigma_e = sigma_e
        self.w_max = w_max

        # RBF 节点
        self.centers = np.random.uniform(-1.0, 1.0, size=(self.q2, self.in_dim))
        self.rho = 0.8 * np.ones(self.q2)

        # Critic 权重 Φ_cs ∈ R^{q2×3}
        self.Phi_cs = 0.1 * np.random.randn(self.q2, self.out_dim)

    # ----------------------------------------------------
    def _rbf_features(self, Es, x_bar):
        """
        ξ_s(E_s, \bar x_s) ∈ R^{q2}
        """
        Es = np.asarray(Es).reshape(-1)
        x_bar = np.asarray(x_bar).reshape(-1)
        # 只用前三维 Es（姿态误差），避免速度噪声
        Es3 = Es[0:3]

        v = np.concatenate([Es3, x_bar], axis=0)  # (3+6=9,) 输入

        diff = v[None, :] - self.centers          # (q2, in_dim)
        dist2 = np.sum(diff ** 2, axis=1)
        xi = np.exp(-dist2 / (2.0 * (self.rho ** 2 + 1e-8)))
        return xi

    # ----------------------------------------------------
    def grad_J(self, Es, x_bar):
        """
        dJ_s^*/dE_s 近似
        """
        Es = np.asarray(Es).reshape(-1)
        x_bar = np.asarray(x_bar).reshape(-1)
        Es3 = Es[0:3]

        # 第一项：与 pursuer 相同量级
        term1 = (2.0 * self.P2 * self.mu_e / self.lambda_e) * Es3  # (3,)

        # 第二项：Identifier 关联项
        term2 = np.zeros_like(Es3)
        if self.identifier is not None:
            try:
                _, xi_fs = self.identifier.forward(x_bar)
                Phi_fs = self.identifier.Phi_fs
                term2 = (2.0 * self.P2 / self.lambda_e) * (Phi_fs.T @ xi_fs)
            except Exception:
                term2 = 0.0 * Es3

        # 第三项：Critic 自身 RBF 项
        xi_s = self._rbf_features(Es, x_bar)
        term3 = (self.P2 / self.lambda_e) * (self.Phi_cs.T @ xi_s)  # (3,)

        dJdEs3 = term1 + term2 + term3
        return dJdEs3

    # ----------------------------------------------------
    def update(self, Es, x_bar, dt):
        """
        Φ̇_cs = - κ_ce ( ξ ξ^T + σ_e I ) Φ_cs
        """
        Es = np.asarray(Es).reshape(-1)
        x_bar = np.asarray(x_bar).reshape(-1)

        xi = self._rbf_features(Es, x_bar)          # (q2,)
        G = np.outer(xi, xi) + self.sigma_e * np.eye(self.q2)

        Phi_dot = - self.kappa_ce * (G @ self.Phi_cs)
        self.Phi_cs += dt * Phi_dot

        if self.w_max is not None:
            norm = np.linalg.norm(self.Phi_cs)
            if norm > self.w_max:
                self.Phi_cs *= (self.w_max / (norm + 1e-8))
