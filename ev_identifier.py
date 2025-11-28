# ev_identifier.py
import numpy as np


class EvaderIdentifier:
    """
    Evader Identifier NN
    --------------------
    近似 F_s^*(\bar x_s)，输入为逃避者的扩展状态 \bar x_s ∈ R^6，
    输出为 3 维等效“虚拟扰动项”。

    结构基本和 PursuerIdentifier 一致，但参数更“温和”，防止发散。
    """

    def __init__(
        self,
        state_dim: int = 6,
        out_dim: int = 3,
        num_centers: int = 10,   # q1
        phi_fs: float = 0.3,     # 学习率
        sigma_fs: float = 0.05,  # 正则
        w_max: float | None = None,
        seed: int | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.state_dim = state_dim
        self.out_dim = out_dim
        self.q1 = num_centers

        # RBF 中心：姿态和角速度都不算太大，取 [-0.5, 0.5]
        self.centers = np.random.uniform(-0.5, 0.5, (self.q1, self.state_dim))
        # 高斯宽度
        self.beta = np.ones(self.q1) * 0.1

        # 权重矩阵 Φ_fs ∈ R^{q1×3}
        self.Phi_fs = np.zeros((self.q1, self.out_dim))

        self.phi_fs = phi_fs
        self.sigma_fs = sigma_fs
        self.w_max = w_max

    # ----------------------------------------------------
    def basis(self, x_bar: np.ndarray) -> np.ndarray:
        """
        ξ_fs(\bar x_s) ∈ R^{q1}
        """
        x_bar = np.asarray(x_bar).reshape(-1)
        diff = self.centers - x_bar  # (q1, state_dim)
        r2 = np.sum(diff * diff, axis=1)
        xi = np.exp(-self.beta * r2)
        return xi

    # ----------------------------------------------------
    def forward(self, x_bar: np.ndarray):
        """
        ˆF_s(\bar x_s) = Φ_fs^T ξ_fs(\bar x_s)
        返回:
          F_hat: (3,)
          xi_fs: (q1,)
        """
        xi = self.basis(x_bar)         # (q1,)
        F_hat = xi @ self.Phi_fs       # (3,)
        return F_hat, xi

    # ----------------------------------------------------
    def update(self, x_bar: np.ndarray, Es_s: np.ndarray, dt: float):
        """
        权重更新（误差驱动）:
          Φ̇_fs = φ_fs ( ξ_fs Es_s^T - σ_fs Φ_fs )
        """
        x_bar = np.asarray(x_bar).reshape(-1)
        Es_s = np.asarray(Es_s).reshape(-1)  # (3,)

        xi = self.basis(x_bar)              # (q1,)
        outer = np.outer(xi, Es_s)          # (q1, 3)

        dPhi = self.phi_fs * (outer - self.sigma_fs * self.Phi_fs)
        self.Phi_fs += dt * dPhi

        # 可选权重范数限制
        if self.w_max is not None:
            norm = np.linalg.norm(self.Phi_fs)
            if norm > self.w_max:
                self.Phi_fs *= (self.w_max / (norm + 1e-8))
