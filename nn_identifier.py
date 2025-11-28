# nn_identifier.py
import numpy as np


class PursuerIdentifier:
    """
    追击者 m 的 Identifier NN，对应论文 (31)(32)
    近似 Fs*_m(\bar x_m)，其中 \bar x_m = [x_pm^T, x_vm^T]^T ∈ R^6

    加了两点稳定性：
    1) 输入 \bar x_m 归一化到有界范围，避免 RBF 爆掉
    2) 权重 Φ_fm 在更新后做范数裁剪，防止权重无限增大
    """

    def __init__(
        self,
        state_dim: int = 6,
        out_dim: int = 3,
        num_centers: int = 15,
        phi_fm: float = 5.0,
        sigma_fm: float = 0.01,
        seed: int | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.state_dim = state_dim
        self.out_dim = out_dim
        self.q1 = num_centers

        # RBF 中心：状态大致在 [-0.5, 0.5] 内波动，取稍大范围
        self.centers = np.random.uniform(-0.5, 0.5, (num_centers, state_dim))
        # beta 控制高斯宽度（越小越“宽”）
        self.beta = np.ones(num_centers) * 0.1

        # 权重矩阵 Φ̂_fm(t) ∈ R^{q1 × 3}
        self.Phi_fm = np.zeros((num_centers, out_dim))

        # 更新律参数
        self.phi_fm = phi_fm
        self.sigma_fm = sigma_fm

        # 权重范数上限（可根据需要调）
        self.weight_clip = 10.0

    # ---------- 内部小工具：权重裁剪 ----------
    def _clip_weights(self):
        norm = np.linalg.norm(self.Phi_fm)
        if norm > self.weight_clip:
            self.Phi_fm *= self.weight_clip / (norm + 1e-8)

    # ---------- RBF 基函数 ξ_fm(\bar x_m) ----------
    def basis(self, x_bar: np.ndarray) -> np.ndarray:
        """
        x_bar: shape (state_dim,)
        return: (q1,)
        """
        x_bar = np.asarray(x_bar).reshape(-1)

        # 归一化输入，避免出现超大状态导致 diff**2 溢出
        norm_x = np.linalg.norm(x_bar)
        x_bar = x_bar / (1.0 + norm_x)

        diff = self.centers - x_bar  # (q1, state_dim)
        r2 = np.sum(diff * diff, axis=1)
        xi = np.exp(-self.beta * r2)
        return xi

    def forward(self, x_bar: np.ndarray):
        """
        F̂_sm(\bar x_m) = Φ̂_fm^T ξ_fm(\bar x_m)
        返回:
            F_hat: (out_dim,)
            xi_fm: (q1,)
        """
        xi = self.basis(x_bar)
        F_hat = xi @ self.Phi_fm
        return F_hat, xi

    def update(self, x_bar: np.ndarray, Es_m: np.ndarray, dt: float):
        """
        Φ̂_fm(k+1) = Φ̂_fm(k) + dt * φ_fm ( ξ_fm Es_m^T - σ_fm Φ̂_fm )
        Es_m: (3,)
        """
        x_bar = np.asarray(x_bar).reshape(-1)
        Es_m = np.asarray(Es_m).reshape(-1)

        xi = self.basis(x_bar)                      # (q1,)
        outer = np.outer(xi, Es_m)                  # (q1, 3)
        dPhi = self.phi_fm * (outer - self.sigma_fm * self.Phi_fm)
        self.Phi_fm += dt * dPhi

        # 防止权重爆炸
        self._clip_weights()
