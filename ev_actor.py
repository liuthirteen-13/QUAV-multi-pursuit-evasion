# ev_actor.py
import numpy as np


class EvaderActor:
    """
    Evader Actor NN  —— 方案 C
    --------------------------
    1. 基于 Es_s 的“反向”线性反馈（ +μ_s E_s ），与追击者形成 min-max 对抗；
    2. 利用 Critic 的 RBF 特征 + 权重 Φ_as 近似非线性最优项；
    3. 叠加一个固定振幅的正弦扰动，增加“逃避难度感”和持久激励；
    4. 权重跟踪律：让 Φ_as 跟随 -Φ_cs（与追击者的结构相对）。
    """

    def __init__(
        self,
        Es_dim: int,
        xbar_dim: int,
        mu_s: float,
        critic,
        num_centers: int,
        kappa_ae: float = 0.08,
        sigma_ae: float = 0.2,
        u_max: float = 1.0,
        sin_amp: float = 0.15,
        sin_freqs: tuple[float, float, float] = (0.6, 0.9, 1.2),
        seed: int | None = None,
    ):
        self.Es_dim = Es_dim
        self.xbar_dim = xbar_dim
        self.mu_s = mu_s

        self.critic = critic
        self.num_centers = num_centers
        self.out_dim = critic.out_dim

        self.kappa_ae = kappa_ae
        self.sigma_ae = sigma_ae
        self.u_max = u_max

        self.sin_amp = sin_amp
        self.omega = np.array(sin_freqs, dtype=float)

        if seed is not None:
            np.random.seed(seed)

        # Actor 权重初始化为 Critic 的“对立值”
        self.Phi_as = -self.critic.Phi_cs.copy()

    # ----------------------------------------------------
    def _features(self, Es, x_bar):
        """
        直接复用 critic 的 RBF 特征，保证特征空间一致。
        """
        return self.critic._rbf_features(Es, x_bar)

    # ----------------------------------------------------
    def policy(self, Es_s, x_bar_s, t: float) -> np.ndarray:
        """
        u_s = + μ_s E_s + 0.5 Φ_as^T ξ_s + u_PE(t)
        其中:
          u_PE(t) 为设计的正弦扰动项（方案 C）
        """
        Es_s = np.asarray(Es_s).reshape(-1)
        x_bar_s = np.asarray(x_bar_s).reshape(-1)
        Es3 = Es_s[0:3]

        xi = self._features(Es_s, x_bar_s)    # (q2,)
        nn_term = xi @ self.Phi_as           # (3,)

        # 反向线性反馈 + NN 补偿
        u_base = + self.mu_s * Es3 + 0.5 * nn_term

        # 正弦扰动（保证每个通道相位不同）
        phase = np.array([0.0, np.pi / 3, 2.0 * np.pi / 3])
        u_pe = self.sin_amp * np.sin(self.omega * t + phase)

        u = u_base + u_pe
        u = np.clip(u, -self.u_max, self.u_max)
        return u

    # ----------------------------------------------------
    def update(self, dt: float):
        """
        Φ̇_as = - κ_ae (Φ_as + Φ_cs) - σ_ae Φ_as
        说明:
          - 追踪 -Φ_cs，使得结构上与追击者“对立”（min-max 感更强）；
          - 加 σ_ae 抑制发散。
        """
        target = -self.critic.Phi_cs
        Phi_err = self.Phi_as - target

        Phi_dot = -self.kappa_ae * Phi_err - self.sigma_ae * self.Phi_as
        self.Phi_as += dt * Phi_dot
