# nn_actor.py
import numpy as np
from nn_critic import Critic


class Actor:
    """
    Pursuer Actor NN（论文风格 I–DAC 版本）

    控制律：
        u_zm = - μ_m E_s - 1/2 Φ_am^T ξ_m(E_s, x̄_m)

    权重更新：
        Φ̇_am = - κ_am (Φ_am - Φ_cm) - σ_am Φ_am
    """

    def __init__(
        self,
        Es_dim: int,
        xbar_dim: int,
        mu_p: float,
        critic: Critic,
        num_centers: int,
        kappa_am: float = 1.0,
        sigma_am: float = 0.1,
        seed: int | None = None,
    ):
        self.Es_dim = Es_dim
        self.xbar_dim = xbar_dim
        self.mu_p = mu_p

        self.critic = critic
        self.num_centers = num_centers
        self.out_dim = critic.out_dim

        self.kappa_am = kappa_am
        self.sigma_am = sigma_am

        if seed is not None:
            np.random.seed(seed)

        # 初始 Actor 权重为 Critic 权重
        self.Phi_am = self.critic.Phi_cm.copy()
        self.weight_clip = 20.0

    def _clip_weights(self):
        norm = np.linalg.norm(self.Phi_am)
        if norm > self.weight_clip:
            self.Phi_am *= self.weight_clip / (norm + 1e-8)

    def _features(self, Es: np.ndarray, x_bar: np.ndarray) -> np.ndarray:
        return self.critic._rbf_features(Es, x_bar)

    def policy(self, Es: np.ndarray, x_bar: np.ndarray) -> np.ndarray:
        Es = np.asarray(Es).reshape(-1)
        x_bar = np.asarray(x_bar).reshape(-1)

        xi = self._features(Es, x_bar)
        nn_term = xi @ self.Phi_am

        u = -self.mu_p * Es - 0.5 * nn_term
        return u

    def update(self, dt: float):
        target = self.critic.Phi_cm
        Phi_err = self.Phi_am - target
        Phi_dot = -self.kappa_am * Phi_err - self.sigma_am * self.Phi_am
        self.Phi_am += dt * Phi_dot

        self._clip_weights()
