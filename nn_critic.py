# nn_critic.py  —— 论文版 Critic，对应式 (33)(34)
import numpy as np
from nn_identifier import PursuerIdentifier


class Critic:
    """
    论文版 Critic NN，对应公式 (33)(34)
    ----------------------------------
    dJ_hat/dEs = (2P1/lambda_m)*mu_m*Es
                 + (2P1/lambda_m)*Phi_fm^T * xi_fm(x_bar)
                 + ( P1/lambda_m)*Phi_cm^T * xi_m(Es, x_bar)

    Phi_cm 更新律：
        Phi_cm_dot = -kappa_cm * ( xi xi^T + sigma_m I ) Phi_cm

    稳定性增强：
    - RBF 输入向量 v 做归一化
    - 权重 Phi_cm 做范数裁剪
    """

    def __init__(
        self,
        Es_dim: int = 3,
        xbar_dim: int = 6,
        out_dim: int = 3,
        num_centers: int = 15,    # q2
        P1: float = 1.0,
        lambda_m: float = 1.0,
        mu_m: float = 3.0,
        identifier: PursuerIdentifier | None = None,
        kappa_cm: float = 1.0,
        sigma_m: float = 0.1,
        seed: int | None = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.Es_dim = Es_dim
        self.xbar_dim = xbar_dim
        self.in_dim = Es_dim + xbar_dim
        self.out_dim = out_dim
        self.q2 = num_centers

        self.P1 = P1
        self.lambda_m = lambda_m
        self.mu_m = mu_m

        self.identifier = identifier

        self.kappa_cm = kappa_cm
        self.sigma_m = sigma_m

        # RBF 节点
        self.centers = np.random.uniform(-1.0, 1.0, size=(self.q2, self.in_dim))
        self.rho = 0.8 * np.ones(self.q2)

        # Critic 权重
        self.Phi_cm = 0.1 * np.random.randn(self.q2, self.out_dim)

        self.weight_clip = 20.0

    def _clip_weights(self):
        norm = np.linalg.norm(self.Phi_cm)
        if norm > self.weight_clip:
            self.Phi_cm *= self.weight_clip / (norm + 1e-8)

    # RBF 特征
    def _rbf_features(self, Es: np.ndarray, x_bar: np.ndarray) -> np.ndarray:
        Es = np.asarray(Es).reshape(-1)
        x_bar = np.asarray(x_bar).reshape(-1)
        v = np.concatenate([Es, x_bar], axis=0)

        # 输入归一化避免 diff**2 溢出
        nv = np.linalg.norm(v)
        v = v / (1.0 + nv)

        diff = v[None, :] - self.centers
        dist2 = np.sum(diff ** 2, axis=1)
        xi = np.exp(-dist2 / (2.0 * (self.rho ** 2 + 1e-8)))
        return xi

    def grad_J(self, Es: np.ndarray, x_bar: np.ndarray) -> np.ndarray:
        Es = np.asarray(Es).reshape(-1)
        x_bar = np.asarray(x_bar).reshape(-1)

        term1 = (2.0 * self.P1 * self.mu_m / self.lambda_m) * Es

        # Identifier 项
        term2 = np.zeros_like(Es)
        if self.identifier is not None:
            try:
                _, xi_fm = self.identifier.forward(x_bar)
                Phi_fm = self.identifier.Phi_fm
                term2 = (2.0 * self.P1 / self.lambda_m) * (Phi_fm.T @ xi_fm)
            except Exception:
                term2 = 0.0 * Es

        # Critic RBF 项
        xi_m = self._rbf_features(Es, x_bar)
        term3 = (self.P1 / self.lambda_m) * (self.Phi_cm.T @ xi_m)

        return term1 + term2 + term3

    def update(self, Es: np.ndarray, x_bar: np.ndarray, dt: float):
        Es = np.asarray(Es).reshape(-1)
        x_bar = np.asarray(x_bar).reshape(-1)

        xi = self._rbf_features(Es, x_bar)
        G = np.outer(xi, xi) + self.sigma_m * np.eye(self.q2)
        Phi_dot = -self.kappa_cm * (G @ self.Phi_cm)
        self.Phi_cm += dt * Phi_dot

        self._clip_weights()
