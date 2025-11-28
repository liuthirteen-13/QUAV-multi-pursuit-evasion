import numpy as np


class QUAVParams:
    """
    存储单架无人机的惯量、阻尼等参数
    对应论文中的 I_x, I_y, I_z, G_φ, G_ψ, G_θ, 臂长 l
    """
    def __init__(self,
                 Ix=0.03, Iy=0.03, Iz=0.05,
                 G_phi=0.02, G_psi=0.02, G_theta=0.02,
                 arm_length=0.25):
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.G_phi = G_phi
        self.G_psi = G_psi
        self.G_theta = G_theta
        self.l = arm_length


class QUAVAttitudeDynamics:
    """
    实现论文中姿态动力学方程 (1)(2)(3)
    状态：state = [φ, ψ, θ, φ_dot, ψ_dot, θ_dot]
    控制：u = [u_φ, u_ψ, u_θ]
    """
    def __init__(self, params: QUAVParams):
        self.p = params

    def f(self, state, u):
        phi, psi, theta, dphi, dpsi, dtheta = state
        u_phi, u_psi, u_theta = u

        Ix, Iy, Iz = self.p.Ix, self.p.Iy, self.p.Iz
        G_phi, G_psi, G_theta = self.p.G_phi, self.p.G_psi, self.p.G_theta
        l = self.p.l

        ddphi = u_phi + dpsi * dtheta * (Iy - Iz) / Ix - (G_phi * l / Ix) * dphi
        ddpsi = u_psi + dphi * dtheta * (Iz - Ix) / Iy - (G_psi * l / Iy) * dpsi
        ddtheta = u_theta + dphi * dpsi * (Ix - Iy) / Iz - (G_theta * l / Iz) * dtheta

        return np.array([dphi, dpsi, dtheta, ddphi, ddpsi, ddtheta])


def rk4_step(dyn: QUAVAttitudeDynamics, state, u, dt):
    k1 = dyn.f(state, u)
    k2 = dyn.f(state + 0.5 * dt * k1, u)
    k3 = dyn.f(state + 0.5 * dt * k2, u)
    k4 = dyn.f(state + dt * k3, u)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
