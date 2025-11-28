import numpy as np
import matplotlib.pyplot as plt
from simulation import QUAVGameSimulation
from config import T_FINAL, DT

# 使用 Times New Roman + LaTeX
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'      # 数学字体 Computer Modern
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

print("Running simulation for tracking plots...")
sim = QUAVGameSimulation()
sim.run(T_FINAL, DT)

# 数据
time = sim.time

phi_s = sim.hist_phi_s[:, 0]
theta_s = sim.hist_phi_s[:, 1]
psi_s = sim.hist_phi_s[:, 2]

phi_p = sim.hist_phi_p[:, 0, 0]
theta_p = sim.hist_phi_p[:, 0, 1]
psi_p = sim.hist_phi_p[:, 0, 2]

# tracking error
phi_err = phi_p - phi_s
theta_err = theta_p - theta_s
psi_err = psi_p - psi_s

LW = 1.8   # 线条宽度


# =====================================================
# Figure 2 — Attitude tracking
# =====================================================
fig1 = plt.figure(figsize=(10, 8))

# Roll
ax1 = fig1.add_subplot(3, 1, 1)
ax1.plot(time, phi_p, 'b', linewidth=LW, label="Single-QUAV attitude")
ax1.plot(time, phi_s, 'm--', linewidth=LW, label="Multi-QUAV attitude")
ax1.set_ylabel(r"$\phi(t)$", fontsize=14)
ax1.grid(True, linewidth=0.8)
ax1.legend(fontsize=12, loc="upper right")

# Pitch
ax2 = fig1.add_subplot(3, 1, 2)
ax2.plot(time, theta_p, 'b', linewidth=LW)
ax2.plot(time, theta_s, 'm--', linewidth=LW)
ax2.set_ylabel(r"$\psi(t)$", fontsize=14)
ax2.grid(True, linewidth=0.8)

# Yaw
ax3 = fig1.add_subplot(3, 1, 3)
ax3.plot(time, psi_p, 'b', linewidth=LW)
ax3.plot(time, psi_s, 'm--', linewidth=LW)
ax3.set_ylabel(r"$\theta(t)$", fontsize=14)
ax3.set_xlabel("Time (s)", fontsize=14)
ax3.grid(True, linewidth=0.8)

fig1.suptitle("Tracking Performance of the Attitude Angles", fontsize=16)
fig1.tight_layout(rect=[0, 0, 1, 0.95])


# =====================================================
# Figure 3 — Tracking error
# =====================================================
fig2 = plt.figure(figsize=(10, 8))

# Roll error
ax1 = fig2.add_subplot(3, 1, 1)
ax1.plot(time, phi_err, 'b', linewidth=LW)
ax1.set_ylabel(r"$\phi_{pk}(t)$", fontsize=14)
ax1.set_title("Roll tracking error", fontsize=15)
ax1.grid(True, linewidth=0.8)

# Pitch error
ax2 = fig2.add_subplot(3, 1, 2)
ax2.plot(time, theta_err, 'b', linewidth=LW)
ax2.set_ylabel(r"$\psi_{pk}(t)$", fontsize=14)
ax2.set_title("Pitch tracking error", fontsize=15)
ax2.grid(True, linewidth=0.8)

# Yaw error
ax3 = fig2.add_subplot(3, 1, 3)
ax3.plot(time, psi_err, 'b', linewidth=LW)
ax3.set_ylabel(r"$\theta_{pk}(t)$", fontsize=14)
ax3.set_title("Yaw tracking error", fontsize=15)
ax3.set_xlabel("Time (s)", fontsize=14)
ax3.grid(True, linewidth=0.8)

fig2.suptitle("Tracking Errors", fontsize=16)
fig2.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
