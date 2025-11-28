from simulation import QUAVGameSimulation
from config import DT, T_FINAL
from uav_vpython_animation import animate_uavs
import numpy as np


if __name__ == "__main__":
    print("Starting QUAV pursuit-evasion simulation...")

    sim = QUAVGameSimulation()
    sim.run(T_FINAL, DT)

    print("Simulation finished. Saving data...")

    # ------------ 画图部分(可选) ------------
    sim.plot()
    sim.plot_identifier_weights()
    sim.plot_actor_weights()
    sim.plot_critic_weights()
    sim.plot_evader_weights()

    import matplotlib.pyplot as plt
    plt.show()

    # ------------ VPython 动画(可选) ------------
    animate_uavs(sim.pursuer_att_history, sim.evader_att_history)

    print("Animation finished.")
