# QUAV Multi-UAV Pursuit–Evasion Simulation

A multi-UAV pursuit–evasion simulation framework based on attitude dynamics, sliding-mode control, and reinforcement learning (I-DAC for pursuers & min–max evader strategy).  
Supports full 3D QUAV dynamics, neural-network actor-critic training, pursuer–evader interaction, and VPython animation.

---

##  Features

- **Full QUAV Attitude Dynamics**
  - RK4 integration  
  - Sliding-mode tracking variables  
  - Moment & torque allocation  

- **Reinforcement Learning Controllers**
  - Pursuer controller: **I-DAC (Identifier-Dynamic Actor-Critic)**  
  - Evader controller: **Min–Max policy with neural network (identifier/critic/actor)**  
  - Online adaptation using neural identifiers  

- **Graph-based Cooperative Pursuit**
  - Laplacian interaction topology  
  - Distributed error propagation  

- **3D Visualization**
  - VPython real-time animation  
  - Pursuer/evader trajectories  
  - Orientation updates, rotor visualization  

- **Experiment Tools**
  - Attitude tracking plots  
  - Pursuer–evader distance over time  
  - Simulation playback  

---

##  Project Structure

