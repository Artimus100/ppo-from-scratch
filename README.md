![Image](/result.png?raw=true )

# PPO-CartPole 🧠🎢

This repo implements the **Proximal Policy Optimization (PPO)** algorithm using **PyTorch** to solve the classic **CartPole-v1** environment from OpenAI Gym. The agent learns to balance a pole on a moving cart using reinforcement learning.

---

## 📌 Features

- 🚀 PPO algorithm with clipped surrogate objective
- 🎯 Advantage estimation using **GAE** (Generalized Advantage Estimation)
- 🧠 Shared actor-critic neural network architecture
- 📈 Tracks episode rewards and action distribution during training
- 🧪 Compatible with both Gymnasium and classic Gym

---

## 📚 PPO Overview

**Proximal Policy Optimization (PPO)** is a policy gradient method that improves training stability by constraining how much the policy can change at each update step. It uses:

- A **clipped surrogate objective** to prevent large policy updates that might destabilize training
- **Actor-Critic architecture** with shared feature layers and separate policy (actor) and value (critic) heads
- **GAE (Generalized Advantage Estimation)** for improved advantage calculation, reducing variance
- **Mini-batch updates** on collected trajectories for efficient optimization

---

## 🛠️ Requirements

- `gym` or `gymnasium`
- `torch`
- `numpy`
- `matplotlib` (for reward plotting)

Install dependencies with:

```bash
pip install gym torch numpy matplotlib
