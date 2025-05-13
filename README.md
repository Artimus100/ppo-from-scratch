# PPO-CartPole 🧠🎢

This repo implements the **Proximal Policy Optimization (PPO)** algorithm using **PyTorch** to solve the classic **CartPole-v1** environment from OpenAI Gym. The agent learns to balance a pole on a moving cart using reinforcement learning principles.

---

## 📌 Features

- 🚀 PPO algorithm with clipping
- 🎯 Advantage estimation using **GAE**
- 🧠 Shared actor-critic neural network
- 📈 Tracks episode rewards and action distribution
- 🧪 Compatible with Gymnasium and classic Gym

---

## 📚 PPO Overview

**Proximal Policy Optimization (PPO)** is a policy gradient method that improves training stability by constraining how much the policy can change at each step. It uses:

- A **clipped surrogate objective** to prevent overly large policy updates.
- **Actor-Critic architecture**: shared features with separate policy and value heads.
- **GAE (Generalized Advantage Estimation)** for better advantage estimates.
- **Mini-batch updates** from collected trajectories.

---

## 🛠️ Requirements

- `gym` or `gymnasium`
- `torch`
- `numpy`

Install with:

```bash
pip install gym torch numpy
