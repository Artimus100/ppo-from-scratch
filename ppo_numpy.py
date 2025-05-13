import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
    
# PPO implementation using PyTorch
# This code implements the Proximal Policy Optimization (PPO) algorithm for reinforcement learning.
# It uses a simple feedforward neural network as the policy and value function approximator.
# The code is designed to work with the OpenAI Gym environment, specifically the CartPole-v1 environment.
# The PPO algorithm is a popular policy gradient method that uses a clipped objective function to stabilize training.
# It collects trajectories from the environment, computes Generalized Advantage Estimation (GAE) for advantage estimation,
# and performs multiple epochs of mini-batch updates to the policy and value networks.

# Hyperparameters
env_name = "CartPole-v1"
gamma = 0.99
lr = 2.5e-4
clip_epsilon = 0.2
update_epochs = 4
steps_per_update = 2048
mini_batch_size = 64
gae_lambda = 0.95
entropy_coef = 0.01
value_loss_coef = 0.5


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

    def get_action_and_value(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

def compute_gae(rewards, values, dones, next_value):
    values = np.append(values, next_value)
    gae = 0
    advantages = np.zeros_like(rewards)
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
        advantages[step] = gae
    return advantages, advantages + values[:-1]

def collect_trajectories(env, model):
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    state, _ = env.reset()
    for _ in range(steps_per_update):
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(state_tensor)
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state)
        actions.append(action.item())
        log_probs.append(log_prob.item())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())

        state = next_state if not done else env.reset()[0]

    with torch.no_grad():
        next_value = model(torch.FloatTensor(np.array(state)).unsqueeze(0))[1].item()

    return (
        np.array(states),
        np.array(actions),
        np.array(log_probs),
        np.array(rewards),
        np.array(dones),
        np.array(values),
        next_value,
    )

def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages):
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)
    returns = torch.FloatTensor(returns)
    advantages = torch.FloatTensor(advantages)

    for _ in range(update_epochs):
        for start in range(0, len(states), mini_batch_size):
            end = start + mini_batch_size
            s_batch = states[start:end]
            a_batch = actions[start:end]
            old_lp_batch = old_log_probs[start:end]
            ret_batch = returns[start:end]
            adv_batch = advantages[start:end]

            logits, values = model(s_batch)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(a_batch)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_lp_batch).exp()
            clip_adv = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_batch
            policy_loss = -torch.min(ratio * adv_batch, clip_adv).mean()
            value_loss = (ret_batch - values.squeeze()).pow(2).mean()

            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

env = gym.make(env_name)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model = ActorCritic(obs_dim, act_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

for iteration in range(1000):
    states, actions, log_probs, rewards, dones, values, next_val = collect_trajectories(env, model)
    advantages, returns = compute_gae(rewards, values, dones, next_val)
    ppo_update(model, optimizer, states, actions, log_probs, returns, advantages)

    avg_reward = sum(rewards) / len(rewards)
    print(f"Iteration {iteration} — Avg reward: {avg_reward:.2f}")
