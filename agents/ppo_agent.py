"""Proximal Policy Optimization (PPO) Agent with clipped surrogate objective."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base_agent import BaseAgent


class ActorCriticNetwork(nn.Module):
    """Shared-backbone Actor-Critic network for PPO."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.actor = nn.Sequential(nn.Linear(hidden_size, action_size), nn.Softmax(dim=-1))
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        feat = self.shared(x)
        return self.actor(feat), self.critic(feat)


class PPOAgent(BaseAgent):
    """
    PPO with clipped surrogate loss and Generalized Advantage Estimation (GAE).

    Collects a full episode trajectory, then performs multiple epochs of
    mini-batch updates — the key difference from A2C.
    """

    def __init__(self, state_size: int, action_size: int, config: dict = None):
        super().__init__(state_size, action_size, config)
        c = self.config
        self.gamma = c.get("gamma", 0.99)
        self.gae_lambda = c.get("gae_lambda", 0.95)
        self.clip_ratio = c.get("clip_ratio", 0.2)
        self.lr = c.get("learning_rate", 3e-4)
        self.hidden_size = c.get("hidden_size", 256)
        self.entropy_coef = c.get("entropy_coef", 0.01)
        self.value_coef = c.get("value_coef", 0.5)
        self.n_epochs = c.get("n_epochs", 4)
        self.batch_size = c.get("batch_size", 64)
        self.max_grad_norm = c.get("max_grad_norm", 0.5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Trajectory buffer (reset each episode)
        self._reset_buffer()
        self.episodes_done = 0
        self._last_loss = 0.0

    def _reset_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs_old = []
        self.rewards = []
        self.values = []
        self.dones = []

    def select_action(self, state, training: bool = True) -> int:
        if isinstance(state, int):
            s = torch.zeros(self.state_size, dtype=torch.float32).to(self.device)
            s[state] = 1.0
        else:
            s = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            probs, value = self.network(s.unsqueeze(0))

        dist = Categorical(probs)
        if training:
            action = dist.sample()
        else:
            action = probs.argmax(dim=-1)

        if training:
            self.log_probs_old.append(dist.log_prob(action).item())
            self.values.append(value.squeeze().item())

        return action.item()

    def store_experience(self, state, action, reward, next_state, done):
        if isinstance(state, int):
            s = np.zeros(self.state_size, dtype=np.float32)
            s[state] = 1.0
        else:
            s = np.array(state, dtype=np.float32)
        self.states.append(s)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def train_step(self) -> float:
        # PPO trains at end_episode, not per-step
        return self._last_loss

    def end_episode(self):
        if len(self.rewards) == 0:
            return

        # ── Compute GAE advantages ────────────────────────────────────────────
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        # Bootstrap value for last step
        last_val = 0.0
        if not self.dones[-1]:
            last_state = self.states[-1]
            s = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, v = self.network(s)
            last_val = v.item()

        gae = 0.0
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * next_val * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Convert to tensors ────────────────────────────────────────────────
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs_old).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # ── PPO update: multiple epochs over mini-batches ─────────────────────
        total_loss = 0.0
        indices = np.arange(T)
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]

                probs, values = self.network(states_t[batch_idx])
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions_t[batch_idx])
                entropy = dist.entropy().mean()

                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                adv_b = advantages_t[batch_idx]
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = nn.functional.mse_loss(values.squeeze(), returns_t[batch_idx])

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

        self._last_loss = total_loss / (self.n_epochs * max(1, T // self.batch_size))
        self.episodes_done += 1
        self._reset_buffer()

    def save(self, filepath: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "episodes_done": self.episodes_done,
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episodes_done = ckpt.get("episodes_done", 0)
