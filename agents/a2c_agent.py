"""Advantage Actor-Critic (A2C) Agent."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base_agent import BaseAgent


class ActorCriticNetwork(nn.Module):
    """Shared-backbone Actor-Critic network."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(hidden_size, action_size), nn.Softmax(dim=-1))
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        feat = self.shared(x)
        return self.actor(feat), self.critic(feat)


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic with n-step returns."""

    def __init__(self, state_size: int, action_size: int, config: dict = None):
        super().__init__(state_size, action_size, config)
        c = self.config
        self.gamma = c.get("gamma", 0.99)
        self.lr = c.get("learning_rate", 1e-3)
        self.hidden_size = c.get("hidden_size", 128)
        self.entropy_coef = c.get("entropy_coef", 0.01)
        self.value_coef = c.get("value_coef", 0.5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Episode buffer
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.dones = []
        self.episodes_done = 0

    def select_action(self, state, training: bool = True) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.network(s)
        dist = Categorical(probs)
        action = dist.sample()
        if training:
            self.log_probs.append(dist.log_prob(action))
            self.values.append(value.squeeze())
            self.entropies.append(dist.entropy())
        return action.item()

    def store_experience(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def train_step(self) -> float:
        return 0.0

    def end_episode(self):
        """Update actor and critic at end of episode."""
        if not self.rewards:
            return

        # Compute discounted returns
        returns = []
        G = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            G = r + self.gamma * G * (1 - float(d))
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)

        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = (returns - values).pow(2).mean()
        entropy_loss = -entropies.mean()

        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.dones.clear()
        self.episodes_done += 1

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
