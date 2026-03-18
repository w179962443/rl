"""REINFORCE (Monte-Carlo Policy Gradient) Agent."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .base_agent import BaseAgent


class PolicyNetwork(nn.Module):
    """Simple policy network that outputs action probabilities."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class REINFORCEAgent(BaseAgent):
    """REINFORCE with baseline (average return)."""

    def __init__(self, state_size: int, action_size: int, config: dict = None):
        super().__init__(state_size, action_size, config)
        c = self.config
        self.gamma = c.get("gamma", 0.99)
        self.lr = c.get("learning_rate", 1e-3)
        self.hidden_size = c.get("hidden_size", 128)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Episode buffer
        self.log_probs = []
        self.rewards = []
        self.baseline = 0.0
        self.baseline_alpha = 0.01
        self.episodes_done = 0

    def select_action(self, state, training: bool = True) -> int:
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(s)
        dist = Categorical(probs)
        action = dist.sample()
        if training:
            self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_experience(self, state, action, reward, next_state, done):
        self.rewards.append(reward)

    def train_step(self) -> float:
        # REINFORCE trains at end of episode, not per step
        return 0.0

    def end_episode(self):
        """Compute discounted returns and update policy."""
        if not self.rewards:
            return

        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # Update baseline
        ep_return = returns[0].item()
        self.baseline += self.baseline_alpha * (ep_return - self.baseline)

        # Normalize returns with baseline
        returns = returns - self.baseline
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss -= log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Clear episode buffer
        self.log_probs.clear()
        self.rewards.clear()
        self.episodes_done += 1

    def save(self, filepath: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "baseline": self.baseline,
            "episodes_done": self.episodes_done,
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.baseline = ckpt.get("baseline", 0.0)
        self.episodes_done = ckpt.get("episodes_done", 0)
