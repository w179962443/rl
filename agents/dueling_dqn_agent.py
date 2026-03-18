"""Double Dueling DQN Agent — separates V(s) and A(s,a) for better value estimation."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent


class DuelingNetwork(nn.Module):
    """Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A)."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.feature(x)
        val = self.value_stream(feat)
        adv = self.adv_stream(feat)
        return val + adv - adv.mean(dim=1, keepdim=True)


class DuelingDQNAgent(BaseAgent):
    """Double Dueling DQN with per-step epsilon decay."""

    def __init__(self, state_size: int, action_size: int, config: dict = None):
        super().__init__(state_size, action_size, config)
        c = self.config
        self.gamma = c.get("gamma", 0.99)
        self.lr = c.get("learning_rate", 3e-4)
        self.batch_size = c.get("batch_size", 64)
        self.buffer_size = c.get("buffer_size", 100_000)
        self.target_update = c.get("target_update", 500)
        self.hidden_size = c.get("hidden_size", 256)
        self.epsilon = c.get("epsilon_start", 1.0)
        self.epsilon_end = c.get("epsilon_end", 0.01)
        self.epsilon_decay = c.get("epsilon_decay", 0.9995)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.target_net = DuelingNetwork(state_size, action_size, self.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=self.buffer_size)
        self.steps_done = 0
        self.episodes_done = 0

    def select_action(self, state, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(s).argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.FloatTensor(np.array(states)).to(self.device)
        a = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(np.array(next_states)).to(self.device)
        d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.policy_net(s).gather(1, a)

        with torch.no_grad():
            next_actions = self.policy_net(ns).argmax(1, keepdim=True)
            next_q = self.target_net(ns).gather(1, next_actions)
            target_q = r + (1 - d) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Per-step epsilon decay and target update
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def end_episode(self):
        self.episodes_done += 1

    def save(self, filepath: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "episodes_done": self.episodes_done,
        }, filepath)

    def load(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self.steps_done = ckpt.get("steps_done", 0)
        self.episodes_done = ckpt.get("episodes_done", 0)
