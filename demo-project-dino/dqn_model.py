"""
DQN Agent for Chrome Dinosaur Game
双网络 DQN（Dueling DQN + Double DQN + PER 可选）
"""

import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ════════════════════════════════════════════════
#  网络结构：Dueling DQN
# ════════════════════════════════════════════════


class DuelingDQN(nn.Module):
    """
    Dueling DQN 网络
    将 Q 值分解为 V(s) + A(s,a) — 对恐龙游戏这种稀疏动作场景效果更好
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        val = self.value_stream(feat)  # (B, 1)
        adv = self.adv_stream(feat)  # (B, A)
        # Q = V + (A - mean(A))
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


# ════════════════════════════════════════════════
#  经验回放
# ════════════════════════════════════════════════


class ReplayBuffer:
    """固定大小的经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ════════════════════════════════════════════════
#  DQN Agent
# ════════════════════════════════════════════════


class DQNAgent:
    """
    Double Dueling DQN Agent

    Parameters
    ----------
    state_size      : 状态维度
    action_size     : 动作维度（0=不动, 1=跳, 2=俯身）
    hidden_size     : 隐藏层宽度
    learning_rate   : 学习率
    gamma           : 折扣因子
    epsilon_start   : ε-贪心初始值
    epsilon_end     : ε-贪心最小值
    epsilon_decay   : ε 乘法衰减系数（每步）
    buffer_size     : 回放缓冲区大小
    batch_size      : 每次训练批大小
    target_update   : 目标网络更新频率（step 次数）
    """

    def __init__(
        self,
        state_size: int = 14,
        action_size: int = 3,
        hidden_size: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update: int = 500,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self._step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Device: {self.device}")

        self.policy_net = DuelingDQN(state_size, action_size, hidden_size).to(
            self.device
        )
        self.target_net = DuelingDQN(state_size, action_size, hidden_size).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.buffer = ReplayBuffer(buffer_size)

    # ─── 与环境交互 ──────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        """ε-贪心选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy_net(s).argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    # ─── 训练 ───────────────────────────────────
    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        # Double DQN: 用 policy_net 选动作，用 target_net 估值
        with torch.no_grad():
            best_actions = self.policy_net(ns).argmax(dim=1, keepdim=True)
            target_q = self.target_net(ns).gather(1, best_actions).squeeze(1)
            target = r + self.gamma * target_q * (1 - d)

        current_q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        loss = F.smooth_l1_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # 更新 ε
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 更新目标网络
        self._step_count += 1
        if self._step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ─── 保存 / 加载 ─────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self._step_count,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self._step_count = checkpoint.get("step_count", 0)
        print(
            f"[DQNAgent] Loaded from {path}  (ε={self.epsilon:.4f}, step={self._step_count})"
        )
