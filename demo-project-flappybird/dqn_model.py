"""
Deep Q-Network (DQN) 强化学习模型
使用PyTorch实现的DQN算法，用于训练Flappy Bird AI
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple


# 经验回放的数据结构
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class DQN(nn.Module):
    """深度Q网络"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        初始化DQN网络

        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            hidden_size: 隐藏层大小
        """
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        """
        初始化回放缓冲区

        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """添加经验"""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样一批经验"""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """返回缓冲区大小"""
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 10,
        device: str = None,
    ):
        """
        初始化DQN智能体

        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            hidden_size: 隐藏层大小
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减
            buffer_size: 经验回放缓冲区大小
            batch_size: 训练批次大小
            target_update: 目标网络更新频率
            device: 计算设备
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # 设置计算设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # 创建策略网络和目标网络
        self.policy_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)

        # 训练计数器
        self.steps_done = 0
        self.episodes_done = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作（epsilon-greedy策略）

        Args:
            state: 当前状态
            training: 是否在训练模式

        Returns:
            选择的动作
        """
        if training and random.random() < self.epsilon:
            # 随机探索
            return random.randrange(self.action_size)
        else:
            # 利用学习到的策略
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, reward, next_state, done)

    def train(self) -> float:
        """
        训练网络

        Returns:
            损失值
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # 从回放缓冲区采样
        experiences = self.memory.sample(self.batch_size)

        # 解包经验
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = (
            torch.LongTensor([e.action for e in experiences])
            .unsqueeze(1)
            .to(self.device)
        )
        rewards = (
            torch.FloatTensor([e.reward for e in experiences])
            .unsqueeze(1)
            .to(self.device)
        )
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(
            self.device
        )
        dones = (
            torch.FloatTensor([e.done for e in experiences])
            .unsqueeze(1)
            .to(self.device)
        )

        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions)

        # 计算目标Q值（使用Double DQN）
        with torch.no_grad():
            # 使用策略网络选择动作
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            # 使用目标网络评估Q值
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
                "episodes_done": self.episodes_done,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        加载模型

        Args:
            filepath: 模型路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.steps_done = checkpoint.get("steps_done", 0)
        self.episodes_done = checkpoint.get("episodes_done", 0)
        print(f"Model loaded from {filepath}")

    def get_stats(self) -> dict:
        """获取智能体统计信息"""
        return {
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "episodes_done": self.episodes_done,
            "memory_size": len(self.memory),
        }


if __name__ == "__main__":
    # 测试DQN模型
    state_size = 7
    action_size = 2

    agent = DQNAgent(state_size, action_size)

    # 测试选择动作
    test_state = np.random.randn(state_size)
    action = agent.select_action(test_state)
    print(f"Selected action: {action}")

    # 测试存储和训练
    for i in range(100):
        state = np.random.randn(state_size)
        action = agent.select_action(state)
        reward = random.random()
        next_state = np.random.randn(state_size)
        done = random.random() > 0.9

        agent.store_experience(state, action, reward, next_state, done)

    # 训练
    loss = agent.train()
    print(f"Training loss: {loss}")

    # 获取统计信息
    stats = agent.get_stats()
    print(f"Agent stats: {stats}")
