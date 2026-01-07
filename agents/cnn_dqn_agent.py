"""
CNN-DQN Agent for image-based environments like Super Mario Bros
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_agent import BaseAgent


class CNN_DQN(nn.Module):
    """卷积神经网络 DQN 模型（用于图像输入）"""

    def __init__(self, input_channels: int, action_size: int):
        """
        初始化 CNN-DQN 网络

        Args:
            input_channels: 输入通道数（通常是frame_stack数量）
            action_size: 动作空间大小
        """
        super(CNN_DQN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 计算卷积后的特征尺寸
        # 假设输入是 84x84
        conv_out_size = self._get_conv_output_size(input_channels)

        # 全连接层
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_size)

        # 初始化权重
        self._initialize_weights()

    def _get_conv_output_size(self, input_channels: int) -> int:
        """计算卷积层输出尺寸"""
        # 创建一个虚拟输入来计算输出尺寸
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return int(np.prod(x.size()))

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNDQNAgent(BaseAgent):
    """CNN-DQN Agent for image-based games"""

    def __init__(self, state_shape: tuple, action_size: int, config: dict = None):
        """
        初始化 CNN-DQN agent

        Args:
            state_shape: 状态形状 (channels, height, width)
            action_size: 动作空间大小
            config: 配置字典
        """
        # 注意：这里不调用super().__init__，因为state_size对于图像是一个tuple
        self.state_shape = state_shape
        self.action_size = action_size
        self.config = config if config else {}

        # 超参数
        self.gamma = config.get("gamma", 0.99)
        self.learning_rate = config.get("learning_rate", 0.00025)
        self.batch_size = config.get("batch_size", 32)
        self.memory_size = config.get("memory_size", 100000)
        self.target_update_freq = config.get("target_update_freq", 10000)

        # Epsilon 参数
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_end", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.9999)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 创建网络
        input_channels = state_shape[0]
        self.policy_net = CNN_DQN(input_channels, action_size).to(self.device)
        self.target_net = CNN_DQN(input_channels, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # 经验回放缓冲区
        self.memory = deque(maxlen=self.memory_size)

        # 训练计数器
        self.train_step_counter = 0
        self.episodes_done = 0

    def select_action(self, state: np.ndarray, epsilon: float = None) -> int:
        """
        使用 epsilon-greedy 策略选择动作

        Args:
            state: 当前状态
            epsilon: 探索率（如果为None，使用self.epsilon）

        Returns:
            选择的动作
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            # 确保状态形状正确
            if len(state.shape) == 3:
                state = np.expand_dims(state, 0)

            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def train(self) -> float:
        """
        使用经验回放训练网络

        Returns:
            损失值
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # 从内存中采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 下一状态Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_step_counter += 1

        # 定期更新目标网络
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str):
        """保存模型"""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step_counter": self.train_step_counter,
                "episodes_done": self.episodes_done,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.train_step_counter = checkpoint.get("train_step_counter", 0)
        self.episodes_done = checkpoint.get("episodes_done", 0)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # 测试 CNN-DQN
    print("Testing CNN-DQN Agent...")

    state_shape = (4, 84, 84)  # 4 stacked frames
    action_size = 7  # Mario SIMPLE actions

    config = {
        "gamma": 0.99,
        "learning_rate": 0.00025,
        "batch_size": 32,
        "memory_size": 10000,
        "target_update_freq": 1000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.9999,
    }

    agent = CNNDQNAgent(state_shape, action_size, config)

    # 测试选择动作
    test_state = np.random.randn(*state_shape).astype(np.float32)
    action = agent.select_action(test_state)
    print(f"Selected action: {action}")

    # 测试记忆和训练
    for i in range(100):
        state = np.random.randn(*state_shape).astype(np.float32)
        action = agent.select_action(state)
        reward = random.random()
        next_state = np.random.randn(*state_shape).astype(np.float32)
        done = random.random() > 0.9

        agent.remember(state, action, reward, next_state, done)

    # 训练
    loss = agent.train()
    print(f"Training loss: {loss:.4f}")
    print("Test complete!")
