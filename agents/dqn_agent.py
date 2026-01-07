"""
Deep Q-Network (DQN) Agent implementation.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base_agent import BaseAgent


class DQN(nn.Module):
    """Deep Q-Network model."""

    def __init__(self, state_size, action_size, hidden_sizes=[128, 128]):
        """
        Initialize DQN network.

        Args:
            state_size: Dimension of input state
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQN, self).__init__()

        layers = []
        prev_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class DQNAgent(BaseAgent):
    """DQN Agent for reinforcement learning."""

    def __init__(self, state_size, action_size, config=None):
        """
        Initialize DQN agent.

        Args:
            state_size: Size of state space
            action_size: Size of action space
            config: Configuration dictionary
        """
        super().__init__(state_size, action_size, config)

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 64)
        self.memory_size = config.get("memory_size", 10000)
        self.target_update_freq = config.get("target_update_freq", 10)
        self.hidden_sizes = config.get("hidden_sizes", [128, 128])

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Networks
        self.policy_net = DQN(state_size, action_size, self.hidden_sizes).to(
            self.device
        )
        self.target_net = DQN(state_size, action_size, self.hidden_sizes).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.memory = deque(maxlen=self.memory_size)

        # Training counter
        self.train_step_counter = 0

        # Store config for later access
        self.config = config if config else {}

        # Epsilon parameters for compatibility with flappybird code
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_end", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

    def select_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate

        Returns:
            Selected action
        """
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(
        self, state=None, action=None, reward=None, next_state=None, done=None
    ):
        """
        Perform one training step using experience replay.

        Args:
            state, action, reward, next_state, done: Experience tuple (optional, can use memory)

        Returns:
            Loss value
        """
        # Store experience if provided
        if state is not None:
            self.remember(state, action, reward, next_state, done)

        # Don't train if not enough samples
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def train(self):
        """
        Train using experience replay (no parameters needed).
        This method is for compatibility with flappybird trainer.

        Returns:
            Loss value
        """
        # Don't train if not enough samples
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_counter += 1

        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        """Save model to file."""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_step_counter": self.train_step_counter,
            },
            filepath,
        )

    def load(self, filepath):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_step_counter = checkpoint.get("train_step_counter", 0)
