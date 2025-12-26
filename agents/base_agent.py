"""
Base Agent class for all reinforcement learning agents.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Base class for all RL agents."""

    def __init__(self, state_size, action_size, config=None):
        """
        Initialize the agent.

        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            config: Configuration dictionary
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}

    @abstractmethod
    def select_action(self, state, epsilon=0.0):
        """
        Select an action given the current state.

        Args:
            state: Current state
            epsilon: Exploration rate for epsilon-greedy policy

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def train_step(self, state, action, reward, next_state, done):
        """
        Perform one training step.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done

        Returns:
            Loss value (if applicable)
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the agent's model."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Load the agent's model."""
        pass
