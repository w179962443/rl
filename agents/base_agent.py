"""Base Agent class for all reinforcement learning agents."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all RL agents."""

    def __init__(self, state_size: int, action_size: int, config: dict = None):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}

    @abstractmethod
    def select_action(self, state, training: bool = True) -> int:
        """Select an action given the current state."""
        pass

    @abstractmethod
    def store_experience(self, state, action, reward, next_state, done):
        """Store a transition for training."""
        pass

    @abstractmethod
    def train_step(self) -> float:
        """Perform one training step. Returns loss value."""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save the agent's model."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load the agent's model."""
        pass

    def end_episode(self):
        """Called at the end of each episode. Override for per-episode updates."""
        pass
