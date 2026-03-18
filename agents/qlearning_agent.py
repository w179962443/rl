"""Tabular Q-Learning Agent for discrete state spaces."""

import pickle

import numpy as np

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Q-Learning agent with epsilon-greedy exploration."""

    def __init__(self, state_size: int, action_size: int, config: dict = None):
        super().__init__(state_size, action_size, config)
        c = self.config
        self.lr = c.get("learning_rate", 0.1)
        self.gamma = c.get("gamma", 0.99)
        self.epsilon = c.get("epsilon_start", 1.0)
        self.epsilon_end = c.get("epsilon_end", 0.01)
        self.epsilon_decay = c.get("epsilon_decay", 0.9995)

        self.q_table = np.zeros((state_size, action_size))
        self._last_transition = None
        self.episodes_done = 0

    def select_action(self, state, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def store_experience(self, state, action, reward, next_state, done):
        self._last_transition = (state, action, reward, next_state, done)

    def train_step(self) -> float:
        if self._last_transition is None:
            return 0.0
        state, action, reward, next_state, done = self._last_transition
        td_target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        self._last_transition = None
        return abs(td_error)

    def end_episode(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "episodes_done": self.episodes_done,
            }, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_end)
        self.episodes_done = data.get("episodes_done", 0)
