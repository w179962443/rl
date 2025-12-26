"""
Utility functions for logging training progress.
"""

import os
import json
from datetime import datetime


class Logger:
    """Logger for training progress."""

    def __init__(self, log_dir="results/logs", experiment_name=None):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.json")

        self.data = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "episodes": [],
            "metrics": {},
        }

    def log_episode(self, episode, reward, steps, epsilon=None, loss=None):
        """
        Log episode data.

        Args:
            episode: Episode number
            reward: Total reward
            steps: Number of steps
            epsilon: Exploration rate
            loss: Training loss
        """
        episode_data = {
            "episode": episode,
            "reward": reward,
            "steps": steps,
        }

        if epsilon is not None:
            episode_data["epsilon"] = epsilon
        if loss is not None:
            episode_data["loss"] = loss

        self.data["episodes"].append(episode_data)

    def log_metric(self, name, value):
        """Log a metric."""
        self.data["metrics"][name] = value

    def save(self):
        """Save log to file."""
        self.data["end_time"] = datetime.now().isoformat()
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def print_episode(self, episode, reward, steps, epsilon=None, loss=None):
        """Print episode information."""
        msg = f"Episode {episode:4d} | Reward: {reward:7.2f} | Steps: {steps:4d}"

        if epsilon is not None:
            msg += f" | Epsilon: {epsilon:.3f}"
        if loss is not None:
            msg += f" | Loss: {loss:.4f}"

        print(msg)
