"""Training logger — saves metrics to JSON and plots training curves."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingLogger:
    """Tracks and persists training metrics."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.scores = []
        self.losses = []
        self.epsilons = []
        self.eval_scores = []  # (episode, avg_score) pairs
        self.best_score = -float("inf")

    def log_episode(self, score: float, loss: float = 0.0, epsilon: float = 0.0):
        self.scores.append(score)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        if score > self.best_score:
            self.best_score = score

    def log_eval(self, episode: int, avg_score: float):
        self.eval_scores.append((episode, avg_score))

    def save(self):
        data = {
            "scores": self.scores,
            "losses": self.losses,
            "epsilons": self.epsilons,
            "eval_scores": self.eval_scores,
            "best_score": self.best_score,
        }
        with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
            json.dump(data, f)

    def load(self):
        path = os.path.join(self.log_dir, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.scores = data.get("scores", [])
            self.losses = data.get("losses", [])
            self.epsilons = data.get("epsilons", [])
            self.eval_scores = data.get("eval_scores", [])
            self.best_score = data.get("best_score", -float("inf"))
            return True
        return False

    def plot(self, filename: str = "training_curve.png"):
        if len(self.scores) < 2:
            return
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Scores
        axes[0, 0].plot(self.scores, alpha=0.3, label="Score")
        if len(self.scores) >= 100:
            ma = np.convolve(self.scores, np.ones(100) / 100, mode="valid")
            axes[0, 0].plot(range(99, len(self.scores)), ma, label="MA100", linewidth=2)
        if self.eval_scores:
            eps, avgs = zip(*self.eval_scores)
            axes[0, 0].scatter(eps, avgs, c="red", s=20, zorder=5, label="Eval")
        axes[0, 0].set_title("Score")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        if any(l > 0 for l in self.losses):
            axes[0, 1].plot(self.losses, alpha=0.5)
            axes[0, 1].set_title("Loss")
            axes[0, 1].grid(True, alpha=0.3)

        # Epsilon
        if self.epsilons:
            axes[1, 0].plot(self.epsilons)
            axes[1, 0].set_title("Epsilon")
            axes[1, 0].grid(True, alpha=0.3)

        # Stats text
        stats = (
            f"Episodes: {len(self.scores)}\n"
            f"Best Score: {self.best_score:.1f}\n"
            f"Avg (last 100): {np.mean(self.scores[-100:]):.2f}\n"
        )
        if self.eval_scores:
            stats += f"Last Eval: {self.eval_scores[-1][1]:.2f}\n"
        axes[1, 1].text(0.1, 0.5, stats, fontsize=13, va="center", family="monospace")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, filename), dpi=100)
        plt.close(fig)
