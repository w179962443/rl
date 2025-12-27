"""
Plotting utilities for visualizing training results.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """Plotter for training results."""

    def __init__(self, save_dir="results/plots"):
        """
        Initialize plotter.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_progress(
        self, rewards, losses=None, window=100, filename="training_progress.png"
    ):
        """
        Plot training progress.

        Args:
            rewards: List of episode rewards
            losses: List of losses (optional)
            window: Window size for moving average
            filename: Filename to save plot
        """
        fig, axes = plt.subplots(
            1, 2 if losses else 1, figsize=(15 if losses else 10, 5)
        )

        if not losses:
            axes = [axes]

        # Plot rewards
        axes[0].plot(rewards, alpha=0.3, label="Episode Reward")

        # Moving average
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            axes[0].plot(
                range(window - 1, len(rewards)),
                moving_avg,
                label=f"{window}-Episode Moving Avg",
                linewidth=2,
            )

        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Training Rewards")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot losses
        if losses:
            # Filter out zero losses
            non_zero_losses = [(i, loss) for i, loss in enumerate(losses) if loss > 0]
            if non_zero_losses:
                indices, loss_values = zip(*non_zero_losses)
                axes[1].plot(indices, loss_values, alpha=0.5, label="Loss")

                # Moving average for losses
                if len(loss_values) >= window:
                    loss_moving_avg = np.convolve(
                        loss_values, np.ones(window) / window, mode="valid"
                    )
                    axes[1].plot(
                        range(indices[window - 1], indices[-1] + 1),
                        loss_moving_avg,
                        label=f"{window}-Step Moving Avg",
                        linewidth=2,
                    )

                axes[1].set_xlabel("Training Step")
                axes[1].set_ylabel("Loss")
                axes[1].set_title("Training Loss")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Plot saved to {save_path}")

    def plot_comparison(
        self,
        data_dict,
        ylabel="Reward",
        title="Comparison",
        filename="comparison.png",
        window=100,
    ):
        """
        Plot comparison of multiple runs.

        Args:
            data_dict: Dictionary of {label: data_list}
            ylabel: Y-axis label
            title: Plot title
            filename: Filename to save plot
            window: Window size for moving average
        """
        plt.figure(figsize=(12, 6))

        for label, data in data_dict.items():
            plt.plot(data, alpha=0.3)

            if len(data) >= window:
                moving_avg = np.convolve(data, np.ones(window) / window, mode="valid")
                plt.plot(
                    range(window - 1, len(data)), moving_avg, label=label, linewidth=2
                )

        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Comparison plot saved to {save_path}")
