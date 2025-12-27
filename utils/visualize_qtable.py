"""
Visualization tools for Q-learning agents.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def visualize_q_table(agent, save_path="results/plots/q_table.png"):
    """
    Visualize Q-table for FrozenLake environment.

    Args:
        agent: QLearningAgent instance
        save_path: Path to save the visualization
    """
    # Get Q-table
    q_table = agent.q_table

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Action names
    actions = ["Left", "Down", "Right", "Up"]

    # Plot Q-values for each action
    for idx, (ax, action_name) in enumerate(zip(axes.flat, actions)):
        # Reshape to 4x4 grid
        q_values = q_table[:, idx].reshape(4, 4)

        # Create heatmap
        im = ax.imshow(q_values, cmap="RdYlGn", aspect="auto")
        ax.set_title(
            f"Q-values for Action: {action_name}", fontsize=14, fontweight="bold"
        )

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = ax.text(
                    j,
                    i,
                    f"{q_values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        # Set ticks
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(np.arange(4))
        ax.set_yticklabels(np.arange(4))

    plt.suptitle("FrozenLake Q-Table Visualization", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Q-table visualization saved to {save_path}")


def visualize_policy(agent, save_path="results/plots/policy.png"):
    """
    Visualize the learned policy for FrozenLake.

    Args:
        agent: QLearningAgent instance
        save_path: Path to save the visualization
    """
    # Get best action for each state
    policy = np.argmax(agent.q_table, axis=1).reshape(4, 4)

    # Action symbols
    action_symbols = {
        0: "←",  # Left
        1: "↓",  # Down
        2: "→",  # Right
        3: "↑",  # Up
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create grid
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")

    # FrozenLake map
    lake_map = [
        ["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"],
    ]

    # Color map
    colors = {"S": "lightblue", "F": "white", "H": "lightcoral", "G": "lightgreen"}

    # Draw cells
    for i in range(4):
        for j in range(4):
            cell_type = lake_map[i][j]
            color = colors[cell_type]

            # Draw rectangle
            rect = plt.Rectangle(
                (j - 0.5, 3 - i - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(rect)

            # Add cell label
            ax.text(
                j,
                3 - i + 0.3,
                cell_type,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

            # Add policy arrow (except for holes and goal)
            if cell_type not in ["H", "G"]:
                action = policy[i, j]
                symbol = action_symbols[action]
                ax.text(
                    j,
                    3 - i - 0.1,
                    symbol,
                    ha="center",
                    va="center",
                    fontsize=24,
                    color="blue",
                    fontweight="bold",
                )

    # Set ticks
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(range(4))
    ax.set_yticklabels(range(3, -1, -1))

    # Labels
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    ax.set_title(
        "Learned Policy for FrozenLake\n(Arrows show best action)",
        fontsize=14,
        fontweight="bold",
    )

    # Grid
    ax.grid(True, linewidth=0.5)

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Policy visualization saved to {save_path}")


def visualize_value_function(agent, save_path="results/plots/value_function.png"):
    """
    Visualize the state value function.

    Args:
        agent: QLearningAgent instance
        save_path: Path to save the visualization
    """
    # Compute state values (max Q-value for each state)
    state_values = np.max(agent.q_table, axis=1).reshape(4, 4)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(state_values, cmap="RdYlGn", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("State Value", fontsize=12)

    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax.text(
                j,
                i,
                f"{state_values[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )

    # Set ticks
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(np.arange(4))
    ax.set_yticklabels(np.arange(4))

    # Labels
    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row", fontsize=12)
    ax.set_title(
        "State Value Function (V(s) = max Q(s,a))", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Value function visualization saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("This module provides visualization tools for Q-learning agents.")
    print("\nUsage:")
    print("  from utils.visualize_qtable import visualize_q_table, visualize_policy")
    print("  agent = QLearningAgent(...)")
    print("  agent.load('models/frozenlake_best.pkl')")
    print("  visualize_q_table(agent)")
    print("  visualize_policy(agent)")
