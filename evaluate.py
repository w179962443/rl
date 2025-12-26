"""
Evaluation and comparison script for trained models.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import Plotter


def load_training_log(log_file):
    """Load training log from JSON file."""
    with open(log_file, "r") as f:
        data = json.load(f)
    return data


def analyze_training(log_file):
    """Analyze training results."""
    data = load_training_log(log_file)

    episodes = data["episodes"]
    rewards = [ep["reward"] for ep in episodes]

    print(f"\nAnalysis for: {data['experiment_name']}")
    print("=" * 60)
    print(f"Total episodes: {len(episodes)}")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")

    # Last 100 episodes
    if len(rewards) >= 100:
        last_100 = rewards[-100:]
        print(f"\nLast 100 episodes:")
        print(f"Average reward: {np.mean(last_100):.2f} ± {np.std(last_100):.2f}")

    # Check if loss data exists
    if "loss" in episodes[0]:
        losses = [ep["loss"] for ep in episodes if ep.get("loss", 0) > 0]
        if losses:
            print(f"\nTraining loss:")
            print(f"Average loss: {np.mean(losses):.4f}")
            print(f"Final loss: {np.mean(losses[-100:]):.4f}")

    print("=" * 60)


def compare_experiments(log_files, output_file="comparison.png"):
    """Compare multiple experiments."""
    plotter = Plotter()

    data_dict = {}
    for log_file in log_files:
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found, skipping...")
            continue

        data = load_training_log(log_file)
        rewards = [ep["reward"] for ep in data["episodes"]]
        data_dict[data["experiment_name"]] = rewards

    if data_dict:
        plotter.plot_comparison(
            data_dict,
            ylabel="Reward",
            title="Training Comparison",
            filename=output_file,
        )


def plot_all_results():
    """Plot results for all available experiments."""
    log_dir = "results/logs"

    if not os.path.exists(log_dir):
        print(f"No logs found in {log_dir}")
        return

    log_files = [
        os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".json")
    ]

    if not log_files:
        print("No training logs found!")
        return

    print(f"\nFound {len(log_files)} training logs\n")

    # Analyze each
    for log_file in log_files:
        analyze_training(log_file)

    # Compare if multiple
    if len(log_files) > 1:
        compare_experiments(log_files, "all_experiments_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate training results")
    parser.add_argument("--analyze", type=str, help="Path to log file to analyze")
    parser.add_argument("--compare", nargs="+", help="Paths to log files to compare")
    parser.add_argument("--all", action="store_true", help="Analyze all available logs")

    args = parser.parse_args()

    if args.all:
        plot_all_results()
    elif args.analyze:
        analyze_training(args.analyze)
    elif args.compare:
        compare_experiments(args.compare)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
