"""
Quick start script for running experiments.
"""

import os
import subprocess
import argparse


def run_experiment(game, episodes):
    """Run a training experiment."""
    print(f"\n{'='*60}")
    print(f"Starting {game.upper()} experiment")
    print(f"{'='*60}\n")

    # Run training
    cmd = f"python train.py --game {game} --episodes {episodes}"
    print(f"Running: {cmd}\n")
    subprocess.run(cmd, shell=True)

    print(f"\n{'='*60}")
    print(f"{game.upper()} experiment completed!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Quick start experiments")
    parser.add_argument(
        "--game",
        type=str,
        choices=["cartpole", "pong", "frozenlake", "all"],
        default="all",
        help="Which game to train",
    )

    args = parser.parse_args()

    # Default episode counts
    episodes = {"cartpole": 500, "pong": 2000, "frozenlake": 10000}

    if args.game == "all":
        print("\n" + "=" * 60)
        print("Running all experiments")
        print("This will take several hours!")
        print("=" * 60)

        for game in ["cartpole", "frozenlake", "pong"]:
            run_experiment(game, episodes[game])
    else:
        run_experiment(args.game, episodes[args.game])


if __name__ == "__main__":
    main()
