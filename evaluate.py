"""
Evaluate a trained model.

Usage:
    python evaluate.py --game flappybird --model outputs/flappybird_dqn/models/best.pth
    python evaluate.py --game dino --agent dueling_dqn --model outputs/dino_dueling_dqn/models/best.pth --episodes 20
    python evaluate.py --game cartpole --model outputs/cartpole_dqn/models/best.pth --render
"""

import argparse
import numpy as np

from config import GAMES
from train import make_env, make_agent, AGENT_MAP


def evaluate(args):
    game_cfg = GAMES[args.game]
    agent_name = args.agent or game_cfg["default_agent"]
    env_kwargs = game_cfg.get("env_kwargs", {})

    env = make_env(game_cfg["env"], render=args.render, **env_kwargs)
    agent = make_agent(agent_name, env.get_state_size(), env.get_action_size(), game_cfg["agent_config"])
    agent.load(args.model)

    print(f"Evaluating {args.game} with {agent_name}, model: {args.model}")
    print(f"Running {args.episodes} episodes...\n")

    scores = []
    for ep in range(1, args.episodes + 1):
        state = env.reset()
        total_steps = 0
        for _ in range(game_cfg["max_steps"]):
            action = agent.select_action(state, training=False)
            state, _, done, info = env.step(action)
            total_steps += 1
            if done:
                break
        score = info.get("score", 0)
        scores.append(score)
        print(f"  Episode {ep:>3d}  Score: {score:>8.1f}  Steps: {total_steps}")

    env.close()

    print(f"\n{'='*40}")
    print(f"  Episodes : {len(scores)}")
    print(f"  Avg Score: {np.mean(scores):.2f}")
    print(f"  Max Score: {np.max(scores):.1f}")
    print(f"  Min Score: {np.min(scores):.1f}")
    print(f"  Std      : {np.std(scores):.2f}")
    print(f"{'='*40}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--game", required=True, choices=list(GAMES.keys()))
    parser.add_argument("--agent", default=None, choices=list(AGENT_MAP.keys()))
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Render game during evaluation")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
