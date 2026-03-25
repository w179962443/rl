"""
Unified training script for all games and agents.

Usage:
    python train.py --game flappybird
    python train.py --game dino --agent dueling_dqn
    python train.py --game cartpole --agent a2c
    python train.py --game flappybird --agent reinforce --episodes 5000
    python train.py --game flappybird --resume outputs/flappybird_dqn/models/best.pth
"""

import os
import time
import argparse
import numpy as np

from config import GAMES
from utils import TrainingLogger


# ── Environment factory ──────────────────────────────────────────────────────

def make_env(env_name: str, render: bool = False, **kwargs):
    """Create an environment by name."""
    if env_name == "flappybird":
        from envs import FlappyBirdEnv
        return FlappyBirdEnv(render=render)
    elif env_name == "dino":
        from envs import DinoEnv
        return DinoEnv(render=render)
    elif env_name == "snake":
        from envs import SnakeEnv
        return SnakeEnv(render=render)
    else:
        from envs import GymEnv
        return GymEnv(env_name, render=render, **kwargs)


# ── Agent factory ────────────────────────────────────────────────────────────

AGENT_MAP = {
    "dqn": "agents.DQNAgent",
    "dueling_dqn": "agents.DuelingDQNAgent",
    "reinforce": "agents.REINFORCEAgent",
    "a2c": "agents.A2CAgent",
    "qlearning": "agents.QLearningAgent",
    "ppo": "agents.PPOAgent",
}


def make_agent(agent_name: str, state_size: int, action_size: int, config: dict):
    """Create an agent by name."""
    import agents
    cls_map = {
        "dqn": agents.DQNAgent,
        "dueling_dqn": agents.DuelingDQNAgent,
        "reinforce": agents.REINFORCEAgent,
        "a2c": agents.A2CAgent,
        "qlearning": agents.QLearningAgent,
        "ppo": agents.PPOAgent,
    }
    if agent_name not in cls_map:
        raise ValueError(f"Unknown agent: {agent_name}. Choose from {list(cls_map)}")
    return cls_map[agent_name](state_size, action_size, config)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(env_name, agent, num_episodes: int = 10, max_steps: int = 5000, **env_kwargs):
    """Run evaluation episodes and return average score."""
    env = make_env(env_name, render=False, **env_kwargs)
    scores = []
    for _ in range(num_episodes):
        state = env.reset()
        for _ in range(max_steps):
            action = agent.select_action(state, training=False)
            state, _, done, info = env.step(action)
            if done:
                break
        scores.append(info.get("score", 0))
    env.close()
    return float(np.mean(scores)), scores


# ── Training loop ────────────────────────────────────────────────────────────

def train(args):
    game_cfg = GAMES[args.game]
    agent_name = args.agent or game_cfg["default_agent"]
    episodes = args.episodes or game_cfg["episodes"]
    max_steps = game_cfg["max_steps"]
    eval_every = args.eval_every or game_cfg["eval_every"]
    eval_episodes = game_cfg.get("eval_episodes", 10)
    save_every = game_cfg.get("save_every", 200)
    log_every = game_cfg.get("log_every", 50)
    warmup = game_cfg.get("warmup_steps", 0)
    env_kwargs = game_cfg.get("env_kwargs", {})

    # Output directory
    out_dir = os.path.join("outputs", f"{args.game}_{agent_name}")
    model_dir = os.path.join(out_dir, "models")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create env and agent
    env = make_env(game_cfg["env"], render=False, **env_kwargs)
    agent = make_agent(agent_name, env.get_state_size(), env.get_action_size(), game_cfg["agent_config"])

    # Resume
    logger = TrainingLogger(log_dir)
    start_ep = 0
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        if logger.load():
            start_ep = len(logger.scores)
        print(f"Resumed from {args.resume}, starting at episode {start_ep}")

    print("=" * 60)
    print(f"  Game: {args.game}  |  Agent: {agent_name}  |  Episodes: {episodes}")
    print(f"  Eval every {eval_every} episodes ({eval_episodes} eval eps)")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    global_step = 0
    start_time = time.time()

    for ep in range(start_ep, start_ep + episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_loss = []

        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)

            if global_step >= warmup:
                loss = agent.train_step()
                if loss > 0:
                    ep_loss.append(loss)

            state = next_state
            ep_reward += reward
            global_step += 1
            if done:
                break

        agent.end_episode()
        score = info.get("score", ep_reward)
        avg_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
        epsilon = getattr(agent, "epsilon", 0.0)
        logger.log_episode(score, avg_loss, epsilon)

        # Save best
        if score >= logger.best_score:
            agent.save(os.path.join(model_dir, "best.pth"))

        # Periodic logging
        if (ep + 1) % log_every == 0:
            avg100 = np.mean(logger.scores[-100:]) if len(logger.scores) >= 100 else np.mean(logger.scores)
            elapsed = time.time() - start_time
            print(
                f"  Ep {ep+1:>5d}/{start_ep+episodes}"
                f"  Score={score:>7.1f}"
                f"  Avg100={avg100:>7.1f}"
                f"  Best={logger.best_score:>7.1f}"
                f"  eps={epsilon:.4f}"
                f"  Loss={avg_loss:.4f}"
                f"  {elapsed/60:.1f}min"
            )

        # Periodic evaluation
        if (ep + 1) % eval_every == 0:
            avg_eval, eval_scores = evaluate(
                game_cfg["env"], agent, eval_episodes, max_steps, **env_kwargs
            )
            logger.log_eval(ep + 1, avg_eval)
            print(f"  >>> EVAL ep {ep+1}: avg={avg_eval:.2f}  scores={eval_scores}")

        # Periodic save
        if (ep + 1) % save_every == 0:
            agent.save(os.path.join(model_dir, f"checkpoint_ep{ep+1}.pth"))
            logger.save()
            logger.plot()

    # Final save
    agent.save(os.path.join(model_dir, "final.pth"))
    logger.save()
    logger.plot(filename="training_final.png")
    env.close()

    print(f"\nDone. Best score: {logger.best_score:.1f}")
    print(f"Models saved to {model_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RL agents on games")
    parser.add_argument("--game", required=True, choices=list(GAMES.keys()), help="Game to train on")
    parser.add_argument("--agent", default=None, choices=list(AGENT_MAP.keys()), help="RL algorithm (default: per-game)")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count")
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluate every N episodes")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
