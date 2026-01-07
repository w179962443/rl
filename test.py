"""
Testing script for trained models.
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np

from agents import DQNAgent, QLearningAgent
from envs import SnakeEnv


def test_cartpole(args):
    """Test DQN on CartPole."""
    print("=" * 50)
    print("Testing DQN on CartPole")
    print("=" * 50)

    # Create environment
    if args.render:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent
    config = {
        "gamma": 0.99,
        "learning_rate": 0.001,
        "batch_size": 64,
        "memory_size": 10000,
        "target_update_freq": 10,
        "hidden_sizes": [128, 128],
    }
    agent = DQNAgent(state_size, action_size, config)

    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return

    agent.load(args.model)
    print(f"Loaded model from {args.model}")

    # Test
    total_rewards = []

    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, epsilon=0.0)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            state = next_state

            if args.render:
                time.sleep(0.01)

        total_rewards.append(episode_reward)
        print(
            f"Episode {episode + 1:3d} | Reward: {episode_reward:7.2f} | Steps: {steps:4d}"
        )

    env.close()

    print("\n" + "=" * 50)
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print("=" * 50)


def test_pong(args):
    """Test DQN on Pong."""
    print("=" * 50)
    print("Testing DQN on Pong")
    print("=" * 50)

    # Create environment
    if args.render:
        env = gym.make("ALE/Pong-v5", render_mode="human")
    else:
        env = gym.make("ALE/Pong-v5")

    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n

    # Create agent
    config = {
        "gamma": 0.99,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "memory_size": 50000,
        "target_update_freq": 100,
        "hidden_sizes": [512, 256],
    }
    agent = DQNAgent(state_size, action_size, config)

    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return

    agent.load(args.model)
    print(f"Loaded model from {args.model}")

    # Test
    total_rewards = []

    for episode in range(args.episodes):
        state, _ = env.reset()
        state = state.flatten()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, epsilon=0.0)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            state = next_state

            if args.render:
                time.sleep(0.01)

            # Limit episode length
            if steps >= 10000:
                break

        total_rewards.append(episode_reward)
        print(
            f"Episode {episode + 1:3d} | Reward: {episode_reward:7.2f} | Steps: {steps:5d}"
        )

    env.close()

    print("\n" + "=" * 50)
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print("=" * 50)


def test_frozenlake(args):
    """Test Q-Learning on FrozenLake."""
    print("=" * 50)
    print("Testing Q-Learning on FrozenLake")
    print("=" * 50)

    # Create environment
    if args.render:
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    else:
        env = gym.make("FrozenLake-v1", is_slippery=True)

    state_size = env.observation_space.n
    action_size = env.action_space.n

    # Create agent
    config = {
        "learning_rate": 0.1,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9995,
    }
    agent = QLearningAgent(state_size, action_size, config)

    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return

    agent.load(args.model)
    print(f"Loaded model from {args.model}")

    # Test
    total_rewards = []
    successes = 0

    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, epsilon=0.0)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            state = next_state

            if args.render:
                time.sleep(0.5)

            # Limit episode length
            if steps >= 100:
                break

        if episode_reward > 0:
            successes += 1

        total_rewards.append(episode_reward)
        result = "SUCCESS" if episode_reward > 0 else "FAIL"
        print(f"Episode {episode + 1:3d} | Result: {result:7s} | Steps: {steps:3d}")

    env.close()

    print("\n" + "=" * 50)
    print(
        f"Success Rate: {successes}/{args.episodes} ({100 * successes / args.episodes:.1f}%)"
    )
    print(f"Average Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
    print("=" * 50)


def test_snake(args):
    """Test DQN on Snake."""
    print("=" * 50)
    print("Testing DQN on Snake")
    print("=" * 50)

    # Create environment
    env = SnakeEnv(grid_size=10, render_mode="human" if args.render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent
    config = {
        "gamma": 0.99,
        "learning_rate": 0.001,
        "batch_size": 64,
        "memory_size": 10000,
        "target_update_freq": 10,
        "hidden_sizes": [256, 256],
    }
    agent = DQNAgent(state_size, action_size, config)

    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return

    agent.load(args.model)
    print(f"Loaded model from {args.model}")

    # Test
    total_rewards = []
    max_lengths = []

    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, epsilon=0.0)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            state = next_state

            if args.render:
                env.render()
                time.sleep(0.1)

            # Limit episode length
            if steps >= 500:
                break

        # Calculate snake length
        snake_length = len(env.snake)
        total_rewards.append(episode_reward)
        max_lengths.append(snake_length)

        print(
            f"Episode {episode + 1:3d} | Reward: {episode_reward:7.2f} | Length: {snake_length:2d} | Steps: {steps:3d}"
        )

    env.close()

    print("\n" + "=" * 50)
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Snake Length: {np.mean(max_lengths):.2f}")
    print(f"Max Snake Length: {np.max(max_lengths):.0f}")
    print("=" * 50)


def test_lunarlander(args):
    """Test DQN on LunarLander."""
    print("=" * 50)
    print("Testing DQN on LunarLander")
    print("=" * 50)

    # Create environment
    if args.render:
        env = gym.make("LunarLander-v2", render_mode="human")
    else:
        env = gym.make("LunarLander-v2")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create agent
    config = {
        "gamma": 0.99,
        "learning_rate": 0.0005,
        "batch_size": 64,
        "memory_size": 100000,
        "target_update_freq": 10,
        "hidden_sizes": [256, 256],
    }
    agent = DQNAgent(state_size, action_size, config)

    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return

    agent.load(args.model)
    print(f"Loaded model from {args.model}")

    # Test
    total_rewards = []
    successful_landings = 0

    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, epsilon=0.0)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            state = next_state

            if args.render:
                env.render()
                time.sleep(0.02)

        total_rewards.append(episode_reward)

        # Count successful landings (reward > 200 is considered solved)
        if episode_reward > 200:
            successful_landings += 1
            status = "✅ SUCCESS"
        else:
            status = "❌ FAILED"

        print(
            f"Episode {episode + 1:3d} | Reward: {episode_reward:7.2f} | Steps: {steps:3d} | {status}"
        )

    env.close()

    print("\n" + "=" * 50)
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(
        f"Success Rate: {successful_landings}/{args.episodes} ({100 * successful_landings / args.episodes:.1f}%)"
    )
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print("=" * 50)


def test_breakout(args):
    """Test DQN on Breakout."""
    print("=" * 50)
    print("Testing DQN on Breakout")
    print("=" * 50)

    # Create environment
    if args.render:
        env = gym.make("ALE/Breakout-v5", render_mode="human")
    else:
        env = gym.make("ALE/Breakout-v5")

    state_size = (
        env.observation_space.shape[0]
        * env.observation_space.shape[1]
        * env.observation_space.shape[2]
    )
    action_size = env.action_space.n

    # Create agent
    config = {
        "gamma": 0.99,
        "learning_rate": 0.00025,
        "batch_size": 32,
        "memory_size": 100000,
        "target_update_freq": 1000,
        "hidden_sizes": [512, 256],
    }
    agent = DQNAgent(state_size, action_size, config)

    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        return

    agent.load(args.model)
    print(f"Loaded model from {args.model}")

    # Test
    total_rewards = []
    total_scores = []

    for episode in range(args.episodes):
        state, _ = env.reset()
        state = state.flatten()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, epsilon=0.0)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            state = next_state

            if args.render:
                env.render()
                time.sleep(0.01)

        total_rewards.append(episode_reward)
        total_scores.append(episode_reward)

        print(
            f"Episode {episode + 1:3d} | Score: {episode_reward:7.2f} | Steps: {steps:5d}"
        )

    env.close()

    print("\n" + "=" * 50)
    print(f"Average Score: {np.mean(total_scores):.2f} ± {np.std(total_scores):.2f}")
    print(f"Max Score: {np.max(total_scores):.2f}")
    print(f"Min Score: {np.min(total_scores):.2f}")
    print("=" * 50)


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test trained RL agents")
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        choices=["cartpole", "pong", "frozenlake", "snake", "lunarlander", "breakout", "flappybird", "mario"],
        help="Game to test",
    )
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to test"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument(
        "--world", type=int, default=1, help="Mario world number (1-8)"
    )
    parser.add_argument(
        "--stage", type=int, default=1, help="Mario stage number (1-4)"
    )

    args = parser.parse_args()

    # Test on selected game
    if args.game == "cartpole":
        test_cartpole(args)
    elif args.game == "pong":
        test_pong(args)
    elif args.game == "frozenlake":
        test_frozenlake(args)
    elif args.game == "snake":
        test_snake(args)
    elif args.game == "lunarlander":
        test_lunarlander(args)
    elif args.game == "breakout":
        test_breakout(args)
    elif args.game == "flappybird":
        from experiments.flappybird.test import test_flappybird
        test_flappybird(args)
    elif args.game == "mario":
        from experiments.mario.test import test_mario
        test_mario(args)


if __name__ == "__main__":
    main()
