"""
Training script for all games.
"""

import argparse
import os
import pickle

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from agents import DQNAgent, QLearningAgent
from envs import SnakeEnv
from utils import Logger, Plotter


def save_training_plots(experiment_dir, rewards, losses, epsilons, episode):
    """
    Save training progress plots.
    
    Args:
        experiment_dir: Directory to save plots
        rewards: List of episode rewards
        losses: List of losses
        epsilons: List of epsilon values
        episode: Current episode number
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axes[0, 0].plot(rewards, alpha=0.3, label="Episode Reward")
    if len(rewards) >= 100:
        avg_rewards = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
        axes[0, 0].plot(avg_rewards, label="Avg (100 ep)", linewidth=2)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Training Rewards")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot losses
    if len(losses) > 0:
        axes[0, 1].plot(losses, alpha=0.6, label="Loss")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Training Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot epsilon
    if len(epsilons) > 0:
        axes[1, 0].plot(epsilons, label="Epsilon")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Epsilon")
        axes[1, 0].set_title("Exploration Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Stats
    stats_text = f"Episode: {episode}\n"
    stats_text += f"Avg Reward (last 100): {np.mean(rewards[-100:]):.2f}\n" if len(rewards) >= 100 else f"Avg Reward: {np.mean(rewards):.2f}\n"
    stats_text += f"Max Reward: {np.max(rewards):.2f}\n"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=14, 
                   verticalalignment="center", family="monospace")
    axes[1, 1].axis("off")
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(experiment_dir, "logs", f"training_episode_{episode}.png")
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    print(f"Training plot saved to {plot_path}")


def train_cartpole(args):
    """Train DQN on CartPole."""
    print("=" * 50)
    print("Training DQN on CartPole")
    print("=" * 50)

    # Create experiment directory
    experiment_dir = "experiments/cartpole"
    os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)

    # Create environment
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
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
    }
    agent = DQNAgent(state_size, action_size, config)

    # Training parameters
    epsilon = agent.epsilon
    render_every = getattr(args, 'render_every', 100)

    # Logger and plotter
    logger = Logger(log_dir=os.path.join(experiment_dir, "logs"), 
                   experiment_name=f"cartpole_{args.episodes}ep")
    plotter = Plotter(save_dir=os.path.join(experiment_dir, "logs"))

    # Training
    rewards_history = []
    losses_history = []
    epsilons_history = []
    best_reward = -float("inf")

    for episode in range(args.episodes):
        # Decide if we should render this episode
        should_render = (episode % render_every == 0)
        if should_render:
            if hasattr(env, 'render_mode'):
                env.close()
                env = gym.make("CartPole-v1", render_mode="human")
            else:
                env = gym.make("CartPole-v1", render_mode="human")
        elif episode > 0 and (episode - 1) % render_every == 0:
            # Close render mode after rendering episode
            env.close()
            env = gym.make("CartPole-v1")
        
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, epsilon)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Train
            loss = agent.train_step(state, action, reward, next_state, done)

            episode_reward += reward
            episode_loss += loss
            steps += 1
            state = next_state

        # Decay epsilon
        epsilon = max(agent.epsilon_min, epsilon * agent.epsilon_decay)
        agent.epsilon = epsilon

        # Log episode
        rewards_history.append(episode_reward)
        losses_history.append(episode_loss / steps if steps > 0 else 0)
        epsilons_history.append(epsilon)
        logger.log_episode(
            episode, episode_reward, steps, epsilon, episode_loss / steps
        )

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            logger.print_episode(
                episode, avg_reward, steps, epsilon, episode_loss / steps
            )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(experiment_dir, "models", "best_model.pth"))

        # Save checkpoint and plots
        if (episode + 1) % 100 == 0:
            agent.save(os.path.join(experiment_dir, "models", f"checkpoint_ep{episode + 1}.pth"))
            save_training_plots(experiment_dir, rewards_history, losses_history, 
                              epsilons_history, episode + 1)

    # Save final model and results
    agent.save(os.path.join(experiment_dir, "models", "final_model.pth"))
    logger.save()
    plotter.plot_training_progress(
        rewards_history, losses_history, filename="cartpole_training.png"
    )
    save_training_plots(experiment_dir, rewards_history, losses_history, 
                       epsilons_history, args.episodes)

    env.close()
    print(f"\nTraining completed! Best reward: {best_reward:.2f}")


def train_pong(args):
    """Train DQN on Pong."""
    print("=" * 50)
    print("Training DQN on Pong")
    print("=" * 50)

    # Create environment
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

    # Training parameters
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.9995
    epsilon = epsilon_start

    # Logger and plotter
    logger = Logger(experiment_name=f"pong_{args.episodes}ep")
    plotter = Plotter()

    # Training
    rewards_history = []
    losses_history = []
    best_reward = -float("inf")

    for episode in range(args.episodes):
        state, _ = env.reset()
        state = state.flatten()  # Flatten image
        episode_reward = 0
        episode_loss = 0
        steps = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, epsilon)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()
            done = terminated or truncated

            # Train
            loss = agent.train_step(state, action, reward, next_state, done)

            episode_reward += reward
            episode_loss += loss
            steps += 1
            state = next_state

            # Limit episode length
            if steps >= 10000:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Log episode
        rewards_history.append(episode_reward)
        losses_history.append(episode_loss / steps if steps > 0 else 0)
        logger.log_episode(
            episode, episode_reward, steps, epsilon, episode_loss / steps
        )

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            logger.print_episode(
                episode, avg_reward, steps, epsilon, episode_loss / steps
            )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs("models", exist_ok=True)
            agent.save("models/pong_best.pth")

        # Save checkpoint
        if (episode + 1) % 200 == 0:
            agent.save(f"models/pong_ep{episode + 1}.pth")

    # Save final model and results
    agent.save("models/pong_final.pth")
    logger.save()
    plotter.plot_training_progress(
        rewards_history, losses_history, filename="pong_training.png"
    )

    env.close()
    print(f"\nTraining completed! Best reward: {best_reward:.2f}")


def train_frozenlake(args):
    """Train Q-Learning on FrozenLake."""
    print("=" * 50)
    print("Training Q-Learning on FrozenLake")
    print("=" * 50)

    # Create environment
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

    # Logger and plotter
    logger = Logger(experiment_name=f"frozenlake_{args.episodes}ep")
    plotter = Plotter()

    # Training
    rewards_history = []
    best_avg_reward = -float("inf")

    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Train
            agent.train_step(state, action, reward, next_state, done)

            episode_reward += reward
            steps += 1
            state = next_state

            # Limit episode length
            if steps >= 100:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Log episode
        rewards_history.append(episode_reward)
        logger.log_episode(episode, episode_reward, steps, agent.epsilon)

        # Print progress
        if episode % 1000 == 0:
            avg_reward = (
                np.mean(rewards_history[-100:])
                if len(rewards_history) >= 100
                else np.mean(rewards_history)
            )
            logger.print_episode(episode, avg_reward, steps, agent.epsilon)

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                os.makedirs("models", exist_ok=True)
                agent.save("models/frozenlake_best.pkl")

        # Save checkpoint
        if (episode + 1) % 2000 == 0:
            agent.save(f"models/frozenlake_ep{episode + 1}.pkl")

    # Save final model and results
    agent.save("models/frozenlake_final.pkl")
    logger.save()
    plotter.plot_training_progress(
        rewards_history, window=100, filename="frozenlake_training.png"
    )

    env.close()
    print(f"\nTraining completed! Best average reward: {best_avg_reward:.2f}")


def train_snake(args):
    """Train DQN on Snake."""
    print("=" * 50)
    print("Training DQN on Snake")
    print("=" * 50)

    # Create environment
    env = SnakeEnv(grid_size=10)
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

    # Training parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99
    epsilon = epsilon_start

    # Logger and plotter
    logger = Logger(experiment_name=f"snake_{args.episodes}ep")
    plotter = Plotter()

    # Training
    rewards_history = []
    losses_history = []
    best_reward = -float("inf")

    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, epsilon)

            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Train
            loss = agent.train_step(state, action, reward, next_state, done)

            episode_reward += reward
            episode_loss += loss
            steps += 1
            state = next_state

            # Limit episode length
            if steps >= 500:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Log episode
        rewards_history.append(episode_reward)
        losses_history.append(episode_loss / steps if steps > 0 else 0)
        logger.log_episode(
            episode,
            episode_reward,
            steps,
            epsilon,
            episode_loss / steps if steps > 0 else 0,
        )

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            logger.print_episode(
                episode,
                avg_reward,
                steps,
                epsilon,
                episode_loss / steps if steps > 0 else 0,
            )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs("models", exist_ok=True)
            agent.save("models/snake_best.pth")

        # Save checkpoint
        if (episode + 1) % 100 == 0:
            agent.save(f"models/snake_ep{episode + 1}.pth")

    # Save final model and results
    agent.save("models/snake_final.pth")
    logger.save()
    plotter.plot_training_progress(
        rewards_history, losses_history, filename="snake_training.png"
    )

    env.close()
    print(f"\nTraining completed! Best reward: {best_reward:.2f}")


def train_lunarlander(args):
    """Train DQN on LunarLander."""
    print("=" * 50)
    print("Training DQN on LunarLander")
    print("=" * 50)

    # Create environment
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

    # Initialize tracking
    logger = Logger("lunarlander_training")
    plotter = Plotter("LunarLander Training")
    rewards_history = []
    losses_history = []
    best_avg_reward = -float("inf")

    # Training loop
    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        steps = 0
        done = False

        while not done:
            # Select and take action
            action = agent.select_action(state, agent.epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward
            steps += 1
            state = next_state

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Update target network
        if episode % config["target_update_freq"] == 0:
            agent.update_target_network()

        # Record metrics
        rewards_history.append(episode_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_history.append(avg_loss)

        # Log progress
        if episode % 10 == 0:
            avg_reward = (
                np.mean(rewards_history[-100:])
                if len(rewards_history) >= 100
                else np.mean(rewards_history)
            )
            logger.print_episode(episode, avg_reward, steps, agent.epsilon)

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                os.makedirs("models", exist_ok=True)
                agent.save("models/lunarlander_best.pth")

        # Save checkpoint
        if (episode + 1) % 100 == 0:
            agent.save(f"models/lunarlander_ep{episode + 1}.pth")

    # Save final model and results
    agent.save("models/lunarlander_final.pth")
    logger.save()
    plotter.plot_training_progress(
        rewards_history, losses_history, filename="lunarlander_training.png"
    )

    env.close()
    print(f"\nTraining completed! Best average reward: {best_avg_reward:.2f}")


def train_breakout(args):
    """Train DQN on Breakout."""
    print("=" * 50)
    print("Training DQN on Breakout")
    print("=" * 50)

    # Create environment
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

    # Initialize tracking
    logger = Logger("breakout_training")
    plotter = Plotter("Breakout Training")
    rewards_history = []
    losses_history = []
    best_avg_reward = -float("inf")

    # Training loop
    for episode in range(args.episodes):
        state, _ = env.reset()
        # Flatten state
        state = state.flatten()
        episode_reward = 0
        episode_losses = []
        steps = 0
        done = False

        while not done:
            # Select and take action
            action = agent.select_action(state, agent.epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.flatten()
            done = terminated or truncated

            # Store transition
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            loss = agent.train()
            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward
            steps += 1
            state = next_state

        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Update target network
        if episode % config["target_update_freq"] == 0:
            agent.update_target_network()

        # Record metrics
        rewards_history.append(episode_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses_history.append(avg_loss)

        # Log progress
        if episode % 10 == 0:
            avg_reward = (
                np.mean(rewards_history[-100:])
                if len(rewards_history) >= 100
                else np.mean(rewards_history)
            )
            logger.print_episode(episode, avg_reward, steps, agent.epsilon)

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                os.makedirs("models", exist_ok=True)
                agent.save("models/breakout_best.pth")

        # Save checkpoint
        if (episode + 1) % 500 == 0:
            agent.save(f"models/breakout_ep{episode + 1}.pth")

    # Save final model and results
    agent.save("models/breakout_final.pth")
    logger.save()
    plotter.plot_training_progress(
        rewards_history, losses_history, filename="breakout_training.png"
    )

    env.close()
    print(f"\nTraining completed! Best average reward: {best_avg_reward:.2f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agents on various games")
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        choices=["cartpole", "pong", "frozenlake", "snake", "lunarlander", "breakout", "flappybird", "mario"],
        help="Game to train on",
    )
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of episodes to train"
    )
    parser.add_argument(
        "--render-every", type=int, default=100, help="Render every N episodes"
    )
    parser.add_argument(
        "--world", type=int, default=1, help="Mario world number (1-8)"
    )
    parser.add_argument(
        "--stage", type=int, default=1, help="Mario stage number (1-4)"
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Train on selected game
    if args.game == "cartpole":
        train_cartpole(args)
    elif args.game == "pong":
        train_pong(args)
    elif args.game == "frozenlake":
        train_frozenlake(args)
    elif args.game == "snake":
        train_snake(args)
    elif args.game == "lunarlander":
        train_lunarlander(args)
    elif args.game == "breakout":
        train_breakout(args)
    elif args.game == "flappybird":
        from experiments.flappybird.train import train_flappybird
        train_flappybird(args)
    elif args.game == "mario":
        from experiments.mario.train import train_mario
        train_mario(args)


if __name__ == "__main__":
    main()
