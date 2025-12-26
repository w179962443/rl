"""
Demo script to show basic usage of the package.
"""
import gymnasium as gym
from agents import DQNAgent
import numpy as np


def demo_cartpole():
    """Simple demo of training DQN on CartPole for a few episodes."""
    print("\n" + "="*60)
    print("CartPole Demo - Training for 50 episodes")
    print("="*60 + "\n")
    
    # Create environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent with simple config
    config = {
        'gamma': 0.99,
        'learning_rate': 0.001,
        'batch_size': 32,
        'memory_size': 1000,
        'target_update_freq': 5,
    }
    agent = DQNAgent(state_size, action_size, config)
    
    # Train for a few episodes
    epsilon = 1.0
    for episode in range(50):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.train_step(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
        
        epsilon = max(0.01, epsilon * 0.99)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.0f}, Epsilon = {epsilon:.3f}")
    
    env.close()
    print("\nDemo completed! This was just a quick demonstration.")
    print("For full training, use: python train.py --game cartpole --episodes 500")


if __name__ == '__main__':
    demo_cartpole()
