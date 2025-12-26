"""
Script to visualize FrozenLake Q-table and policy.
"""
import argparse
import os
from agents import QLearningAgent
from utils.visualize_qtable import visualize_q_table, visualize_policy, visualize_value_function


def main():
    parser = argparse.ArgumentParser(description='Visualize FrozenLake Q-table')
    parser.add_argument('--model', type=str, default='models/frozenlake_best.pkl',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        print("Please train FrozenLake first:")
        print("  python train.py --game frozenlake --episodes 10000")
        return
    
    # Create agent and load model
    print(f"Loading model from {args.model}...")
    agent = QLearningAgent(16, 4)  # FrozenLake has 16 states and 4 actions
    agent.load(args.model)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_q_table(agent, 'results/plots/frozenlake_qtable.png')
    visualize_policy(agent, 'results/plots/frozenlake_policy.png')
    visualize_value_function(agent, 'results/plots/frozenlake_values.png')
    
    print("\n✓ All visualizations created successfully!")
    print("\nGenerated files:")
    print("  - results/plots/frozenlake_qtable.png")
    print("  - results/plots/frozenlake_policy.png")
    print("  - results/plots/frozenlake_values.png")


if __name__ == '__main__':
    main()
