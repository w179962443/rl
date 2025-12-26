"""
Q-Learning Agent implementation for discrete state spaces.
"""
import numpy as np
import pickle
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Q-Learning agent for tabular environments."""
    
    def __init__(self, state_size, action_size, config=None):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_size: Size of state space (number of states)
            action_size: Size of action space
            config: Configuration dictionary
        """
        super().__init__(state_size, action_size, config)
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        
        # Q-table
        self.q_table = np.zeros((state_size, action_size))
        
        # Current epsilon
        self.epsilon = self.epsilon_start
        
    def select_action(self, state, epsilon=None):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (integer)
            epsilon: Exploration rate (if None, use internal epsilon)
            
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])
    
    def train_step(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            TD error (absolute difference)
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # TD error
        td_error = td_target - current_q
        
        # Update Q-value
        self.q_table[state, action] += self.learning_rate * td_error
        
        return abs(td_error)
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'state_size': self.state_size,
                'action_size': self.action_size,
            }, f)
    
    def load(self, filepath):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.epsilon = data['epsilon']
            self.state_size = data['state_size']
            self.action_size = data['action_size']
