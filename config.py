"""
Configuration file for hyperparameters.
"""

# CartPole configuration
CARTPOLE_CONFIG = {
    'agent': 'DQN',
    'env_name': 'CartPole-v1',
    'episodes': 500,
    'hyperparameters': {
        'gamma': 0.99,
        'learning_rate': 0.001,
        'batch_size': 64,
        'memory_size': 10000,
        'target_update_freq': 10,
        'hidden_sizes': [128, 128],
    },
    'exploration': {
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
    },
    'success_threshold': 195,
}

# Pong configuration
PONG_CONFIG = {
    'agent': 'DQN',
    'env_name': 'ALE/Pong-v5',
    'episodes': 2000,
    'hyperparameters': {
        'gamma': 0.99,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'memory_size': 50000,
        'target_update_freq': 100,
        'hidden_sizes': [512, 256],
    },
    'exploration': {
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 0.9995,
    },
    'success_threshold': 18,
}

# FrozenLake configuration
FROZENLAKE_CONFIG = {
    'agent': 'QLearning',
    'env_name': 'FrozenLake-v1',
    'env_kwargs': {'is_slippery': True},
    'episodes': 10000,
    'hyperparameters': {
        'learning_rate': 0.1,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.9995,
    },
    'success_threshold': 0.7,  # 70% success rate
}

# All configurations
CONFIGS = {
    'cartpole': CARTPOLE_CONFIG,
    'pong': PONG_CONFIG,
    'frozenlake': FROZENLAKE_CONFIG,
}
