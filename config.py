"""
Game configurations — each entry defines environment, agent, and hyperparameters.

Usage:
    python train.py --game flappybird
    python train.py --game flappybird --agent ppo
    python train.py --game dino --agent dueling_dqn
"""

# Shared PPO hyperparameters (can be overridden per game)
_PPO_BASE = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "learning_rate": 3e-4,
    "hidden_size": 256,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "n_epochs": 4,
    "batch_size": 64,
    "max_grad_norm": 0.5,
}

GAMES = {
    "flappybird": {
        "env": "flappybird",
        "default_agent": "dqn",
        "episodes": 10000,
        "max_steps": 10000,
        "eval_every": 200,
        "eval_episodes": 10,
        "save_every": 200,
        "log_every": 50,
        "agent_config": {
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "buffer_size": 100_000,
            "hidden_size": 256,
            "target_update": 10,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.9992,
            # PPO params (used when --agent ppo)
            **_PPO_BASE,
        },
    },
    "dino": {
        "env": "dino",
        "default_agent": "dueling_dqn",
        "episodes": 8000,
        "max_steps": 5000,
        "eval_every": 200,
        "eval_episodes": 10,
        "save_every": 200,
        "log_every": 50,
        "warmup_steps": 2000,
        "agent_config": {
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "buffer_size": 100_000,
            "hidden_size": 256,
            "target_update": 500,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.9995,
            # PPO params
            **_PPO_BASE,
        },
    },
    "snake": {
        "env": "snake",
        "default_agent": "dqn",
        "episodes": 5000,
        "max_steps": 500,
        "eval_every": 200,
        "eval_episodes": 10,
        "save_every": 200,
        "log_every": 50,
        "agent_config": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "buffer_size": 50_000,
            "hidden_size": 256,
            "target_update": 10,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.999,
            # PPO params
            **{**_PPO_BASE, "learning_rate": 1e-3},
        },
    },
    "cartpole": {
        "env": "CartPole-v1",
        "default_agent": "dqn",
        "episodes": 500,
        "max_steps": 500,
        "eval_every": 50,
        "eval_episodes": 10,
        "save_every": 100,
        "log_every": 10,
        "agent_config": {
            "gamma": 0.99,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "buffer_size": 10_000,
            "hidden_size": 128,
            "target_update": 10,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            # PPO params
            **{**_PPO_BASE, "hidden_size": 128, "learning_rate": 3e-4},
        },
    },
    "lunarlander": {
        "env": "LunarLander-v3",
        "default_agent": "dqn",
        "episodes": 1000,
        "max_steps": 1000,
        "eval_every": 100,
        "eval_episodes": 10,
        "save_every": 100,
        "log_every": 10,
        "agent_config": {
            "gamma": 0.99,
            "learning_rate": 5e-4,
            "batch_size": 64,
            "buffer_size": 100_000,
            "hidden_size": 256,
            "target_update": 10,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            # PPO params
            **{**_PPO_BASE, "learning_rate": 5e-4},
        },
    },
    "frozenlake": {
        "env": "FrozenLake-v1",
        "env_kwargs": {"is_slippery": True},
        "default_agent": "qlearning",
        "episodes": 10000,
        "max_steps": 100,
        "eval_every": 500,
        "eval_episodes": 100,
        "save_every": 1000,
        "log_every": 500,
        "agent_config": {
            "learning_rate": 0.1,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.9995,
            # PPO params (state_size=16 for one-hot encoding)
            **{**_PPO_BASE, "hidden_size": 64, "learning_rate": 1e-3, "n_epochs": 8},
        },
    },
}
