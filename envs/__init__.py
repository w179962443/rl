"""Environments package."""

from .snake_env import SnakeEnv
from .flappybird_env import FlappyBirdEnv
from .mario_env import MarioEnv
from .dino_env import DinoBirdEnv

__all__ = ["SnakeEnv", "FlappyBirdEnv", "MarioEnv", "DinoBirdEnv"]
