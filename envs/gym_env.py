"""
Wrapper for standard Gymnasium environments (CartPole, LunarLander, FrozenLake, etc.).
Provides a consistent interface matching our custom envs.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple


class GymEnv:
    """Thin wrapper around gymnasium envs with unified interface."""

    def __init__(self, env_name: str, render: bool = False, **kwargs):
        self.env_name = env_name
        self.render_mode = render
        render_mode = "human" if render else None
        self.env = gym.make(env_name, render_mode=render_mode, **kwargs)

        obs_space = self.env.observation_space
        if hasattr(obs_space, "n"):
            # Discrete observation space (e.g. FrozenLake)
            self.state_size = obs_space.n
            self._discrete_obs = True
        else:
            self.state_size = int(np.prod(obs_space.shape))
            self._discrete_obs = False

        act_space = self.env.action_space
        self.action_size = act_space.n

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return self._process_obs(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info["score"] = info.get("score", reward)
        return self._process_obs(obs), float(reward), done, info

    def _process_obs(self, obs):
        if self._discrete_obs:
            return int(obs)
        return np.array(obs, dtype=np.float32).flatten()

    def close(self):
        self.env.close()

    def get_state_size(self) -> int:
        return self.state_size

    def get_action_size(self) -> int:
        return self.action_size
