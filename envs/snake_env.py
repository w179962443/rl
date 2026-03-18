"""
Snake Game Environment.
State: flattened grid (grid_size^2) | Actions: 4 (up, down, left, right)
"""

import numpy as np
from typing import Tuple


class SnakeEnv:
    """Snake game for RL training."""

    def __init__(self, grid_size: int = 10, render: bool = False):
        self.grid_size = grid_size
        self.render_mode = render
        self.state_size = grid_size * grid_size
        self.action_size = 4
        self.snake = []
        self.food = None
        self.direction = 3  # RIGHT
        self.steps = 0
        self.max_steps = grid_size * grid_size * 10
        self.rng = np.random.default_rng()

    def reset(self) -> np.ndarray:
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        self.direction = 3
        self.steps = 0
        self._place_food()
        return self._get_state()

    def _place_food(self):
        while True:
            pos = (self.rng.integers(0, self.grid_size), self.rng.integers(0, self.grid_size))
            if pos not in self.snake:
                self.food = pos
                return

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        if action != opposites.get(self.direction, -1):
            self.direction = action

        hx, hy = self.snake[0]
        dx_dy = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dx, dy = dx_dy[self.direction]
        new_head = (hx + dx, hy + dy)

        # Wall collision
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            return self._get_state(), -10.0, True, {"score": len(self.snake) - 3}

        # Self collision
        if new_head in self.snake:
            return self._get_state(), -10.0, True, {"score": len(self.snake) - 3}

        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 10.0
            self._place_food()
        else:
            reward = -0.1
            self.snake.pop()

        done = self.steps >= self.max_steps
        return self._get_state(), reward, done, {"score": len(self.snake) - 3}

    def _get_state(self) -> np.ndarray:
        grid = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        for seg in self.snake:
            grid[seg[0] * self.grid_size + seg[1]] = 1.0
        if self.food:
            grid[self.food[0] * self.grid_size + self.food[1]] = 2.0
        return grid

    def close(self):
        pass

    def get_state_size(self) -> int:
        return self.state_size

    def get_action_size(self) -> int:
        return self.action_size
