"""
Snake Game Environment - Custom Gymnasium environment for Snake game.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum


class Direction(Enum):
    """Direction enum for snake movement."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class SnakeEnv(gym.Env):
    """
    Snake Game Environment compatible with Gymnasium.
    
    State: Grid representation of snake, food, and walls
    Action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Reward: +10 for eating food, -1 for each step, -10 for hitting wall/self
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, grid_size=10, render_mode=None):
        """
        Initialize Snake environment.
        
        Args:
            grid_size: Size of the game grid (grid_size x grid_size)
            render_mode: 'human' or 'rgb_array' or None
        """
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)
        
        # Observation space: flattened grid
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(grid_size * grid_size,), dtype=np.int32
        )
        
        self.snake = None
        self.food = None
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.game_over = False
        self.steps = 0
        self.max_steps = grid_size * grid_size * 10
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Initialize snake in the middle
        mid = self.grid_size // 2
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]
        
        # Place food randomly
        self.food = self._place_food()
        
        self.direction = Direction.RIGHT
        self.next_direction = Direction.RIGHT
        self.game_over = False
        self.steps = 0
        
        return self._get_observation(), {}
    
    def _place_food(self):
        """Place food at a random location not occupied by snake."""
        while True:
            food = (
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            )
            if food not in self.snake:
                return food
    
    def step(self, action):
        """Execute one step."""
        if self.game_over:
            return self._get_observation(), 0, True, False, {}
        
        self.steps += 1
        
        # Update direction (prevent reversing into self)
        direction = Direction(action)
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT,
        }
        
        if direction != opposite[self.direction]:
            self.direction = direction
        
        # Move snake
        head_x, head_y = self.snake[0]
        
        if self.direction == Direction.UP:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.DOWN:
            new_head = (head_x + 1, head_y)
        elif self.direction == Direction.LEFT:
            new_head = (head_x, head_y - 1)
        else:  # RIGHT
            new_head = (head_x, head_y + 1)
        
        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.game_over = True
            return self._get_observation(), -10, True, False, {}
        
        # Check collision with self
        if new_head in self.snake:
            self.game_over = True
            return self._get_observation(), -10, True, False, {}
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        reward = -1  # Step penalty
        if new_head == self.food:
            reward = 10  # Food reward
            self.food = self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
        
        # Check max steps
        if self.steps >= self.max_steps:
            self.game_over = True
            terminated = True
        else:
            terminated = False
        
        return self._get_observation(), reward, terminated, False, {}
    
    def _get_observation(self):
        """Get current observation as flattened grid."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Mark snake
        for segment in self.snake:
            if 0 <= segment[0] < self.grid_size and 0 <= segment[1] < self.grid_size:
                grid[segment[0], segment[1]] = 1
        
        # Mark food
        if self.food:
            grid[self.food[0], self.food[1]] = 2
        
        return grid.flatten()
    
    def render(self):
        """Render the game."""
        if self.render_mode is None:
            return None
        
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            if 0 <= segment[0] < self.grid_size and 0 <= segment[1] < self.grid_size:
                if i == 0:
                    grid[segment[0], segment[1]] = 'H'  # Head
                else:
                    grid[segment[0], segment[1]] = 'B'  # Body
        
        # Draw food
        if self.food:
            grid[self.food[0], self.food[1]] = 'F'
        
        # Print grid
        if self.render_mode == "human":
            print("\033c", end="")  # Clear screen
            for row in grid:
                print(' '.join(row))
            print(f"Length: {len(self.snake)}, Steps: {self.steps}")
            print()
    
    def close(self):
        """Close the environment."""
        pass
