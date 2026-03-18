"""
Flappy Bird Game Environment (Pygame).
State: 7D continuous | Actions: 2 (noop, flap)
"""

import pygame
import random
import numpy as np
from typing import Tuple

SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
FPS = 60
BIRD_WIDTH, BIRD_HEIGHT, BIRD_X = 34, 24, 50
GRAVITY, FLAP_STRENGTH = 0.5, -7
PIPE_WIDTH, PIPE_GAP, PIPE_VELOCITY, PIPE_SPAWN_DIST = 70, 150, 3, 300

WHITE, BLACK, GREEN, BLUE, YELLOW, RED = (
    (255, 255, 255), (0, 0, 0), (0, 255, 0),
    (135, 206, 250), (255, 255, 0), (255, 0, 0),
)


class _Bird:
    def __init__(self):
        self.x, self.y, self.velocity = BIRD_X, SCREEN_HEIGHT // 2, 0
        self.rect = pygame.Rect(self.x, self.y, BIRD_WIDTH, BIRD_HEIGHT)

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = int(self.y)

    def draw(self, screen):
        cx, cy = int(self.x + BIRD_WIDTH // 2), int(self.y + BIRD_HEIGHT // 2)
        pygame.draw.circle(screen, YELLOW, (cx, cy), BIRD_WIDTH // 2)
        pygame.draw.circle(screen, BLACK, (cx + 5, cy - 3), 3)


class _Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(150, SCREEN_HEIGHT - 150 - PIPE_GAP)
        self.bottom_y = self.gap_y + PIPE_GAP
        self.passed = False
        self.top_rect = pygame.Rect(x, 0, PIPE_WIDTH, self.gap_y)
        self.bottom_rect = pygame.Rect(x, self.bottom_y, PIPE_WIDTH, SCREEN_HEIGHT - self.bottom_y)

    def update(self):
        self.x -= PIPE_VELOCITY
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.top_rect)
        pygame.draw.rect(screen, BLACK, self.top_rect, 2)
        pygame.draw.rect(screen, GREEN, self.bottom_rect)
        pygame.draw.rect(screen, BLACK, self.bottom_rect, 2)

    def collides(self, bird):
        return bird.rect.colliderect(self.top_rect) or bird.rect.colliderect(self.bottom_rect)

    def is_off_screen(self):
        return self.x + PIPE_WIDTH < 0


class FlappyBirdEnv:
    """Flappy Bird RL environment with consistent interface."""

    STATE_SIZE = 7
    ACTION_SIZE = 2

    def __init__(self, render: bool = False):
        self.render_mode = render
        pygame.init()
        if render:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        else:
            self.screen = self.clock = self.font = None
        self.bird = None
        self.pipes = []
        self.score = self.frames = 0

    def reset(self) -> np.ndarray:
        self.bird = _Bird()
        self.pipes = [_Pipe(SCREEN_WIDTH + i * PIPE_SPAWN_DIST) for i in range(3)]
        self.score = self.frames = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.frames += 1
        if action == 1:
            self.bird.flap()
        self.bird.update()
        for p in self.pipes:
            p.update()
        if self.pipes[-1].x < SCREEN_WIDTH - PIPE_SPAWN_DIST:
            self.pipes.append(_Pipe(SCREEN_WIDTH))
        if self.pipes[0].is_off_screen():
            self.pipes.pop(0)

        reward, done = 1.0, False

        # Position shaping reward
        closest = next((p for p in self.pipes if p.x + PIPE_WIDTH > self.bird.x), None)
        if closest and closest.x - self.bird.x < 150:
            dist = abs(self.bird.y + BIRD_HEIGHT / 2 - (closest.gap_y + PIPE_GAP / 2))
            if dist < 50:
                reward += 1.0
            elif dist < 100:
                reward += 0.5

        for p in self.pipes:
            if p.collides(self.bird):
                reward, done = -10.0, True
            if not p.passed and p.x + PIPE_WIDTH < self.bird.x:
                p.passed = True
                self.score += 1
                reward = 15.0

        if self.bird.y < 0 or self.bird.y > SCREEN_HEIGHT - BIRD_HEIGHT:
            reward, done = -10.0, True

        if self.render_mode:
            self._render()

        return self._get_state(), reward, done, {"score": self.score, "frames": self.frames}

    def _get_state(self) -> np.ndarray:
        nxt = next((p for p in self.pipes if p.x + PIPE_WIDTH > self.bird.x), self.pipes[0])
        return np.array([
            self.bird.y / SCREEN_HEIGHT,
            self.bird.velocity / 10,
            (nxt.x - self.bird.x) / SCREEN_WIDTH,
            nxt.gap_y / SCREEN_HEIGHT,
            nxt.bottom_y / SCREEN_HEIGHT,
            (nxt.x - self.bird.x) / SCREEN_WIDTH,
            ((nxt.gap_y + PIPE_GAP / 2) - self.bird.y) / SCREEN_HEIGHT,
        ], dtype=np.float32)

    def _render(self):
        if not self.screen:
            return
        self.screen.fill(BLUE)
        for p in self.pipes:
            p.draw(self.screen)
        self.bird.draw(self.screen)
        self.screen.blit(self.font.render(f"Score: {self.score}", True, WHITE), (10, 10))
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render_mode and pygame.get_init():
            pygame.quit()

    def get_state_size(self) -> int:
        return self.STATE_SIZE

    def get_action_size(self) -> int:
        return self.ACTION_SIZE
