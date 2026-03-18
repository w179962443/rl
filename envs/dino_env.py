"""
Chrome Dinosaur Game Environment (Pygame).
State: 14D continuous | Actions: 3 (noop, jump, duck)
"""

import pygame
import random
import numpy as np
from typing import Tuple, List, Optional

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 300
FPS = 60
GROUND_Y = 240
DINO_WIDTH, DINO_HEIGHT, DINO_DUCK_HEIGHT, DINO_X = 44, 47, 26, 80
GRAVITY, JUMP_VEL, MAX_FALL = 0.8, -15.0, 15.0
INIT_SPEED, MAX_SPEED, SPEED_INC = 6.0, 18.0, 0.003
PTERO_WIDTH, PTERO_HEIGHT = 46, 40
MIN_OBS_GAP = 400

SKY = (247, 247, 247)
GRAY = (83, 83, 83)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (215, 215, 215)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

CACTUS_CONFIGS = [
    (17, 35, 1), (25, 50, 1), (34, 35, 2), (40, 50, 2), (51, 35, 3),
]
CACTUS_COLORS = [(0, 120, 0), (0, 100, 0), (0, 140, 0)]
PTERO_HEIGHTS = [
    GROUND_Y - DINO_HEIGHT - 10,
    GROUND_Y - DINO_HEIGHT - 60,
    GROUND_Y - DINO_HEIGHT - 100,
]


class _Dino:
    def __init__(self):
        self.x = DINO_X
        self.y = float(GROUND_Y - DINO_HEIGHT)
        self.vel_y = 0.0
        self.on_ground = True
        self.ducking = False
        self._tick = 0

    def jump(self):
        if self.on_ground and not self.ducking:
            self.vel_y = JUMP_VEL
            self.on_ground = False

    def duck(self, d: bool):
        self.ducking = d and self.on_ground

    def update(self):
        self._tick += 1
        if not self.on_ground:
            self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL)
            self.y += self.vel_y
            ground = GROUND_Y - (DINO_DUCK_HEIGHT if self.ducking else DINO_HEIGHT)
            if self.y >= ground:
                self.y = float(ground)
                self.vel_y = 0.0
                self.on_ground = True
        else:
            self.y = float(GROUND_Y - (DINO_DUCK_HEIGHT if self.ducking else DINO_HEIGHT))

    def get_rect(self):
        h = DINO_DUCK_HEIGHT if self.ducking else DINO_HEIGHT
        m = 4
        return pygame.Rect(int(self.x) + m, int(self.y) + m, DINO_WIDTH - 2 * m, h - 2 * m)

    def draw(self, screen):
        h = DINO_DUCK_HEIGHT if self.ducking else DINO_HEIGHT
        r = pygame.Rect(int(self.x), int(self.y), DINO_WIDTH, h)
        pygame.draw.rect(screen, GRAY, r, border_radius=4)
        ex = int(self.x) + DINO_WIDTH - 10
        ey = int(self.y) + 8
        pygame.draw.circle(screen, WHITE, (ex, ey), 6)
        pygame.draw.circle(screen, BLACK, (ex + 1, ey), 3)


class _Cactus:
    def __init__(self, x):
        self.width, self.height, self.stems = random.choice(CACTUS_CONFIGS)
        self.x = x
        self.y = GROUND_Y - self.height
        self.color = random.choice(CACTUS_COLORS)

    def update(self, speed):
        self.x -= speed

    def get_rect(self):
        m = 3
        return pygame.Rect(int(self.x) + m, int(self.y) + m, self.width - 2 * m, self.height - 2 * m)

    def draw(self, screen):
        sw = max(6, self.width // self.stems)
        pygame.draw.rect(screen, self.color, (
            int(self.x) + (self.width - sw) // 2, int(self.y), sw, self.height
        ))

    @property
    def off_screen(self):
        return self.x + self.width < 0


class _Ptero:
    def __init__(self, x):
        self.x = x
        self.y = float(random.choice(PTERO_HEIGHTS))
        self.width = PTERO_WIDTH
        self.height = PTERO_HEIGHT
        self._tick = 0

    def update(self, speed):
        self.x -= speed
        self._tick += 1

    def get_rect(self):
        m = 5
        return pygame.Rect(int(self.x) + m, int(self.y) + m, PTERO_WIDTH - 2 * m, PTERO_HEIGHT - 2 * m)

    def draw(self, screen):
        cx = int(self.x) + PTERO_WIDTH // 2
        cy = int(self.y) + PTERO_HEIGHT // 2
        wu = -12 if (self._tick // 6) % 2 == 0 else 8
        pygame.draw.polygon(screen, DARK_GRAY, [(cx, cy), (cx - 25, cy + wu), (cx - 10, cy + 5)])
        pygame.draw.polygon(screen, DARK_GRAY, [(cx, cy), (cx + 25, cy + wu), (cx + 10, cy + 5)])
        pygame.draw.ellipse(screen, GRAY, (cx - 12, cy - 8, 24, 16))

    @property
    def off_screen(self):
        return self.x + PTERO_WIDTH < 0


class DinoEnv:
    """Chrome Dinosaur RL environment."""

    STATE_SIZE = 14
    ACTION_SIZE = 3  # 0=noop, 1=jump, 2=duck

    def __init__(self, render: bool = False):
        self.render_mode = render
        self._screen: Optional[pygame.Surface] = None
        self._clock = None
        self._font = None
        if render:
            self._init_pygame()
        else:
            pygame.init()
        self.dino = None
        self.obstacles: List = []
        self.speed = 0.0
        self.score = 0.0
        self.steps = 0

    def _init_pygame(self):
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Chrome Dino")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 22, bold=True)

    def reset(self) -> np.ndarray:
        self.dino = _Dino()
        self.obstacles = []
        self.speed = INIT_SPEED
        self.score = 0.0
        self.steps = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if action == 1:
            self.dino.jump()
            self.dino.duck(False)
        elif action == 2:
            self.dino.duck(True)
        else:
            self.dino.duck(False)

        self.speed = min(MAX_SPEED, self.speed + SPEED_INC)
        self.dino.update()
        self._spawn()
        for o in self.obstacles:
            o.update(self.speed)
        self.obstacles = [o for o in self.obstacles if not o.off_screen]

        collision = any(
            self.dino.get_rect().colliderect(o.get_rect()) for o in self.obstacles
        )
        self.score += self.speed / FPS
        self.steps += 1

        if collision:
            reward, done = -10.0, True
        else:
            reward, done = 0.1, False

        if self.render_mode:
            self._render()

        info = {"score": int(self.score), "speed": round(self.speed, 2), "steps": self.steps}
        return self._get_state(), reward, done, info

    def _spawn(self):
        rightmost = max((o.x for o in self.obstacles), default=0)
        if rightmost < SCREEN_WIDTH - MIN_OBS_GAP or not self.obstacles:
            x = float(SCREEN_WIDTH + random.randint(50, 150))
            if self.score > 200 and random.random() < 0.3:
                self.obstacles.append(_Ptero(x))
            else:
                self.obstacles.append(_Cactus(x))

    def _get_state(self) -> np.ndarray:
        dy = (self.dino.y - (GROUND_Y - DINO_HEIGHT)) / DINO_HEIGHT
        vn = self.dino.vel_y / abs(JUMP_VEL)
        dk = 1.0 if self.dino.ducking else 0.0
        sn = (self.speed - INIT_SPEED) / (MAX_SPEED - INIT_SPEED) if MAX_SPEED > INIT_SPEED else 0.0

        vis = sorted(
            [o for o in self.obstacles if o.x + getattr(o, "width", PTERO_WIDTH) > DINO_X],
            key=lambda o: o.x,
        )

        def feat(o):
            if o is None:
                return [1.0, 0.0, 0.0, 0.0, 0.0]
            return [
                (o.x - DINO_X) / SCREEN_WIDTH,
                getattr(o, "width", PTERO_WIDTH) / SCREEN_WIDTH,
                getattr(o, "height", PTERO_HEIGHT) / SCREEN_HEIGHT,
                o.y / SCREEN_HEIGHT,
                0.0 if isinstance(o, _Cactus) else 1.0,
            ]

        f1 = feat(vis[0] if len(vis) > 0 else None)
        f2 = feat(vis[1] if len(vis) > 1 else None)
        return np.array([dy, vn, dk, sn] + f1 + f2, dtype=np.float32)

    def _render(self):
        if not self._screen:
            return
        self._screen.fill(SKY)
        pygame.draw.line(self._screen, GRAY, (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)
        for o in self.obstacles:
            o.draw(self._screen)
        self.dino.draw(self._screen)
        txt = self._font.render(f"SCORE {int(self.score):06d}", True, GRAY)
        self._screen.blit(txt, (SCREEN_WIDTH - 220, 15))
        pygame.display.flip()
        self._clock.tick(FPS)

    def close(self):
        if self.render_mode and pygame.get_init():
            pygame.quit()

    def get_state_size(self) -> int:
        return self.STATE_SIZE

    def get_action_size(self) -> int:
        return self.ACTION_SIZE
