"""
Chrome Dinosaur Game Environment
使用Pygame实现的Chrome恐龙跑酷游戏，支持强化学习训练

游戏规则：
- 按空格键/向上键：跳跃
- 按向下键：俯身（躲避翼龙）
- 躲开仙人掌和翼龙获得更高分数
- 游戏速度随分数线性增加
"""

import pygame
import random
import numpy as np
from typing import Tuple, Optional, List

# ────────────────────────────────────────────────
#  游戏常量
# ────────────────────────────────────────────────
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 300
FPS = 60
GROUND_Y = 240  # 地面 Y 坐标（恐龙脚底）

# 颜色（Chrome 风格灰度）
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (83, 83, 83)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (215, 215, 215)
SKY = (247, 247, 247)

# 恐龙参数
DINO_WIDTH = 44
DINO_HEIGHT = 47
DINO_DUCK_HEIGHT = 26
DINO_X = 80

# 物理参数
GRAVITY = 0.8
JUMP_VELOCITY = -15.0
MAX_FALL_SPEED = 15.0

# 游戏速度
INIT_SPEED = 6.0
MAX_SPEED = 18.0
SPEED_INCREMENT = 0.003  # 每帧增量

# 仙人掌参数
CACTUS_COLORS = [
    (0, 120, 0),
    (0, 100, 0),
    (0, 140, 0),
]
CACTUS_CONFIGS = [
    # (宽, 高, 茎数)
    (17, 35, 1),
    (25, 50, 1),
    (34, 35, 2),
    (40, 50, 2),
    (51, 35, 3),
]

# 翼龙参数
PTERO_WIDTH = 46
PTERO_HEIGHT = 40
PTERO_HEIGHTS = [
    GROUND_Y - DINO_HEIGHT - 10,  # 低空 — 需要俯身
    GROUND_Y - DINO_HEIGHT - 60,  # 中空 — 需要跳跃或俯身
    GROUND_Y - DINO_HEIGHT - 100,  # 高空 — 正常通过
]

# 最小障碍物间距（像素）
MIN_OBSTACLE_GAP = 400


# ════════════════════════════════════════════════
#  游戏对象类
# ════════════════════════════════════════════════


class Dinosaur:
    """恐龙角色"""

    def __init__(self):
        self.x = DINO_X
        self.y = float(GROUND_Y - DINO_HEIGHT)
        self.vel_y = 0.0
        self.on_ground = True
        self.ducking = False
        self.alive = True
        # 动画帧计数
        self._anim_tick = 0

    # ─── 动作 ───────────────────────────────────
    def jump(self):
        """跳跃（仅在地面时有效）"""
        if self.on_ground and not self.ducking:
            self.vel_y = JUMP_VELOCITY
            self.on_ground = False

    def duck(self, ducking: bool):
        """俯身 / 站立"""
        if ducking and self.on_ground:
            self.ducking = True
        else:
            self.ducking = False

    def update(self):
        """更新物理状态"""
        self._anim_tick += 1
        if not self.on_ground:
            self.vel_y = min(self.vel_y + GRAVITY, MAX_FALL_SPEED)
            self.y += self.vel_y

            ground = GROUND_Y - (DINO_DUCK_HEIGHT if self.ducking else DINO_HEIGHT)
            if self.y >= ground:
                self.y = float(ground)
                self.vel_y = 0.0
                self.on_ground = True
        else:
            # 保持贴地
            ground = GROUND_Y - (DINO_DUCK_HEIGHT if self.ducking else DINO_HEIGHT)
            self.y = float(ground)

    def get_rect(self) -> pygame.Rect:
        h = DINO_DUCK_HEIGHT if self.ducking else DINO_HEIGHT
        # 给碰撞盒留一点余量，更公平
        margin = 4
        return pygame.Rect(
            int(self.x) + margin,
            int(self.y) + margin,
            DINO_WIDTH - margin * 2,
            h - margin * 2,
        )

    def draw(self, screen: pygame.Surface):
        h = DINO_DUCK_HEIGHT if self.ducking else DINO_HEIGHT
        rect = pygame.Rect(int(self.x), int(self.y), DINO_WIDTH, h)

        # 身体
        pygame.draw.rect(screen, GRAY, rect, border_radius=4)

        # 眼睛
        eye_x = int(self.x) + DINO_WIDTH - 10
        eye_y = int(self.y) + 8
        pygame.draw.circle(screen, WHITE, (eye_x, eye_y), 6)
        pygame.draw.circle(screen, BLACK, (eye_x + 1, eye_y), 3)

        # 腿（跑步动画）
        if self.on_ground:
            phase = (self._anim_tick // 5) % 2
            leg_y = int(self.y) + h
            if phase == 0:
                pygame.draw.line(
                    screen,
                    DARK_GRAY,
                    (int(self.x) + 10, leg_y),
                    (int(self.x) + 5, leg_y + 12),
                    4,
                )
                pygame.draw.line(
                    screen,
                    DARK_GRAY,
                    (int(self.x) + 25, leg_y),
                    (int(self.x) + 30, leg_y + 12),
                    4,
                )
            else:
                pygame.draw.line(
                    screen,
                    DARK_GRAY,
                    (int(self.x) + 10, leg_y),
                    (int(self.x) + 15, leg_y + 12),
                    4,
                )
                pygame.draw.line(
                    screen,
                    DARK_GRAY,
                    (int(self.x) + 25, leg_y),
                    (int(self.x) + 20, leg_y + 12),
                    4,
                )


# ────────────────────────────────────────────────


class Cactus:
    """仙人掌障碍物"""

    def __init__(self, x: float):
        cfg = random.choice(CACTUS_CONFIGS)
        self.width, self.height, self.stems = cfg
        self.x = x
        self.y = GROUND_Y - self.height
        self.color = random.choice(CACTUS_COLORS)

    def update(self, speed: float):
        self.x -= speed

    def get_rect(self) -> pygame.Rect:
        margin = 3
        return pygame.Rect(
            int(self.x) + margin,
            int(self.y) + margin,
            self.width - margin * 2,
            self.height - margin * 2,
        )

    def draw(self, screen: pygame.Surface):
        stem_w = max(6, self.width // self.stems)
        # 主茎
        pygame.draw.rect(
            screen,
            self.color,
            (
                int(self.x) + (self.width - stem_w) // 2,
                int(self.y),
                stem_w,
                self.height,
            ),
        )
        # 侧臂（简化版）
        arm_h = self.height // 3
        arm_y = int(self.y) + self.height // 4
        arm_w = max(4, stem_w - 2)
        if self.stems >= 1:
            # 左臂
            pygame.draw.rect(screen, self.color, (int(self.x), arm_y, stem_w, arm_h))
            pygame.draw.rect(
                screen, self.color, (int(self.x), arm_y - arm_h // 2, arm_w, arm_h // 2)
            )
        if self.stems >= 2:
            # 右臂
            rx = int(self.x) + self.width - stem_w
            pygame.draw.rect(screen, self.color, (rx, arm_y + 10, stem_w, arm_h))
            pygame.draw.rect(
                screen, self.color, (rx, arm_y + 10 - arm_h // 2, arm_w, arm_h // 2)
            )

    @property
    def off_screen(self) -> bool:
        return self.x + self.width < 0


# ────────────────────────────────────────────────


class Pterodactyl:
    """翼龙障碍物"""

    def __init__(self, x: float):
        self.x = x
        self.height = random.choice(PTERO_HEIGHTS)
        self.y = float(self.height)
        self._tick = 0

    def update(self, speed: float):
        self.x -= speed
        self._tick += 1

    def get_rect(self) -> pygame.Rect:
        margin = 5
        return pygame.Rect(
            int(self.x) + margin,
            int(self.y) + margin,
            PTERO_WIDTH - margin * 2,
            PTERO_HEIGHT - margin * 2,
        )

    def draw(self, screen: pygame.Surface):
        cx = int(self.x) + PTERO_WIDTH // 2
        cy = int(self.y) + PTERO_HEIGHT // 2

        # 翅膀（振翅动画）
        wing_phase = (self._tick // 6) % 2
        wing_up = -12 if wing_phase == 0 else 8

        # 左翼
        pygame.draw.polygon(
            screen,
            DARK_GRAY,
            [
                (cx, cy),
                (cx - 25, cy + wing_up),
                (cx - 10, cy + 5),
            ],
        )
        # 右翼
        pygame.draw.polygon(
            screen,
            DARK_GRAY,
            [
                (cx, cy),
                (cx + 25, cy + wing_up),
                (cx + 10, cy + 5),
            ],
        )
        # 身体
        pygame.draw.ellipse(screen, GRAY, (cx - 12, cy - 8, 24, 16))
        # 头
        pygame.draw.circle(screen, GRAY, (cx + 14, cy - 4), 8)
        # 喙
        pygame.draw.line(screen, DARK_GRAY, (cx + 20, cy - 4), (cx + 30, cy - 6), 2)

    @property
    def off_screen(self) -> bool:
        return self.x + PTERO_WIDTH < 0


# ════════════════════════════════════════════════
#  主游戏类
# ════════════════════════════════════════════════


class DinoGame:
    """
    Chrome 恐龙游戏主类

    用法（手动游玩）::

        game = DinoGame(render=True)
        game.start_manual()

    用法（RL 环境接口）::

        game = DinoGame(render=False)
        state = game.reset()
        state, reward, done, info = game.step(action)
        # action: 0=不动, 1=跳跃, 2=俯身
    """

    # ─── Actions ───────────────────────────────
    ACTION_NOOOP = 0
    ACTION_JUMP = 1
    ACTION_DUCK = 2

    def __init__(self, render: bool = True, speed_multiplier: float = 1.0):
        self.render_mode = render
        self.speed_mult = speed_multiplier
        self._screen: Optional[pygame.Surface] = None
        self._clock = None
        self._font = None

        if render:
            self._init_pygame()

        self.reset()

    # ─── 初始化 ─────────────────────────────────
    def _init_pygame(self):
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Chrome Dino — RL Training")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 22, bold=True)

    # ─── RL 接口 ────────────────────────────────
    def reset(self) -> np.ndarray:
        """重置游戏，返回初始状态"""
        self.dino = Dinosaur()
        self.obstacles: List = []
        self.speed = INIT_SPEED * self.speed_mult
        self.score = 0.0
        self.steps = 0
        self._next_obs_x = SCREEN_WIDTH + random.randint(0, 200)
        self._ground_scroll = 0.0
        self._cloud_x = [SCREEN_WIDTH // 2, SCREEN_WIDTH]
        self._cloud_y = [80, 50]
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步

        Args:
            action: 0=不动, 1=跳跃, 2=俯身

        Returns:
            (state, reward, done, info)
        """
        # 处理动作
        if action == self.ACTION_JUMP:
            self.dino.jump()
            self.dino.duck(False)
        elif action == self.ACTION_DUCK:
            self.dino.duck(True)
        else:
            self.dino.duck(False)

        # 更新游戏速度
        self.speed = min(
            MAX_SPEED * self.speed_mult,
            self.speed + SPEED_INCREMENT * self.speed_mult,
        )

        # 更新恐龙
        self.dino.update()

        # 生成 / 更新障碍物
        self._spawn_obstacles()
        self._update_obstacles()

        # 碰撞检测
        dino_rect = self.dino.get_rect()
        collision = any(dino_rect.colliderect(obs.get_rect()) for obs in self.obstacles)

        # 积分
        self.score += self.speed / FPS
        self.steps += 1

        # 奖励设计
        if collision:
            reward = -10.0
            done = True
        else:
            reward = 0.1  # 每步存活奖励
            done = False

        # 渲染
        if self.render_mode:
            self._render()

        info = {
            "score": int(self.score),
            "speed": round(self.speed, 2),
            "steps": self.steps,
        }
        return self._get_state(), reward, done, info

    # ─── 状态 ────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        """
        返回 9 维状态向量（归一化）：
        [dino_y, dino_vel_y, ducking,
         obs1_dist, obs1_width, obs1_height, obs1_y,
         obs2_dist, obs_type]
        """
        dino_y_norm = (self.dino.y - (GROUND_Y - DINO_HEIGHT)) / DINO_HEIGHT
        vel_norm = self.dino.vel_y / abs(JUMP_VELOCITY)
        ducking = 1.0 if self.dino.ducking else 0.0
        speed_norm = (self.speed - INIT_SPEED) / (MAX_SPEED - INIT_SPEED)

        # 最近障碍物特征
        visible_obs = [
            o
            for o in self.obstacles
            if o.x + (getattr(o, "width", PTERO_WIDTH)) > DINO_X
        ]
        visible_obs.sort(key=lambda o: o.x)

        def obs_features(obs):
            if obs is None:
                return [1.0, 0.0, 0.0, 0.0, 0.0]
            dist = (obs.x - DINO_X) / SCREEN_WIDTH
            w = getattr(obs, "width", PTERO_WIDTH) / SCREEN_WIDTH
            h = getattr(obs, "height", PTERO_HEIGHT) / SCREEN_HEIGHT
            y = obs.y / SCREEN_HEIGHT
            typ = 0.0 if isinstance(obs, Cactus) else 1.0
            return [dist, w, h, y, typ]

        f1 = obs_features(visible_obs[0] if len(visible_obs) > 0 else None)
        f2 = obs_features(visible_obs[1] if len(visible_obs) > 1 else None)

        state = np.array(
            [dino_y_norm, vel_norm, ducking, speed_norm] + f1 + f2,
            dtype=np.float32,
        )
        return state

    @staticmethod
    def get_state_size() -> int:
        return 14

    @staticmethod
    def get_action_size() -> int:
        return 3

    # ─── 内部逻辑 ────────────────────────────────
    def _spawn_obstacles(self):
        """按间距生成新障碍物"""
        if not self.obstacles or self.obstacles[-1].x < self._next_obs_x:
            pass  # 等到最后一个障碍物滚过阈值再生
        # 用最右侧障碍物的 x 判断
        rightmost_x = max((o.x for o in self.obstacles), default=0)
        if rightmost_x < SCREEN_WIDTH - MIN_OBSTACLE_GAP or not self.obstacles:
            # 随机选择障碍类型：前期以仙人掌为主
            if self.score > 200 and random.random() < 0.3:
                obs = Pterodactyl(float(SCREEN_WIDTH + random.randint(50, 150)))
            else:
                obs = Cactus(float(SCREEN_WIDTH + random.randint(50, 150)))
            self.obstacles.append(obs)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs.update(self.speed)
        self.obstacles = [o for o in self.obstacles if not o.off_screen]

    # ─── 渲染 ────────────────────────────────────
    def _render(self):
        if self._screen is None:
            return

        self._screen.fill(SKY)

        # 云朵
        for i in range(len(self._cloud_x)):
            self._cloud_x[i] -= self.speed * 0.2
            if self._cloud_x[i] < -80:
                self._cloud_x[i] = SCREEN_WIDTH + random.randint(0, 200)
                self._cloud_y[i] = random.randint(30, 100)
            cx, cy = int(self._cloud_x[i]), self._cloud_y[i]
            pygame.draw.ellipse(self._screen, LIGHT_GRAY, (cx, cy + 10, 60, 20))
            pygame.draw.ellipse(self._screen, LIGHT_GRAY, (cx + 10, cy, 40, 25))
            pygame.draw.ellipse(self._screen, LIGHT_GRAY, (cx + 30, cy + 5, 35, 22))

        # 地面
        self._ground_scroll = (self._ground_scroll + self.speed) % 40
        pygame.draw.line(self._screen, GRAY, (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)
        for i in range(-1, SCREEN_WIDTH // 40 + 2):
            bx = int(i * 40 - self._ground_scroll)
            pygame.draw.line(
                self._screen, LIGHT_GRAY, (bx, GROUND_Y + 5), (bx + 15, GROUND_Y + 5), 2
            )

        # 障碍物
        for obs in self.obstacles:
            obs.draw(self._screen)

        # 恐龙
        self.dino.draw(self._screen)

        # 分数
        score_surf = self._font.render(f"SCORE  {int(self.score):06d}", True, GRAY)
        self._screen.blit(score_surf, (SCREEN_WIDTH - 220, 15))
        speed_surf = self._font.render(f"SPD {self.speed:.1f}", True, LIGHT_GRAY)
        self._screen.blit(speed_surf, (10, 15))

        pygame.display.flip()
        self._clock.tick(FPS)

    def close(self):
        if self.render_mode and pygame.get_init():
            pygame.quit()

    # ─── 手动游玩 ────────────────────────────────
    def start_manual(self):
        """启动手动游玩模式"""
        if not self.render_mode:
            self._init_pygame()
            self.render_mode = True

        print("Chrome Dino 手动模式")
        print("  空格 / ↑  : 跳跃")
        print("  ↓         : 俯身（躲避翼龙）")
        print("  ESC / Q   : 退出")

        running = True
        while running:
            self.reset()
            game_over = False
            while not game_over:
                action = self.ACTION_NOOOP
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        game_over = True
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                            game_over = True
                        if event.key in (pygame.K_SPACE, pygame.K_UP):
                            action = self.ACTION_JUMP
                keys = pygame.key.get_pressed()
                if keys[pygame.K_DOWN]:
                    action = self.ACTION_DUCK

                _, _, done, info = self.step(action)
                if done:
                    game_over = True
                    # 显示 Game Over
                    over_font = pygame.font.SysFont("consolas", 36, bold=True)
                    txt = over_font.render(
                        f"GAME OVER   Score: {info['score']}", True, DARK_GRAY
                    )
                    self._screen.blit(
                        txt,
                        (
                            SCREEN_WIDTH // 2 - txt.get_width() // 2,
                            SCREEN_HEIGHT // 2 - 20,
                        ),
                    )
                    pygame.display.flip()
                    pygame.time.wait(1800)

        self.close()


# ────────────────────────────────────────────────
#  入口
# ────────────────────────────────────────────────
if __name__ == "__main__":
    game = DinoGame(render=True)
    game.start_manual()
