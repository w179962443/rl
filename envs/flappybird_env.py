"""
Flappy Bird Game Environment
使用Pygame实现的Flappy Bird游戏，支持强化学习训练
"""

import pygame
import random
import numpy as np
from typing import Tuple, Optional

# 游戏常量
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (135, 206, 250)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

# 小鸟参数
BIRD_WIDTH = 34
BIRD_HEIGHT = 24
BIRD_X = 50
GRAVITY = 0.5
FLAP_STRENGTH = -7  # 降低跳跃强度（原来-9太强）

# 管道参数
PIPE_WIDTH = 70
PIPE_GAP = 150
PIPE_VELOCITY = 3
PIPE_SPAWN_DISTANCE = 300


class Bird:
    """小鸟类"""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.velocity = 0
        self.rect = pygame.Rect(x, y, BIRD_WIDTH, BIRD_HEIGHT)

    def flap(self):
        """拍打翅膀，向上飞"""
        self.velocity = FLAP_STRENGTH

    def update(self):
        """更新小鸟位置"""
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = self.y

    def draw(self, screen: pygame.Surface):
        """绘制小鸟"""
        pygame.draw.circle(
            screen,
            YELLOW,
            (int(self.x + BIRD_WIDTH // 2), int(self.y + BIRD_HEIGHT // 2)),
            BIRD_WIDTH // 2,
        )
        # 眼睛
        pygame.draw.circle(
            screen,
            BLACK,
            (int(self.x + BIRD_WIDTH // 2 + 5), int(self.y + BIRD_HEIGHT // 2 - 3)),
            3,
        )

    def get_rect(self) -> pygame.Rect:
        """获取碰撞矩形"""
        return self.rect


class Pipe:
    """管道类"""

    def __init__(self, x: int):
        self.x = x
        self.gap_y = random.randint(150, SCREEN_HEIGHT - 150 - PIPE_GAP)
        self.top_height = self.gap_y
        self.bottom_y = self.gap_y + PIPE_GAP
        self.passed = False

        self.top_rect = pygame.Rect(x, 0, PIPE_WIDTH, self.top_height)
        self.bottom_rect = pygame.Rect(
            x, self.bottom_y, PIPE_WIDTH, SCREEN_HEIGHT - self.bottom_y
        )

    def update(self):
        """更新管道位置"""
        self.x -= PIPE_VELOCITY
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, screen: pygame.Surface):
        """绘制管道"""
        # 上管道
        pygame.draw.rect(screen, GREEN, self.top_rect)
        pygame.draw.rect(screen, BLACK, self.top_rect, 2)

        # 下管道
        pygame.draw.rect(screen, GREEN, self.bottom_rect)
        pygame.draw.rect(screen, BLACK, self.bottom_rect, 2)

    def collides(self, bird: Bird) -> bool:
        """检测碰撞"""
        bird_rect = bird.get_rect()
        return bird_rect.colliderect(self.top_rect) or bird_rect.colliderect(
            self.bottom_rect
        )

    def is_off_screen(self) -> bool:
        """检查是否离开屏幕"""
        return self.x + PIPE_WIDTH < 0


class FlappyBirdEnv:
    """Flappy Bird游戏环境"""

    def __init__(self, render: bool = False):
        """
        初始化游戏

        Args:
            render: 是否渲染游戏画面
        """
        self.render_mode = render

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird - RL Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        else:
            pygame.init()
            self.screen = None
            self.clock = None
            self.font = None

        self.reset()

    def reset(self) -> np.ndarray:
        """重置游戏状态"""
        self.bird = Bird(BIRD_X, SCREEN_HEIGHT // 2)
        self.pipes = [Pipe(SCREEN_WIDTH + i * PIPE_SPAWN_DISTANCE) for i in range(3)]
        self.score = 0
        self.game_over = False
        self.frames = 0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        获取游戏状态用于强化学习

        Returns:
            状态向量: [bird_y, bird_velocity, next_pipe_x, next_pipe_gap_y,
                      next_pipe_bottom_y, horizontal_distance, vertical_distance]
        """
        # 找到最近的管道
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                next_pipe = pipe
                break

        if next_pipe is None:
            next_pipe = self.pipes[0]

        # 归一化状态
        bird_y = self.bird.y / SCREEN_HEIGHT
        bird_velocity = self.bird.velocity / 10
        pipe_x = (next_pipe.x - self.bird.x) / SCREEN_WIDTH
        pipe_gap_y = next_pipe.gap_y / SCREEN_HEIGHT
        pipe_bottom_y = next_pipe.bottom_y / SCREEN_HEIGHT

        # 水平和垂直距离
        horizontal_distance = (next_pipe.x - self.bird.x) / SCREEN_WIDTH
        vertical_distance = (
            (next_pipe.gap_y + PIPE_GAP / 2) - self.bird.y
        ) / SCREEN_HEIGHT

        state = np.array(
            [
                bird_y,
                bird_velocity,
                pipe_x,
                pipe_gap_y,
                pipe_bottom_y,
                horizontal_distance,
                vertical_distance,
            ],
            dtype=np.float32,
        )

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步游戏

        Args:
            action: 0不跳，1跳

        Returns:
            (state, reward, done, info)
        """
        self.frames += 1

        # 执行动作
        if action == 1:
            self.bird.flap()

        # 更新游戏对象
        self.bird.update()

        for pipe in self.pipes:
            pipe.update()

        # 生成新管道
        if self.pipes[-1].x < SCREEN_WIDTH - PIPE_SPAWN_DISTANCE:
            self.pipes.append(Pipe(SCREEN_WIDTH))

        # 移除离开屏幕的管道
        if self.pipes[0].is_off_screen():
            self.pipes.pop(0)

        # 检测碰撞
        reward = 1.0  # 提高存活奖励，鼓励长期生存
        done = False

        # 添加位置奖励：鼓励小鸟保持在管道缝隙附近
        closest_pipe = None
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > self.bird.x:
                closest_pipe = pipe
                break

        # 根据与管道中心的距离给予奖励/惩罚
        if closest_pipe:
            gap_center = closest_pipe.gap_y + PIPE_GAP / 2
            bird_center = self.bird.y + BIRD_HEIGHT / 2
            distance_to_center = abs(bird_center - gap_center)

            # 当管道较近时，根据位置给奖励
            if closest_pipe.x - self.bird.x < 150:
                # 距离中心越近奖励越高
                if distance_to_center < 50:
                    reward += 1.0  # 非常接近中心
                elif distance_to_center < 100:
                    reward += 0.5  # 较接近中心
                elif distance_to_center > 150:
                    reward -= 0.3  # 离中心太远给小惩罚

        # 检查管道碰撞
        for pipe in self.pipes:
            if pipe.collides(self.bird):
                reward = -10  # 碰撞惩罚
                done = True
                self.game_over = True

            # 通过管道获得分数
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
                self.score += 1
                reward = 15  # 提高通过管道奖励，鼓励长期目标

        # 检查边界碰撞
        if self.bird.y < 0 or self.bird.y > SCREEN_HEIGHT - BIRD_HEIGHT:
            reward = -10  # 边界碰撞惩罚
            done = True
            self.game_over = True

        # 获取新状态
        state = self._get_state()

        info = {"score": self.score, "frames": self.frames}

        return state, reward, done, info

    def render(self):
        """渲染游戏画面"""
        if not self.render_mode or self.screen is None:
            return

        # 清空屏幕
        self.screen.fill(BLUE)

        # 绘制管道
        for pipe in self.pipes:
            pipe.draw(self.screen)

        # 绘制小鸟
        self.bird.draw(self.screen)

        # 绘制分数
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # 绘制帧数
        frames_text = self.font.render(f"Frames: {self.frames}", True, WHITE)
        self.screen.blit(frames_text, (10, 50))

        if self.game_over:
            game_over_text = self.font.render("GAME OVER", True, RED)
            text_rect = game_over_text.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            )
            self.screen.blit(game_over_text, text_rect)

        pygame.display.flip()

        if self.clock:
            self.clock.tick(FPS)

    def close(self):
        """关闭游戏"""
        if self.render_mode:
            pygame.quit()

    def get_state_size(self) -> int:
        """获取状态空间大小"""
        return 7

    def get_action_size(self) -> int:
        """获取动作空间大小"""
        return 2  # 0: 不跳, 1: 跳
