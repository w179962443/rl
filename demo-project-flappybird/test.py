"""
测试脚本
加载训练好的模型，测试AI玩Flappy Bird的表现
"""

import os
import time
import numpy as np
import pygame
from game import FlappyBirdGame
from dqn_model import DQNAgent


class Tester:
    """测试器类"""

    def __init__(self, model_path: str):
        """
        初始化测试器

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path

        # 初始化游戏
        self.game = FlappyBirdGame(render=True)

        # 初始化智能体
        self.agent = DQNAgent(
            state_size=self.game.get_state_size(),
            action_size=self.game.get_action_size(),
        )

        # 加载模型
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Error: Model file not found at {model_path}")
            exit(1)

    def test(self, num_episodes: int = 10, fps: int = 60):
        """
        测试模型

        Args:
            num_episodes: 测试回合数
            fps: 帧率（可以设置更高以加速观看）
        """
        print("=" * 60)
        print("Flappy Bird - AI Testing")
        print("=" * 60)
        print(f"Running {num_episodes} test episodes...")
        print("Press ESC to quit, SPACE to pause")
        print("=" * 60)

        scores = []
        frames_list = []

        for episode in range(1, num_episodes + 1):
            state = self.game.reset()
            total_reward = 0
            paused = False

            running = True
            while running:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        self.game.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            self.game.close()
                            return
                        if event.key == pygame.K_SPACE:
                            paused = not paused

                if not paused:
                    # 使用AI选择动作（不使用探索）
                    action = self.agent.select_action(state, training=False)

                    # 执行动作
                    next_state, reward, done, info = self.game.step(action)

                    total_reward += reward
                    state = next_state

                    # 渲染
                    self.game.render()

                    if done:
                        score = info["score"]
                        frames = info["frames"]
                        scores.append(score)
                        frames_list.append(frames)

                        print(
                            f"Episode {episode}/{num_episodes} | "
                            f"Score: {score} | "
                            f"Frames: {frames} | "
                            f"Total Reward: {total_reward:.2f}"
                        )

                        # 等待一下再开始下一局
                        time.sleep(1)
                        break
                else:
                    # 暂停状态下仅渲染
                    self.game.render()

        # 打印统计信息
        print("\n" + "=" * 60)
        print("Testing Complete!")
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Max Score: {np.max(scores)}")
        print(f"Min Score: {np.min(scores)}")
        print(f"Average Frames: {np.mean(frames_list):.2f}")
        print("=" * 60)

        self.game.close()

    def interactive_test(self):
        """
        交互式测试 - 可以随时切换人类/AI控制
        """
        print("=" * 60)
        print("Flappy Bird - Interactive Testing")
        print("=" * 60)
        print("Controls:")
        print("  SPACE - Jump (Human mode)")
        print("  A - Toggle AI mode")
        print("  R - Restart game")
        print("  ESC - Quit")
        print("=" * 60)

        state = self.game.reset()
        ai_mode = True  # 默认AI模式

        running = True
        while running:
            # 处理事件
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_a:
                        ai_mode = not ai_mode
                        mode_text = "AI" if ai_mode else "Human"
                        print(f"Switched to {mode_text} mode")
                    elif event.key == pygame.K_r:
                        state = self.game.reset()
                        print("Game reset")
                    elif event.key == pygame.K_SPACE and not ai_mode:
                        action = 1

            # 选择动作
            if ai_mode:
                action = self.agent.select_action(state, training=False)

            # 执行动作
            next_state, reward, done, info = self.game.step(action)
            state = next_state

            # 渲染（显示当前模式）
            self.game.render()

            # 在屏幕上显示模式
            if self.game.screen:
                mode_text = "AI Mode" if ai_mode else "Human Mode"
                mode_font = pygame.font.Font(None, 28)
                mode_surface = mode_font.render(mode_text, True, (255, 255, 255))
                self.game.screen.blit(mode_surface, (10, 90))
                pygame.display.flip()

            if done:
                print(f"Game Over! Score: {info['score']}, Frames: {info['frames']}")
                time.sleep(1)
                state = self.game.reset()

        self.game.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Flappy Bird AI")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pth",
        help="Path to the model file",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of test episodes"
    )
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (switch between AI and human)",
    )

    args = parser.parse_args()

    # 创建测试器
    tester = Tester(args.model)

    if args.interactive:
        # 交互模式
        tester.interactive_test()
    else:
        # 自动测试模式
        tester.test(num_episodes=args.episodes, fps=args.fps)


if __name__ == "__main__":
    main()
