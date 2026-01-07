"""
Testing script for Flappy Bird AI
"""

import os
import time
import numpy as np
import pygame

from envs.flappybird_env import FlappyBirdEnv
from agents.dqn_agent import DQNAgent


def test_flappybird(args):
    """Test DQN on Flappy Bird."""
    print("=" * 50)
    print("Testing DQN on Flappy Bird")
    print("=" * 50)

    model_path = getattr(args, "model", "experiments/flappybird/models/best_model.pth")

    # 初始化游戏
    env = FlappyBirdEnv(render=True)

    # 初始化智能体
    config = {
        "gamma": 0.99,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "memory_size": 100000,
        "target_update_freq": 10,
        "hidden_sizes": [256, 256, 256],
    }

    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        config=config,
    )

    # 加载模型
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    # 测试参数
    num_episodes = getattr(args, "episodes", 10)

    print(f"Running {num_episodes} test episodes...")
    print("Press ESC to quit, SPACE to pause")
    print("=" * 50)

    scores = []
    frames_list = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        paused = False

        running = True
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        env.close()
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            if not paused:
                # 使用AI选择动作（不使用探索）
                action = agent.select_action(state, epsilon=0.0)

                # 执行动作
                next_state, reward, done, info = env.step(action)

                total_reward += reward
                state = next_state

                # 渲染
                env.render()

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
                env.render()

    # 打印统计信息
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Min Score: {np.min(scores)}")
    print(f"Average Frames: {np.mean(frames_list):.2f}")
    print("=" * 50)

    env.close()
