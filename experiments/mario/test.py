"""
Testing script for Super Mario Bros AI
"""

import os
import time
import numpy as np

from envs.mario_env import MarioEnv
from agents.cnn_dqn_agent import CNNDQNAgent


def test_mario(args):
    """Test CNN-DQN on Super Mario Bros."""
    print("=" * 50)
    print("Testing CNN-DQN on Super Mario Bros")
    print("=" * 50)

    model_path = getattr(args, "model", "experiments/mario/models/best_score_model.pth")
    world = getattr(args, "world", 1)
    stage = getattr(args, "stage", 1)

    # 初始化环境（渲染模式）
    env = MarioEnv(
        world=world,
        stage=stage,
        render=True,
        movement="SIMPLE",
        frame_skip=4,
        frame_stack=4,
    )

    # 初始化智能体
    config = {
        "gamma": 0.99,
        "learning_rate": 0.00025,
        "batch_size": 32,
        "memory_size": 100000,
        "target_update_freq": 10000,
    }

    agent = CNNDQNAgent(
        state_shape=env.get_state_size(),
        action_size=env.get_action_size(),
        config=config,
    )

    # 加载模型
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    # 测试参数
    num_episodes = getattr(args, "episodes", 10)

    print(f"Running {num_episodes} test episodes...")
    print("Press ESC to quit")
    print("=" * 50)

    scores = []
    max_x_positions = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # 使用AI选择动作（不探索）
            action = agent.select_action(state, epsilon=0.0)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state
            steps += 1

            # 渲染
            env.render()

            if done:
                score = info.get("episode_score", 0)
                max_x = info.get("max_x_position", 0)
                flag_get = info.get("flag_get", False)

                scores.append(score)
                max_x_positions.append(max_x)

                status = "COMPLETED!" if flag_get else "Failed"
                print(
                    f"Episode {episode}/{num_episodes} | "
                    f"Score: {score} | "
                    f"Max X: {max_x} | "
                    f"Steps: {steps} | "
                    f"Status: {status}"
                )

                # 等待一下再继续
                time.sleep(2)
                break

    # 打印统计
    print("\n" + "=" * 50)
    print("Testing Complete!")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Average Distance: {np.mean(max_x_positions):.2f}")
    print(f"Max Distance: {np.max(max_x_positions)}")
    print("=" * 50)

    env.close()
