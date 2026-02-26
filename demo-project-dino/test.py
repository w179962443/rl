"""
测试 / 演示脚本 — Chrome Dinosaur RL

用法：
    python test.py                           # 加载 models/best_model.pth 演示
    python test.py --model models/final_model.pth
    python test.py --episodes 20 --no_render # 无界面批量测试，输出统计
    python test.py --manual                  # 手动玩
"""

import os
import sys
import argparse
import time
import numpy as np

from game import DinoGame
from dqn_model import DQNAgent


def run_agent(model_path: str, episodes: int, render: bool, slow: bool):
    """运行 AI 智能体演示"""
    print(f"\n[AI 演示] 加载模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"  错误: 模型文件不存在 — {model_path}")
        print("  请先运行 train.py 训练模型，或指定 --model 参数")
        sys.exit(1)

    agent = DQNAgent(
        state_size=DinoGame.get_state_size(),
        action_size=DinoGame.get_action_size(),
    )
    agent.load(model_path)
    agent.epsilon = 0.0  # 关闭随机探索，纯贪心

    game = DinoGame(render=render)

    scores = []
    survivals = []

    print(f"\n{'Episode':>8}  {'Score':>8}  {'Steps':>8}  {'Speed':>8}")
    print("-" * 40)

    try:
        for ep in range(1, episodes + 1):
            state = game.reset()
            done = False
            total_steps = 0

            while not done:
                action = agent.select_action(state)
                state, reward, done, info = game.step(action)
                total_steps += 1

                if slow and render:
                    time.sleep(0.005)  # 慢放

            scores.append(info["score"])
            survivals.append(total_steps)
            print(
                f"  {ep:>5}    {info['score']:>8}  {total_steps:>8}  {info['speed']:>8.1f}"
            )

    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        game.close()

    if scores:
        print("\n─── 统计 ───────────────────────")
        print(f"  Episodes    : {len(scores)}")
        print(f"  平均分      : {np.mean(scores):.1f}")
        print(f"  最高分      : {np.max(scores)}")
        print(f"  最低分      : {np.min(scores)}")
        print(f"  中位数      : {np.median(scores):.1f}")
        print(f"  平均存活步数: {np.mean(survivals):.1f}")
        print("────────────────────────────────")


def run_manual():
    """手动游玩模式"""
    game = DinoGame(render=True)
    game.start_manual()


# ════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="Test Chrome Dino DQN Agent")
    p.add_argument(
        "--model", type=str, default="models/best_model.pth", help="模型文件路径"
    )
    p.add_argument("--episodes", type=int, default=10, help="测试回合数")
    p.add_argument(
        "--no_render", action="store_true", help="不显示游戏画面（批量测试用）"
    )
    p.add_argument("--slow", action="store_true", help="慢放，便于观察")
    p.add_argument("--manual", action="store_true", help="手动游玩模式")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.manual:
        run_manual()
    else:
        run_agent(
            model_path=args.model,
            episodes=args.episodes,
            render=not args.no_render,
            slow=args.slow,
        )
