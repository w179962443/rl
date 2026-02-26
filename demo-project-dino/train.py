"""
训练脚本 — Chrome Dinosaur RL
使用 Double Dueling DQN 训练恐龙智能体

用法：
    python train.py                         # 从头训练
    python train.py --resume models/best_model.pth   # 继续训练
    python train.py --episodes 5000 --render_every 200
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无头绘图
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

from game import DinoGame
from dqn_model import DQNAgent


# ════════════════════════════════════════════════
#  超参数
# ════════════════════════════════════════════════
DEFAULT_CONFIG = dict(
    episodes=8000,
    max_steps=5000,
    hidden_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.9995,
    buffer_size=100_000,
    batch_size=64,
    target_update=500,
    save_every=200,
    log_every=20,
    render_every=500,  # 每隔多少 episode 渲染一次
    warmup_steps=2000,  # 预热步数（只收集不训练）
)


# ════════════════════════════════════════════════
#  Trainer
# ════════════════════════════════════════════════


class Trainer:
    def __init__(self, cfg: dict, model_dir: str = "models", log_dir: str = "logs"):
        self.cfg = cfg
        self.model_dir = model_dir
        self.log_dir = log_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.game = DinoGame(render=False)
        self.agent = DQNAgent(
            state_size=DinoGame.get_state_size(),
            action_size=DinoGame.get_action_size(),
            hidden_size=cfg["hidden_size"],
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            epsilon_start=cfg["epsilon_start"],
            epsilon_end=cfg["epsilon_end"],
            epsilon_decay=cfg["epsilon_decay"],
            buffer_size=cfg["buffer_size"],
            batch_size=cfg["batch_size"],
            target_update=cfg["target_update"],
        )

        # 历史记录
        self.episode_scores = []
        self.episode_steps = []
        self.episode_losses = []
        self.best_score = 0
        self.start_ep = 1

    def resume(self, path: str):
        self.agent.load(path)
        # 尝试恢复历史（同目录下的 history.npz）
        hist_path = os.path.join(self.model_dir, "history.npz")
        if os.path.exists(hist_path):
            d = np.load(hist_path)
            self.episode_scores = list(d["scores"])
            self.episode_steps = list(d["steps"])
            self.best_score = float(d["best_score"])
            self.start_ep = len(self.episode_scores) + 1
            print(
                f"[Resume] 从 episode {self.start_ep} 继续，历史最高分 {self.best_score:.0f}"
            )

    def _save_history(self):
        np.savez(
            os.path.join(self.model_dir, "history.npz"),
            scores=np.array(self.episode_scores),
            steps=np.array(self.episode_steps),
            best_score=np.array(self.best_score),
        )

    def _plot(self):
        if len(self.episode_scores) < 2:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 分数曲线
        scores = np.array(self.episode_scores)
        axes[0].plot(scores, alpha=0.4, color="steelblue", label="Score")
        if len(scores) >= 50:
            ma = np.convolve(scores, np.ones(50) / 50, mode="valid")
            axes[0].plot(range(49, len(scores)), ma, color="navy", label="MA50")
        axes[0].set_title("Episode Score")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Score")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 步数曲线
        steps = np.array(self.episode_steps)
        axes[1].plot(steps, alpha=0.4, color="coral", label="Steps")
        if len(steps) >= 50:
            ma = np.convolve(steps, np.ones(50) / 50, mode="valid")
            axes[1].plot(range(49, len(steps)), ma, color="darkred", label="MA50")
        axes[1].set_title("Episode Steps (Survival)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_curve.png"), dpi=100)
        plt.close(fig)

    def train(self):
        cfg = self.cfg
        total_eps = cfg["episodes"]
        max_steps = cfg["max_steps"]
        recent100 = deque(maxlen=100)

        print("=" * 60)
        print("  Chrome Dinosaur — DQN Training")
        print(f"  Episodes : {total_eps}")
        print(f"  Device   : {self.agent.device}")
        print(f"  State dim: {DinoGame.get_state_size()}")
        print(f"  Action   : 0=不动  1=跳跃  2=俯身")
        print("=" * 60)

        global_step = 0
        start_time = time.time()

        for ep in range(self.start_ep, self.start_ep + total_eps):
            render_this = ep % cfg["render_every"] == 0
            if render_this:
                self.game.render_mode = True
                if self.game._screen is None:
                    self.game._init_pygame()
            else:
                self.game.render_mode = False

            state = self.game.reset()
            ep_score = 0.0
            ep_loss = []

            for step in range(max_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.game.step(action)

                self.agent.store(state, action, reward, next_state, done)

                # 预热结束后正式训练
                if global_step >= cfg["warmup_steps"]:
                    loss = self.agent.train_step()
                    if loss is not None:
                        ep_loss.append(loss)

                state = next_state
                ep_score = info["score"]
                global_step += 1

                if done:
                    break

            # 记录
            self.episode_scores.append(ep_score)
            self.episode_steps.append(step + 1)
            self.episode_losses.append(np.mean(ep_loss) if ep_loss else 0.0)
            recent100.append(ep_score)

            # 保存最优模型
            if ep_score > self.best_score:
                self.best_score = ep_score
                self.agent.save(os.path.join(self.model_dir, "best_model.pth"))

            # 定期保存
            if ep % cfg["save_every"] == 0:
                ckpt = os.path.join(self.model_dir, f"checkpoint_ep{ep}.pth")
                self.agent.save(ckpt)
                self._save_history()
                self._plot()

            # 日志
            if ep % cfg["log_every"] == 0 or ep == self.start_ep:
                elapsed = time.time() - start_time
                avg100 = np.mean(recent100) if recent100 else 0.0
                print(
                    f"  Ep {ep:5d}/{self.start_ep + total_eps - 1}"
                    f"  Score={ep_score:6.0f}"
                    f"  Avg100={avg100:6.1f}"
                    f"  Best={self.best_score:6.0f}"
                    f"  ε={self.agent.epsilon:.4f}"
                    f"  Loss={self.episode_losses[-1]:.4f}"
                    f"  {elapsed/60:.1f}min"
                )

        # 最终保存
        self.agent.save(os.path.join(self.model_dir, "final_model.pth"))
        self._save_history()
        self._plot()
        self.game.close()

        print("\n[训练完成]")
        print(f"  历史最高分   : {self.best_score:.0f}")
        print(f"  最终 100 均分 : {np.mean(list(recent100)):.1f}")
        print(f"  模型保存至   : {self.model_dir}/")
        print(f"  训练曲线     : {self.log_dir}/training_curve.png")


# ════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(description="Train Chrome Dino DQN Agent")
    p.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["episodes"])
    p.add_argument("--hidden_size", type=int, default=DEFAULT_CONFIG["hidden_size"])
    p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        dest="learning_rate",
    )
    p.add_argument("--render_every", type=int, default=DEFAULT_CONFIG["render_every"])
    p.add_argument("--save_every", type=int, default=DEFAULT_CONFIG["save_every"])
    p.add_argument(
        "--resume", type=str, default=None, help="恢复训练：指定 .pth 文件路径"
    )
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--log_dir", type=str, default="logs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg["episodes"] = args.episodes
    cfg["hidden_size"] = args.hidden_size
    cfg["learning_rate"] = args.learning_rate
    cfg["render_every"] = args.render_every
    cfg["save_every"] = args.save_every

    trainer = Trainer(cfg, model_dir=args.model_dir, log_dir=args.log_dir)

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()
