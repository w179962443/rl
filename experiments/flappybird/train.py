"""
Training script for Flappy Bird using DQN
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import pickle

from envs.flappybird_env import FlappyBirdEnv
from agents.dqn_agent import DQNAgent


class FlappyBirdTrainer:
    """训练器类"""

    def __init__(
        self,
        experiment_dir: str = "experiments/flappybird",
        num_episodes: int = 10000,
        max_steps: int = 10000,
        render_every: int = 100,
        save_every: int = 100,
        log_every: int = 10,
    ):
        """
        初始化训练器

        Args:
            experiment_dir: 实验目录
            num_episodes: 训练回合数
            max_steps: 每回合最大步数
            render_every: 每隔多少回合渲染一次
            save_every: 每隔多少回合保存一次模型
            log_every: 每隔多少回合记录一次日志
        """
        self.experiment_dir = experiment_dir
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.render_every = render_every
        self.save_every = save_every
        self.log_every = log_every

        # 创建目录
        self.model_dir = os.path.join(experiment_dir, "models")
        self.log_dir = os.path.join(experiment_dir, "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"Model Dir: {self.model_dir}")
        print(f"Log Dir: {self.log_dir}")

        # 初始化游戏和智能体
        self.env = FlappyBirdEnv(render=False)
        
        # 创建DQN agent配置
        config = {
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "memory_size": 100000,
            "target_update_freq": 10,
            "hidden_sizes": [256, 256, 256],
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.9992,
        }
        
        self.agent = DQNAgent(
            state_size=self.env.get_state_size(),
            action_size=self.env.get_action_size(),
            config=config
        )

        # 训练统计
        self.scores = []
        self.avg_scores = []
        self.losses = []
        self.epsilons = []
        self.max_score = 0

        # 训练历史文件路径
        self.history_file = os.path.join(self.log_dir, "training_history.pkl")

    def train(self, resume_from: str = None):
        """
        开始训练

        Args:
            resume_from: 从已有模型继续训练
        """
        print("=" * 60)
        print("Flappy Bird - DQN Training")
        print("=" * 60)

        # 加载已有模型
        start_episode = 1
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from {resume_from}")
            self.agent.load(resume_from)
            # 从检查点获取已训练的轮数
            start_episode = getattr(self.agent, 'episodes_done', 0) + 1
            print(f"Continuing from episode {start_episode}")

            # 加载训练历史
            self.load_training_history()
        else:
            # 如果没有检查点但有历史文件，清空历史文件
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
                print("Cleared previous training history (starting fresh)")

        start_time = time.time()

        # 计算结束的 episode 数
        end_episode = start_episode + self.num_episodes

        for episode in range(start_episode, end_episode):
            # 是否渲染
            render = episode % self.render_every == 0
            if render and not self.env.render_mode:
                self.env.close()
                self.env = FlappyBirdEnv(render=True)
            elif not render and self.env.render_mode:
                self.env.close()
                self.env = FlappyBirdEnv(render=False)

            # 重置游戏
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            loss_count = 0

            for step in range(self.max_steps):
                # 处理pygame事件，避免窗口无响应
                if render:
                    import pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            # 用户关闭窗口，停止渲染但继续训练
                            render = False
                            self.env.close()
                            self.env = FlappyBirdEnv(render=False)
                            break

                # 选择动作
                action = self.agent.select_action(state, self.agent.epsilon)

                # 执行动作
                next_state, reward, done, info = self.env.step(action)

                # 存储经验并训练
                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.train()
                
                if loss is not None and loss > 0:
                    episode_loss += loss
                    loss_count += 1

                # 渲染
                if render:
                    self.env.render()

                episode_reward += reward
                state = next_state

                if done:
                    break

            # 更新目标网络
            if episode % self.agent.config["target_update_freq"] == 0:
                self.agent.update_target_network()

            # 衰减探索率
            self.agent.epsilon = max(
                self.agent.epsilon_min,
                self.agent.epsilon * self.agent.epsilon_decay
            )

            # 记录统计信息
            score = info["score"]
            self.scores.append(score)
            self.epsilons.append(self.agent.epsilon)

            if loss_count > 0:
                avg_loss = episode_loss / loss_count
                self.losses.append(avg_loss)
            else:
                self.losses.append(0)

            # 计算平均分数
            if len(self.scores) >= 100:
                avg_score = np.mean(self.scores[-100:])
            else:
                avg_score = np.mean(self.scores)
            self.avg_scores.append(avg_score)

            # 更新最高分
            if score > self.max_score:
                self.max_score = score
                # 保存最佳模型
                best_model_path = os.path.join(self.model_dir, "best_model.pth")
                self.agent.save(best_model_path)

            # 打印日志
            if episode % self.log_every == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Episode {episode}/{end_episode-1} | "
                    f"Score: {score} | "
                    f"Avg Score: {avg_score:.2f} | "
                    f"Max Score: {self.max_score} | "
                    f"Epsilon: {self.agent.epsilon:.3f} | "
                    f"Loss: {self.losses[-1]:.4f} | "
                    f"Time: {elapsed_time:.1f}s"
                )

            # 保存模型
            if episode % self.save_every == 0:
                checkpoint_path = os.path.join(
                    self.model_dir, f"checkpoint_episode_{episode}.pth"
                )
                self.agent.save(checkpoint_path)

                # 保存训练历史
                self.save_training_history()

                # 绘制训练曲线
                self.plot_training_progress(episode)

        # 保存最终模型
        final_model_path = os.path.join(self.model_dir, "final_model.pth")
        self.agent.save(final_model_path)

        # 保存训练历史
        self.save_training_history()

        # 绘制最终训练曲线
        self.plot_training_progress(end_episode - 1, final=True)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Max Score: {self.max_score}")
        print(f"Final Avg Score (last 100): {self.avg_scores[-1]:.2f}")
        print(f"Total Time: {time.time() - start_time:.1f}s")
        print("=" * 60)

        self.env.close()

    def plot_training_progress(self, episode: int, final: bool = False):
        """
        绘制训练进度

        Args:
            episode: 当前回合数
            final: 是否为最终图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 分数曲线
        axes[0, 0].plot(self.scores, alpha=0.3, label="Score")
        if len(self.avg_scores) > 0:
            axes[0, 0].plot(self.avg_scores, label="Avg Score (100)", linewidth=2)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_title("Training Score")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 损失曲线
        if len(self.losses) > 0:
            axes[0, 1].plot(self.losses, alpha=0.6)
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].set_title("Training Loss")
            axes[0, 1].grid(True)

        # Epsilon曲线
        axes[1, 0].plot(self.epsilons)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Epsilon")
        axes[1, 0].set_title("Exploration Rate")
        axes[1, 0].grid(True)

        # 统计信息
        stats_text = f"Episode: {episode}\n"
        stats_text += f"Max Score: {self.max_score}\n"
        if len(self.avg_scores) > 0:
            stats_text += f"Avg Score: {self.avg_scores[-1]:.2f}\n"
        stats_text += f"Epsilon: {self.agent.epsilon:.3f}\n"
        stats_text += f"Memory Size: {len(self.agent.memory)}"

        axes[1, 1].text(
            0.1,
            0.5,
            stats_text,
            fontsize=14,
            verticalalignment="center",
            family="monospace",
        )
        axes[1, 1].axis("off")

        plt.tight_layout()

        # 保存图表
        if final:
            filename = "training_final.png"
        else:
            filename = f"training_episode_{episode}.png"

        filepath = os.path.join(self.log_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        print(f"Training plot saved to {filepath}")

    def save_training_history(self):
        """
        保存训练历史到文件
        """
        history = {
            "scores": self.scores,
            "avg_scores": self.avg_scores,
            "losses": self.losses,
            "epsilons": self.epsilons,
            "max_score": self.max_score,
        }
        with open(self.history_file, "wb") as f:
            pickle.dump(history, f)

    def load_training_history(self):
        """
        从文件加载训练历史
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "rb") as f:
                    history = pickle.load(f)
                self.scores = history.get("scores", [])
                self.avg_scores = history.get("avg_scores", [])
                self.losses = history.get("losses", [])
                self.epsilons = history.get("epsilons", [])
                self.max_score = history.get("max_score", 0)
                print(f"Loaded training history: {len(self.scores)} episodes")
                print(f"Previous max score: {self.max_score}")
            except Exception as e:
                print(f"Failed to load training history: {e}")
                print("Starting with fresh history")
        else:
            print("No training history found, starting fresh")


def train_flappybird(args):
    """Train DQN on Flappy Bird."""
    print("=" * 50)
    print("Training DQN on Flappy Bird")
    print("=" * 50)

    # 创建训练器
    trainer = FlappyBirdTrainer(
        experiment_dir="experiments/flappybird",
        num_episodes=args.episodes,
        max_steps=10000,
        render_every=getattr(args, 'render_every', 100),
        save_every=100,
        log_every=10,
    )

    # 自动查找最新的检查点文件
    checkpoint_pattern = os.path.join(trainer.model_dir, "checkpoint_episode_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if checkpoint_files:
        # 根据文件修改时间排序，获取最新的文件
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Found latest checkpoint: {latest_checkpoint}")
        print(f"Resuming training from checkpoint...")
        trainer.train(resume_from=latest_checkpoint)
    else:
        print(f"No checkpoint found, starting from scratch...")
        trainer.train()
