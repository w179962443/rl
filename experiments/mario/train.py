"""
Training script for Super Mario Bros using CNN-DQN
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

from envs.mario_env import MarioEnv
from agents.cnn_dqn_agent import CNNDQNAgent


class MarioTrainer:
    """Super Mario Bros 训练器"""

    def __init__(
        self,
        experiment_dir: str = "experiments/mario",
        world: int = 1,
        stage: int = 1,
        num_episodes: int = 10000,
        render_every: int = 100,
        save_every: int = 100,
        log_every: int = 10,
    ):
        """
        初始化训练器

        Args:
            experiment_dir: 实验目录
            world: 世界编号 (1-8)
            stage: 关卡编号 (1-4)
            num_episodes: 训练回合数
            render_every: 每隔多少回合渲染一次
            save_every: 每隔多少回合保存一次模型
            log_every: 每隔多少回合记录一次日志
        """
        self.experiment_dir = experiment_dir
        self.world = world
        self.stage = stage
        self.num_episodes = num_episodes
        self.render_every = render_every
        self.save_every = save_every
        self.log_every = log_every

        # 创建目录
        self.model_dir = os.path.join(experiment_dir, "models")
        self.log_dir = os.path.join(experiment_dir, "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"Training Mario World {world}-{stage}")
        print(f"Model Dir: {self.model_dir}")
        print(f"Log Dir: {self.log_dir}")

        # 初始化环境
        self.env = MarioEnv(
            world=world,
            stage=stage,
            render=False,
            movement="SIMPLE",
            frame_skip=4,
            frame_stack=4,
        )

        # 创建 CNN-DQN agent
        config = {
            "gamma": 0.99,
            "learning_rate": 0.00025,
            "batch_size": 32,
            "memory_size": 100000,
            "target_update_freq": 10000,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.999995,  # 很慢的衰减
        }

        self.agent = CNNDQNAgent(
            state_shape=self.env.get_state_size(),
            action_size=self.env.get_action_size(),
            config=config,
        )

        # 训练统计
        self.scores = []
        self.avg_scores = []
        self.losses = []
        self.epsilons = []
        self.max_x_positions = []
        self.max_score = 0
        self.max_x_pos = 0

        # 训练历史文件
        self.history_file = os.path.join(self.log_dir, "training_history.pkl")

    def train(self, resume_from: str = None):
        """
        开始训练

        Args:
            resume_from: 从已有模型继续训练
        """
        print("=" * 60)
        print("Super Mario Bros - CNN-DQN Training")
        print("=" * 60)

        # 加载已有模型
        start_episode = 1
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from {resume_from}")
            self.agent.load(resume_from)
            start_episode = self.agent.episodes_done + 1
            print(f"Continuing from episode {start_episode}")
            self.load_training_history()
        else:
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
                print("Cleared previous training history")

        start_time = time.time()
        end_episode = start_episode + self.num_episodes

        for episode in range(start_episode, end_episode):
            # 是否渲染
            render = episode % self.render_every == 0
            if render and not self.env.render_mode:
                self.env.close()
                self.env = MarioEnv(
                    world=self.world,
                    stage=self.stage,
                    render=True,
                    movement="SIMPLE",
                )
            elif not render and self.env.render_mode:
                self.env.close()
                self.env = MarioEnv(
                    world=self.world,
                    stage=self.stage,
                    render=False,
                    movement="SIMPLE",
                )

            # 重置环境
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            loss_count = 0
            done = False

            while not done:
                # 选择动作
                action = self.agent.select_action(state, self.agent.epsilon)

                # 执行动作
                next_state, reward, done, info = self.env.step(action)

                # 存储经验并训练
                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.train()

                if loss > 0:
                    episode_loss += loss
                    loss_count += 1

                episode_reward += reward
                state = next_state

            # 衰减 epsilon
            self.agent.epsilon = max(
                self.agent.epsilon_min, self.agent.epsilon * self.agent.epsilon_decay
            )
            self.agent.episodes_done = episode

            # 记录统计信息
            score = info.get("episode_score", 0)
            max_x = info.get("max_x_position", 0)
            
            self.scores.append(score)
            self.epsilons.append(self.agent.epsilon)
            self.max_x_positions.append(max_x)

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

            # 更新最高分和最远距离
            if score > self.max_score:
                self.max_score = score
                best_model_path = os.path.join(self.model_dir, "best_score_model.pth")
                self.agent.save(best_model_path)

            if max_x > self.max_x_pos:
                self.max_x_pos = max_x
                best_x_model_path = os.path.join(self.model_dir, "best_distance_model.pth")
                self.agent.save(best_x_model_path)

            # 打印日志
            if episode % self.log_every == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Ep {episode}/{end_episode-1} | "
                    f"Score: {score} | "
                    f"MaxX: {max_x} | "
                    f"AvgScore: {avg_score:.2f} | "
                    f"Epsilon: {self.agent.epsilon:.4f} | "
                    f"Loss: {self.losses[-1]:.4f} | "
                    f"Time: {elapsed_time:.1f}s"
                )

            # 保存模型和图表
            if episode % self.save_every == 0:
                checkpoint_path = os.path.join(
                    self.model_dir, f"checkpoint_episode_{episode}.pth"
                )
                self.agent.save(checkpoint_path)
                self.save_training_history()
                self.plot_training_progress(episode)

        # 保存最终模型
        final_model_path = os.path.join(self.model_dir, "final_model.pth")
        self.agent.save(final_model_path)
        self.save_training_history()
        self.plot_training_progress(end_episode - 1, final=True)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Max Score: {self.max_score}")
        print(f"Max Distance: {self.max_x_pos}")
        print(f"Final Avg Score (last 100): {self.avg_scores[-1]:.2f}")
        print(f"Total Time: {time.time() - start_time:.1f}s")
        print("=" * 60)

        self.env.close()

    def plot_training_progress(self, episode: int, final: bool = False):
        """绘制训练进度"""
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

        # 距离曲线
        axes[0, 1].plot(self.max_x_positions, alpha=0.5)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Max X Position")
        axes[0, 1].set_title("Max Distance Reached")
        axes[0, 1].grid(True)

        # Epsilon 和 Loss
        axes[1, 0].plot(self.epsilons)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Epsilon")
        axes[1, 0].set_title("Exploration Rate")
        axes[1, 0].grid(True)

        if len(self.losses) > 0:
            axes[1, 1].plot(self.losses, alpha=0.6)
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].set_title("Training Loss")
            axes[1, 1].grid(True)

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
        """保存训练历史"""
        history = {
            "scores": self.scores,
            "avg_scores": self.avg_scores,
            "losses": self.losses,
            "epsilons": self.epsilons,
            "max_x_positions": self.max_x_positions,
            "max_score": self.max_score,
            "max_x_pos": self.max_x_pos,
        }
        with open(self.history_file, "wb") as f:
            pickle.dump(history, f)

    def load_training_history(self):
        """加载训练历史"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "rb") as f:
                    history = pickle.load(f)
                self.scores = history.get("scores", [])
                self.avg_scores = history.get("avg_scores", [])
                self.losses = history.get("losses", [])
                self.epsilons = history.get("epsilons", [])
                self.max_x_positions = history.get("max_x_positions", [])
                self.max_score = history.get("max_score", 0)
                self.max_x_pos = history.get("max_x_pos", 0)
                print(f"Loaded training history: {len(self.scores)} episodes")
            except Exception as e:
                print(f"Failed to load training history: {e}")


def train_mario(args):
    """Train CNN-DQN on Super Mario Bros."""
    print("=" * 50)
    print("Training CNN-DQN on Super Mario Bros")
    print("=" * 50)

    # 获取world和stage参数
    world = getattr(args, "world", 1)
    stage = getattr(args, "stage", 1)

    # 创建训练器
    trainer = MarioTrainer(
        experiment_dir="experiments/mario",
        world=world,
        stage=stage,
        num_episodes=args.episodes,
        render_every=getattr(args, "render_every", 100),
        save_every=100,
        log_every=10,
    )

    # 查找最新检查点
    checkpoint_pattern = os.path.join(trainer.model_dir, "checkpoint_episode_*.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Found latest checkpoint: {latest_checkpoint}")
        trainer.train(resume_from=latest_checkpoint)
    else:
        print("No checkpoint found, starting from scratch...")
        trainer.train()
