"""
Super Mario Bros Environment
使用 gym-super-mario-bros 的环境包装器
"""

import numpy as np
from typing import Tuple, Optional
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import cv2


class MarioEnv:
    """Super Mario Bros 环境包装器"""

    def __init__(
        self,
        world: int = 1,
        stage: int = 1,
        version: int = 0,
        movement: str = "SIMPLE",
        render: bool = False,
        frame_skip: int = 4,
        frame_stack: int = 4,
        image_size: int = 84,
    ):
        """
        初始化 Mario 环境

        Args:
            world: 世界编号 (1-8)
            stage: 关卡编号 (1-4)
            version: 版本 (0=standard, 1=random, 2=enemies)
            movement: 动作集 ("SIMPLE", "COMPLEX", "RIGHT_ONLY")
            render: 是否渲染
            frame_skip: 帧跳过数（动作重复）
            frame_stack: 帧堆叠数
            image_size: 处理后的图像大小
        """
        self.world = world
        self.stage = stage
        self.render_mode = render
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.image_size = image_size

        # 创建环境
        env_name = f"SuperMarioBros-{world}-{stage}-v{version}"

        if render:
            self.env = gym_super_mario_bros.make(env_name, render_mode="human")
        else:
            self.env = gym_super_mario_bros.make(env_name)

        # 选择动作空间
        if movement == "SIMPLE":
            actions = SIMPLE_MOVEMENT
        elif movement == "COMPLEX":
            actions = COMPLEX_MOVEMENT
        elif movement == "RIGHT_ONLY":
            actions = RIGHT_ONLY
        else:
            actions = SIMPLE_MOVEMENT

        self.env = JoypadSpace(self.env, actions)
        self.action_size = self.env.action_space.n

        # 帧缓冲区用于帧堆叠
        self.frames = []

        # 统计信息
        self.current_score = 0
        self.max_x_position = 0
        self.steps = 0

    def reset(self) -> np.ndarray:
        """重置环境"""
        state = self.env.reset()

        if isinstance(state, tuple):
            state = state[0]

        # 初始化帧堆叠
        processed_frame = self._preprocess_frame(state)
        self.frames = [processed_frame] * self.frame_stack

        # 重置统计
        self.current_score = 0
        self.max_x_position = 0
        self.steps = 0

        return self._get_stacked_frames()

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧
        - 转换为灰度
        - 调整大小到 84x84
        - 归一化到 [0, 1]
        """
        # 转换为灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 调整大小
        resized = cv2.resize(
            gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )

        # 归一化
        normalized = resized / 255.0

        return normalized

    def _get_stacked_frames(self) -> np.ndarray:
        """获取堆叠的帧"""
        # 堆叠最近的 frame_stack 帧
        stacked = np.stack(self.frames, axis=0)
        return stacked

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作

        Args:
            action: 动作索引

        Returns:
            (state, reward, done, info)
        """
        total_reward = 0
        done = False
        info = {}

        # 执行 frame_skip 次相同动作
        for _ in range(self.frame_skip):
            next_state, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            info.update(step_info)
            self.steps += 1

            if done:
                break

        # 预处理并更新帧堆叠
        processed_frame = self._preprocess_frame(next_state)
        self.frames.pop(0)
        self.frames.append(processed_frame)

        # 获取堆叠帧作为状态
        state = self._get_stacked_frames()

        # 自定义奖励塑形
        custom_reward = self._shape_reward(reward, info, done)

        # 更新统计
        self.current_score = info.get("score", 0)
        current_x = info.get("x_pos", 0)
        if current_x > self.max_x_position:
            self.max_x_position = current_x

        # 添加额外信息
        info["episode_score"] = self.current_score
        info["max_x_position"] = self.max_x_position
        info["steps"] = self.steps

        return state, custom_reward, done, info

    def _shape_reward(self, reward: float, info: dict, done: bool) -> float:
        """
        奖励塑形 - 鼓励向右移动和得分

        Args:
            reward: 原始奖励
            info: 环境信息
            done: 是否结束

        Returns:
            塑形后的奖励
        """
        # 基础奖励
        shaped_reward = reward

        # 鼓励向右移动（探索）
        current_x = info.get("x_pos", 0)
        if current_x > self.max_x_position:
            shaped_reward += (current_x - self.max_x_position) * 0.1
            self.max_x_position = current_x

        # 死亡惩罚
        if done and info.get("flag_get", False) is False:
            shaped_reward -= 50

        # 成功通关奖励
        if info.get("flag_get", False):
            shaped_reward += 500

        return shaped_reward

    def render(self):
        """渲染环境（在创建时已设置render_mode）"""
        if self.render_mode:
            self.env.render()

    def close(self):
        """关闭环境"""
        self.env.close()

    def get_state_size(self) -> Tuple[int, int, int]:
        """获取状态空间大小 (channels, height, width)"""
        return (self.frame_stack, self.image_size, self.image_size)

    def get_action_size(self) -> int:
        """获取动作空间大小"""
        return self.action_size


if __name__ == "__main__":
    # 测试环境
    print("Testing Mario Environment...")

    env = MarioEnv(world=1, stage=1, render=True, movement="SIMPLE")

    print(f"State size: {env.get_state_size()}")
    print(f"Action size: {env.get_action_size()}")

    state = env.reset()
    print(f"Initial state shape: {state.shape}")

    # 运行几步
    for step in range(100):
        action = np.random.randint(0, env.get_action_size())
        next_state, reward, done, info = env.step(action)

        if step % 10 == 0:
            print(
                f"Step {step}: Reward={reward:.2f}, Score={info.get('score', 0)}, X={info.get('x_pos', 0)}"
            )

        if done:
            print(f"Episode finished at step {step}")
            print(f"Final score: {info.get('episode_score', 0)}")
            print(f"Max X position: {info.get('max_x_position', 0)}")
            break

    env.close()
    print("Test complete!")
