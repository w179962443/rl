"""
Chrome Dinosaur Game — 主项目环境封装
将 demo-project-dino/game.py 的 DinoGame 封装成与项目其他环境一致的接口
"""

import sys
import os

# 将 demo-project-dino 加入路径
_DINO_DIR = os.path.join(os.path.dirname(__file__), "..", "demo-project-dino")
if _DINO_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_DINO_DIR))

from game import DinoGame  # noqa: E402


class DinoBirdEnv:
    """
    Chrome Dinosaur Gym-style 环境封装

    状态空间：14 维连续向量（见 game.py 注释）
    动作空间：3 个离散动作（0=不动, 1=跳跃, 2=俯身）
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render: bool = False):
        self._game = DinoGame(render=render)
        self.state_size = DinoGame.get_state_size()
        self.action_size = DinoGame.get_action_size()
        self.action_space_n = self.action_size

    # ─── 标准接口 ────────────────────────────────
    def reset(self):
        return self._game.reset()

    def step(self, action: int):
        return self._game.step(action)

    def render(self):
        self._game.render_mode = True
        if self._game._screen is None:
            self._game._init_pygame()

    def close(self):
        self._game.close()

    # ─── 辅助属性 ────────────────────────────────
    @property
    def observation_space_shape(self):
        return (self.state_size,)

    def get_state_size(self):
        return self.state_size

    def get_action_size(self):
        return self.action_size

    def __repr__(self):
        return (
            f"DinoBirdEnv("
            f"state_size={self.state_size}, "
            f"action_size={self.action_size})"
        )
