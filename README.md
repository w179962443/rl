# 强化学习游戏包 (Reinforcement Learning Game Package)

一个完整的强化学习项目，包含多个经典游戏环境的训练、测试和评估代码。

## 🎮 支持的游戏

1. **CartPole (倒立摆)** - 使用 DQN 算法
2. **Pong (乒乓球)** - 使用 DQN 算法
3. **FrozenLake (冰湖)** - 使用 Q-Learning 算法
4. **Snake (贪吃蛇)** - 使用 DQN 算法
5. **LunarLander (登月飞船)** - 使用 DQN 算法
6. **Breakout (打砖块)** - 使用 DQN 算法
7. **Flappy Bird (飞扬的小鸟)** - 使用 DQN 算法
8. **🆕 Super Mario Bros (超级马里奥)** - 使用 CNN-DQN 算法

## ✨ 新功能

### 游戏环境
- **Flappy Bird** - 完整的 Pygame 实现，7维状态空间
- **🆕 Super Mario Bros** - NES 经典游戏，CNN处理图像输入

### CNN-DQN 支持
- 🆕 专门的卷积神经网络 DQN Agent
- 支持图像状态输入（帧堆叠）
- 适用于 Atari、Mario 等视觉游戏

### 训练系统增强
- ✅ **自动保存训练图表** - 每N轮自动生成训练曲线
- ✅ **训练历史持久化** - 完整保存训练数据，支持断点续训
- ✅ **智能模型管理** - 自动保存最佳模型和定期检查点
- ✅ **定期渲染** - 可配置的游戏画面渲染频率
- ✅ **详细日志记录** - JSON和Pickle格式的训练日志

## 📁 项目结构

```
rl/
├── agents/              # 强化学习算法实现
│   ├── dqn_agent.py    # Deep Q-Network (改进版)
│   ├── qlearning_agent.py  # Q-Learning
│   └── base_agent.py   # 基础智能体类
├── envs/                # 游戏环境
│   ├── snake_env.py    # 贪吃蛇环境
│   └── flappybird_env.py  # 🆕 Flappy Bird环境
├── experiments/         # 实验配置和脚本
│   ├── cartpole/       
│   ├── pong/           
│   ├── frozenlake/     
│   ├── snake/
│   ├── lunarlander/
│   ├── breakout/
│   └── flappybird/     # 🆕 Flappy Bird实验
│       ├── train.py    # 训练脚本
│       ├── test.py     # 测试脚本
│       ├── models/     # 模型保存目录
│       └── logs/       # 日志和图表目录
├── utils/              # 工具函数
│   ├── logger.py       # 日志记录
│   ├── plotter.py      # 结果可视化
│   └── visualize_qtable.py
├── train.py            # 统一训练脚本
├── test.py             # 统一测试脚本
├── evaluate.py         # 评估脚本
├── requirements.txt    # 项目依赖
└── README.md           # 项目文档
```

## 🚀 快速开始

### 安装

### 训练模型

训练 CartPole：

```bash
python train.py --game cartpole --episodes 500
```

训练 Pong：

```bash
python train.py --game pong --episodes 2000
```
其他游戏

训练 CartPole：

```bash
python train.py --game cartpole --episodes 500 --render-every 50
```

训练 Snake：

```bash
python train.py --game snake --episodes 1000 --render-every 100
```

训练 LunarLander：

```bash
python train.py --game lunarlander --episodes 2000 --render-every 100
```

### 测试模型

测试 Flappy Bird：
🧠 算法说明

### DQN (Deep Q-Network)

- 用于 CartPole、Pong、Snake、LunarLander、Breakout 和 Flappy Bird
- 使用经验回放和目标网络
- 支持 epsilon-greedy 探索策略
- 适合连续观察空间
- **改进特性**:
  - 梯度裁剪
  - 可配置的网络结构
  - 自动目标网络更新

### Q-Learning

- 用于 FrozenLake
- 基于表格的方法
- 适合离散状态空间

## 📊 性能基准

| 游戏         | 目标平均奖励 | 建议训练轮数 | 状态空间   | 算法 |
| ------------ | ------------ | ------------ | ---------- | ---- |
| CartPole     | 195+         | 500          | 连续 (4)   | DQN |
| Pong         | 18+          | 2000         | 图像       | DQN |
| FrozenLake   | 0.7+         | 10000        | 离散 (16)  | Q-Learning |
| Snake        | 10+          | 1000         | 离散       | DQN |
| LunarLander  | 200+         | 2000         | 连续 (8)   | DQN |
| Breakout     | 30+          | 5000         | 图像       | DQN |
| Flappy Bird  | 50+          | 10000        | 连续 (7)   | DQN |
| **Super Mario** | **通关**  | **10000+**   | **图像 (4x84x84)** | **CNN-DQN** |

## 🔥 主要特性

### 1. 完善的训练系统
- 自动保存最佳模型
- 定期检查点保存
- 训练历史持久化
- 支持断点续训

### 2. 可视化工具
- 实时训练图表生成
- 多维度性能指标
- 训练过程录制（Flappy Bird）

### 3. 灵活的配置
- 可调的渲染频率
- 自定义训练参数
- 模块化的实验设计

### 4. 代码质量
- 清晰的项目结构
- 详细的文档说明
- 易于扩展的架构

## 📦 依赖项

- Python 3.8+
- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Pygame >= 2.5.0 (Flappy Bird)
- TensorBoard >= 2.13.0 (可选)

## 📚 相关文档

- [Flappy Bird 集成文档](FLAPPYBIRD_INTEGRATION.md) - 详细的集成说明
- [快速开始指南](QUICKSTART.md) - 新手入门
- [开发文档](DEVELOPMENT.md) - 开发者指南

## 🤝 贡献

欢迎贡献代码！请查看开发文档了解更多信息。

## 📄`

**训练图表包含：**
- 分数曲线和移动平均
- 训练损失曲线
- Epsilon探索率曲线
- 训练统计信息
- 适合离散状态空间

## 性能基准

| 游戏       | 平均奖励 | 训练轮数 |
| ---------- | -------- | -------- |
| CartPole   | 195+     | 500      |
| Pong       | 18+      | 2000     |
| FrozenLake | 0.7+     | 10000    |

## 依赖项

- Python 3.8+
- PyTorch
- Gymnasium (OpenAI Gym)
- NumPy
- Matplotlib
- TensorBoard (可选)

## 许可证

MIT License
