# 强化学习游戏包 (Reinforcement Learning Game Package)

一个完整的强化学习项目，包含多个经典游戏环境的训练、测试和评估代码。

## 支持的游戏

1. **CartPole (倒立摆)** - 使用 DQN 算法
2. **Pong (乒乓球)** - 使用 DQN 算法
3. **FrozenLake (冰湖)** - 使用 Q-Learning 算法

## 项目结构

```
demo-project-rl/
├── agents/              # 强化学习算法实现
│   ├── dqn_agent.py    # Deep Q-Network
│   ├── qlearning_agent.py  # Q-Learning
│   └── base_agent.py   # 基础智能体类
├── experiments/         # 实验配置和脚本
│   ├── cartpole/       # CartPole 实验
│   ├── pong/           # Pong 实验
│   └── frozenlake/     # FrozenLake 实验
├── models/             # 保存的模型
├── results/            # 训练结果和日志
├── utils/              # 工具函数
│   ├── logger.py       # 日志记录
│   ├── plotter.py      # 结果可视化
│   └── replay_buffer.py # 经验回放缓冲区
├── train.py            # 统一训练脚本
├── test.py             # 统一测试脚本
├── requirements.txt    # 项目依赖
└── README.md           # 项目文档
```

## 安装

1. 克隆项目：
```bash
git clone <repository-url>
cd demo-project-rl
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

训练 CartPole：
```bash
python train.py --game cartpole --episodes 500
```

训练 Pong：
```bash
python train.py --game pong --episodes 2000
```

训练 FrozenLake：
```bash
python train.py --game frozenlake --episodes 10000
```

### 测试模型

测试训练好的模型：
```bash
python test.py --game cartpole --model models/cartpole_best.pth --episodes 10
```

### 可视化结果

训练过程中会自动保存：
- 训练曲线图到 `results/` 目录
- 模型检查点到 `models/` 目录
- 日志文件到 `results/logs/` 目录

## 算法说明

### DQN (Deep Q-Network)
- 用于 CartPole 和 Pong
- 使用经验回放和目标网络
- 适合连续观察空间

### Q-Learning
- 用于 FrozenLake
- 基于表格的方法
- 适合离散状态空间

## 性能基准

| 游戏 | 平均奖励 | 训练轮数 |
|------|----------|----------|
| CartPole | 195+ | 500 |
| Pong | 18+ | 2000 |
| FrozenLake | 0.7+ | 10000 |

## 依赖项

- Python 3.8+
- PyTorch
- Gymnasium (OpenAI Gym)
- NumPy
- Matplotlib
- TensorBoard (可选)

## 许可证

MIT License
