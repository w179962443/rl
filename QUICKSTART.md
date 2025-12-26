# 快速开始指南

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**: 安装 Atari 游戏环境可能需要一些时间。

## 2. 运行第一个实验 - CartPole

CartPole 是最简单的环境，适合快速测试：

```bash
python train.py --game cartpole --episodes 500
```

训练完成后，测试模型：

```bash
python test.py --game cartpole --model models/cartpole_best.pth --episodes 10 --render
```

## 3. 训练 FrozenLake

FrozenLake 是表格方法的经典例子：

```bash
python train.py --game frozenlake --episodes 10000
```

测试：

```bash
python test.py --game frozenlake --model models/frozenlake_best.pkl --episodes 100 --render
```

## 4. 训练 Pong (需要较长时间)

```bash
python train.py --game pong --episodes 2000
```

**提示**: Pong 训练时间较长，建议在 GPU 上运行或减少训练轮数进行测试。

## 5. 查看结果

训练完成后，查看生成的图表：

- `results/plots/` - 训练曲线图
- `results/logs/` - JSON 格式的训练日志
- `models/` - 保存的模型文件

## 6. 评估和比较

分析单个实验：

```bash
python evaluate.py --analyze results/logs/cartpole_500ep.json
```

比较多个实验：

```bash
python evaluate.py --compare results/logs/cartpole_500ep.json results/logs/cartpole_1000ep.json
```

分析所有实验：

```bash
python evaluate.py --all
```

## 7. 运行所有实验

如果你想一次性运行所有实验：

```bash
python run_experiments.py --game all
```

或运行单个游戏：

```bash
python run_experiments.py --game cartpole
```

## 常见问题

### Q: 如何调整训练参数？

A: 编辑 [train.py](train.py) 中对应游戏的配置字典。

### Q: 训练太慢了怎么办？

A:

1. 确保安装了 PyTorch 的 GPU 版本
2. 减少训练轮数进行测试
3. 减小 batch_size 或 memory_size

### Q: 如何保存训练过程的视频？

A: 可以使用 gymnasium 的`RecordVideo`包装器：

```python
from gymnasium.wrappers import RecordVideo
env = RecordVideo(env, video_folder='./videos/')
```

### Q: 模型性能不好怎么办？

A:

1. 增加训练轮数
2. 调整学习率
3. 调整探索率衰减
4. 增加网络层数或隐藏单元数

## 项目结构说明

```
demo-project-rl/
├── agents/              # RL算法实现
│   ├── base_agent.py   # 基类
│   ├── dqn_agent.py    # DQN算法
│   └── qlearning_agent.py  # Q-Learning算法
├── experiments/         # 实验文档
├── models/             # 保存的模型 (训练后生成)
├── results/            # 结果和日志 (训练后生成)
│   ├── logs/          # JSON日志
│   └── plots/         # 训练曲线图
├── utils/              # 工具函数
│   ├── logger.py      # 日志记录
│   └── plotter.py     # 可视化
├── train.py           # 训练脚本
├── test.py            # 测试脚本
├── evaluate.py        # 评估脚本
└── run_experiments.py # 批量实验脚本
```

## 下一步

1. 尝试修改超参数，观察对训练的影响
2. 实现新的算法（如 PPO、A3C）
3. 添加新的游戏环境
4. 实现更复杂的神经网络结构（如 CNN for Pong）
5. 添加 TensorBoard 支持进行更详细的可视化

祝你训练愉快！ 🚀
