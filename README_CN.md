# 强化学习多游戏训练框架 - 中文说明

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📌 项目介绍

这是一个**完整的生产级强化学习框架**，包含三个经典游戏环境的训练、测试和评估系统。

### 🎮 支持的游戏

| 游戏                 | 算法       | 难度        | 训练时间  |
| -------------------- | ---------- | ----------- | --------- |
| 🛒 CartPole (倒立摆) | DQN        | ⭐ 简单     | 5-10 分钟 |
| 🥊 Pong (乒乓球)     | DQN        | ⭐⭐⭐ 困难 | 2-4 小时  |
| ❄️ FrozenLake (冰湖) | Q-Learning | ⭐⭐ 中等   | 1-2 分钟  |

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆或进入项目目录
cd demo-project-rl

# 安装依赖
pip install -r requirements.txt

# 验证环境
python validate_setup.py
```

### 2. 运行演示 (1 分钟)

```bash
python demo.py
```

### 3. 训练模型

```bash
# CartPole - 最简单，推荐首先尝试
python train.py --game cartpole --episodes 500

# FrozenLake - 快速，适合学习Q-Learning
python train.py --game frozenlake --episodes 10000

# Pong - 复杂，需要更长训练时间
python train.py --game pong --episodes 2000
```

### 4. 测试模型

```bash
python test.py --game cartpole --model models/cartpole_best.pth --episodes 10
```

### 5. 查看结果

```bash
python evaluate.py --analyze results/logs/cartpole_500ep.json
```

## 📂 项目结构

```
demo-project-rl/
├── 📄 文档 (重要！请先阅读)
│   ├── README.md                    # 英文说明
│   ├── QUICKSTART.md               # 快速开始 (5分钟)
│   ├── INSTALL.md                  # 安装指南
│   ├── EXAMPLES.md                 # 10个详细示例
│   ├── DEVELOPMENT.md              # 开发指南
│   └── 项目完成总结.md              # 中文总结
│
├── 🤖 核心训练脚本
│   ├── train.py                    # 训练脚本
│   ├── test.py                     # 测试脚本
│   ├── evaluate.py                 # 评估脚本
│   ├── run_experiments.py          # 批量运行
│   └── demo.py                     # 演示脚本
│
├── 🧠 智能体实现 (agents/)
│   ├── base_agent.py              # 基础类
│   ├── dqn_agent.py               # DQN算法
│   └── qlearning_agent.py         # Q-Learning算法
│
├── 🛠️ 工具函数 (utils/)
│   ├── logger.py                  # 日志记录
│   ├── plotter.py                 # 结果可视化
│   └── visualize_qtable.py        # Q表可视化
│
├── 📊 实验文档 (experiments/)
│   ├── cartpole/                  # CartPole实验
│   ├── pong/                      # Pong实验
│   └── frozenlake/                # FrozenLake实验
│
├── 💾 生成的文件 (自动创建)
│   ├── models/                    # 保存的模型
│   └── results/                   # 训练结果和图表
│
├── ⚙️ 配置文件
│   ├── requirements.txt           # 依赖包列表
│   ├── config.py                  # 超参数配置
│   ├── .gitignore                 # Git忽略规则
│   └── validate_setup.py          # 环境验证脚本
```

## 📖 文档导航

根据你的需求选择合适的文档：

### 🟢 初学者入门 (5-15 分钟)

1. **QUICKSTART.md** - 快速开始指南
2. **demo.py** - 运行演示代码
3. **INSTALL.md** - 安装问题解决

### 🟡 进阶用户 (30 分钟-1 小时)

1. **EXAMPLES.md** - 10 个详细使用示例
2. **config.py** - 理解超参数
3. **train.py** - 阅读训练代码

### 🔴 开发者 (1 小时以上)

1. **DEVELOPMENT.md** - 开发和扩展指南
2. **agents/** - 查看算法实现
3. **utils/** - 理解工具函数

## 🎯 使用案例

### 案例 1: 快速验证环境 ⚡

```bash
# 只用1分钟验证一切是否工作
python demo.py
```

### 案例 2: 学习强化学习 📚

```bash
# 1. 阅读QUICKSTART.md
# 2. 运行CartPole实验
python train.py --game cartpole --episodes 500
# 3. 观察训练曲线
python evaluate.py --analyze results/logs/cartpole_500ep.json
```

### 案例 3: 研究不同算法 🔬

```bash
# 对比DQN (CartPole) 和 Q-Learning (FrozenLake)
python train.py --game cartpole --episodes 300
python train.py --game frozenlake --episodes 5000
python evaluate.py --all
```

### 案例 4: 优化超参数 ⚙️

```bash
# 编辑train.py中的config字典，修改超参数
# 然后训练多个版本进行对比
python train.py --game cartpole --episodes 500
# ... 修改参数后再运行一次 ...
python evaluate.py --all
```

## 💡 关键概念速览

### Q-Learning (FrozenLake)

- **原理**: 使用表格存储每个状态的动作值
- **适用**: 离散状态空间的小问题
- **训练快**: ~1 分钟完成

### DQN (CartPole & Pong)

- **原理**: 用神经网络逼近 Q 函数
- **适用**: 连续或高维状态空间
- **特点**: 使用经验回放和目标网络

### Epsilon-Greedy 策略

- **目的**: 平衡探索和利用
- **工作**: 以概率 ε 随机探索，否则选择最优动作
- **衰减**: 随着训练进行，ε 逐渐减小

## 🔧 常见任务速查

| 任务            | 命令                                                              |
| --------------- | ----------------------------------------------------------------- |
| 快速演示        | `python demo.py`                                                  |
| 验证环境        | `python validate_setup.py`                                        |
| 训练 CartPole   | `python train.py --game cartpole --episodes 500`                  |
| 训练 FrozenLake | `python train.py --game frozenlake --episodes 10000`              |
| 训练 Pong       | `python train.py --game pong --episodes 2000`                     |
| 测试模型        | `python test.py --game cartpole --model models/cartpole_best.pth` |
| 查看结果        | `python evaluate.py --analyze results/logs/*.json`                |
| 比较实验        | `python evaluate.py --compare log1.json log2.json`                |
| 可视化策略      | `python visualize_frozenlake.py`                                  |
| 批量运行        | `python run_experiments.py --game all`                            |

## 📊 预期性能

成功训练后的性能指标：

| 游戏       | 成功标准            | 预期性能             | 稳定性 |
| ---------- | ------------------- | -------------------- | ------ |
| CartPole   | avg_reward >= 195   | avg_reward ~200-210  | 高     |
| FrozenLake | success_rate >= 70% | success_rate ~75-85% | 中     |
| Pong       | avg_reward >= 18    | avg_reward ~20-30    | 低\*   |

\*Pong 性能较不稳定，取决于网络结构和超参数

## ⚠️ 常见问题快速解决

### Q: 如何加快训练速度？

A:

1. 使用 GPU 版 PyTorch
2. 减少 batch_size 或 hidden_sizes
3. 减少训练 episodes 进行测试

### Q: 内存不足怎么办？

A:

1. 减少 memory_size 和 batch_size
2. 关闭其他应用
3. 在较小的 episode 上测试

### Q: 模型性能不好？

A:

1. 增加训练轮数
2. 调整学习率（尝试 0.0005-0.005 之间的值）
3. 增加网络规模 (hidden_sizes)

### Q: 如何保存和加载模型？

A:

```python
# 保存
agent.save('my_model.pth')

# 加载
agent.load('my_model.pth')
```

## 📚 学习资源

- 📖 Sutton & Barto《强化学习导论》 - 理论基础
- 🎥 David Silver 强化学习课程 - 系统学习
- 📄 DQN 论文 - 深度 Q 网络方法
- 🔗 [Gymnasium 文档](https://gymnasium.farama.org/) - 环境 API

## 🔄 工作流程

### 标准训练流程

```
1. 准备环境 (validate_setup.py)
   ↓
2. 训练模型 (train.py)
   ↓
3. 保存结果 (自动生成)
   ↓
4. 测试模型 (test.py)
   ↓
5. 分析结果 (evaluate.py)
   ↓
6. 查看可视化 (results/plots/)
```

## 🚀 进阶用法

### 添加新算法

```python
# 1. 继承BaseAgent
from agents import BaseAgent

class PPOAgent(BaseAgent):
    def select_action(self, state, epsilon=0.0):
        # 实现你的算法
        pass

    def train_step(self, ...):
        pass

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass

# 2. 在train.py中添加训练函数
# 3. 测试和验证
```

### 添加新环境

```python
# 1. 在experiments/中创建新目录
# 2. 在train.py和test.py中添加对应函数
# 3. 配置超参数
# 4. 运行训练
```

## 🤝 贡献

欢迎提交改进建议或新功能！

## 📄 许可证

MIT License - 自由使用和修改

## 👨‍💻 开发者信息

- **项目版本**: 1.0.0
- **Python 要求**: 3.8+
- **主要依赖**: PyTorch, Gymnasium, NumPy, Matplotlib
- **最后更新**: 2025-12-26

## 🎓 这个项目教你什么？

✅ 强化学习基础 (Q-Learning, DQN)  
✅ 深度学习实践 (PyTorch)  
✅ 强化学习环境交互 (Gymnasium)  
✅ 工程最佳实践 (模块化、文档、日志)  
✅ 超参数调优方法  
✅ 结果分析和可视化

## 🎯 下一步行动

1. **现在就开始** ⚡

   ```bash
   python demo.py
   ```

2. **深入学习** 📚

   ```bash
   python train.py --game cartpole --episodes 500
   python evaluate.py --all
   ```

3. **探索更多** 🔍
   - 修改超参数看效果
   - 实现新算法
   - 添加新游戏环境

---

**祝你学习和研究愉快！** 🎉

有任何问题？查看详细文档：

- 快速开始 → QUICKSTART.md
- 使用示例 → EXAMPLES.md
- 开发指南 → DEVELOPMENT.md
- 完整说明 → README.md
