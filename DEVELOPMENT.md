# 项目开发指南

## 项目概述

这是一个强化学习框架，支持多个经典游戏环境的训练、测试和评估。项目使用PyTorch实现神经网络，gymnasium库提供游戏环境。

## 核心概念

### 智能体(Agent)

智能体是与环境交互的学习实体。项目实现了两种主要的智能体：

1. **DQNAgent** - 深度Q网络
   - 使用神经网络估计Q值
   - 适用于连续或高维状态空间
   - 使用经验回放和目标网络技术
   - 用于CartPole和Pong

2. **QLearningAgent** - 表格Q学习
   - 使用Q表存储状态-动作价值
   - 适用于离散、低维状态空间
   - 更新规则：`Q(s,a) ← Q(s,a) + α(r + γ·max Q(s',a') - Q(s,a))`
   - 用于FrozenLake

### 环境(Environment)

使用gymnasium库提供的环境：

- **CartPole-v1**: 控制杆子保持直立
- **ALE/Pong-v5**: 经典乒乓球游戏
- **FrozenLake-v1**: 在冰面上导航到达目标

## 代码结构

### agents/

智能体实现：

```python
# 基础智能体类
class BaseAgent:
    def select_action(state, epsilon)     # 选择动作
    def train_step(state, action, ...)    # 训练一步
    def save(filepath)                    # 保存模型
    def load(filepath)                    # 加载模型
```

### utils/

工具函数：

- **logger.py**: 训练日志记录
- **plotter.py**: 结果可视化
- **visualize_qtable.py**: Q表可视化

### 训练脚本

```python
train.py          # 主训练脚本
test.py           # 模型测试脚本
evaluate.py       # 结果评估
visualize_frozenlake.py  # FrozenLake可视化
```

## 工作流程

### 1. 训练

```bash
python train.py --game cartpole --episodes 500
```

训练过程：
1. 初始化环境和智能体
2. 每个episode：
   - 重置环境
   - 根据epsilon-greedy策略选择动作
   - 与环境交互，收集经验
   - 训练神经网络/Q表
   - 衰减探索率
3. 定期保存最佳模型
4. 保存训练日志和可视化结果

### 2. 测试

```bash
python test.py --game cartpole --model models/cartpole_best.pth --episodes 10
```

测试过程：
- 加载已训练的模型
- 运行多个episode，不进行探索(epsilon=0)
- 计算平均性能指标

### 3. 评估

```bash
python evaluate.py --analyze results/logs/cartpole_500ep.json
```

分析训练结果：
- 统计奖励信息
- 绘制训练曲线
- 比较不同运行

## 关键超参数

### 通用超参数

```python
gamma = 0.99            # 折扣因子 - 未来奖励的权重
learning_rate = 0.001   # 学习率 - 学习步长
```

### 探索策略

```python
epsilon_start = 1.0     # 初始探索率
epsilon_end = 0.01      # 最终探索率
epsilon_decay = 0.995   # 每步的衰减因子
```

epsilon-greedy策略：
- 以概率ε随机选择动作(探索)
- 以概率(1-ε)选择最优动作(利用)
- 随着训练进行，ε逐渐减小

### DQN特定

```python
batch_size = 64         # 经验回放的批大小
memory_size = 10000     # 回放缓冲区容量
target_update_freq = 10 # 目标网络更新频率
```

## 扩展指南

### 添加新算法

1. 继承BaseAgent：
```python
from agents import BaseAgent

class MyAgent(BaseAgent):
    def select_action(self, state, epsilon=0.0):
        # 实现动作选择逻辑
        pass
    
    def train_step(self, state, action, reward, next_state, done):
        # 实现训练逻辑
        pass
    
    def save(self, filepath):
        # 保存模型
        pass
    
    def load(self, filepath):
        # 加载模型
        pass
```

2. 在train.py中添加训练函数

3. 在test.py中添加测试函数

### 添加新游戏环境

1. 在experiments/目录下创建新文件夹

2. 在train.py中添加训练函数：
```python
def train_newgame(args):
    env = gym.make('GameName-vX')
    # ... 初始化代码
    
    for episode in range(args.episodes):
        # ... 训练循环
        pass
```

3. 更新main()函数中的命令行参数

### 改进神经网络

对于Pong，可以使用卷积神经网络处理图像：

```python
class CNNDQN(nn.Module):
    def __init__(self, in_channels, action_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # ... 全连接层
```

## 常见问题排查

### 训练不收敛

1. 检查学习率是否过大/过小
2. 增加网络深度或宽度
3. 增加训练轮数
4. 检查reward是否正常

### 内存不足

1. 减少batch_size
2. 减少memory_size
3. 使用GPU加速
4. 降低图像分辨率(Pong)

### 模型过拟合

1. 增加探索时间
2. 增加网络正则化
3. 减少网络复杂度
4. 使用早停法

## 性能优化

### GPU加速

DQN会自动使用GPU，确保PyTorch有GPU支持：
```python
# 在dqn_agent.py中
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 并行训练

可以使用多进程并行运行多个实验：
```bash
python train.py --game cartpole --episodes 500 &
python train.py --game frozenlake --episodes 10000 &
wait
```

## 参考资源

- Gymnasium文档: https://gymnasium.farama.org/
- PyTorch官方教程: https://pytorch.org/tutorials/
- DQN论文: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- Q-Learning基础: Sutton & Barto的《强化学习导论》

## 许可证

MIT License - 自由使用和修改
