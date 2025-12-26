# 使用示例

本文档提供了如何使用此项目的具体示例。

## 例子 1: 快速演示 (1分钟)

最快速地体验项目功能：

```bash
python demo.py
```

这将在CartPole上训练DQN 50个episode。

## 例子 2: 训练CartPole (5-10分钟)

CartPole是最简单的环境，适合学习和测试：

```bash
# 训练
python train.py --game cartpole --episodes 500

# 查看训练日志
cat results/logs/cartpole_*.json | head

# 测试训练好的模型 (需要渲染)
python test.py --game cartpole --model models/cartpole_best.pth --episodes 5 --render

# 测试无渲染 (更快)
python test.py --game cartpole --model models/cartpole_best.pth --episodes 20

# 分析结果
python evaluate.py --analyze results/logs/cartpole_500ep.json
```

**预期结果**: 
- 训练后：平均奖励 > 190
- 成功率：50+ episodes达到195以上分数

## 例子 3: 训练FrozenLake (1-2分钟)

FrozenLake是一个离散环境，使用Q-Learning：

```bash
# 训练
python train.py --game frozenlake --episodes 10000

# 测试
python test.py --game frozenlake --model models/frozenlake_best.pkl --episodes 100

# 可视化学习到的策略
python visualize_frozenlake.py --model models/frozenlake_best.pkl
```

**预期结果**:
- 训练后：成功率 > 70%
- 生成的可视化显示最优策略

## 例子 4: 训练Pong (需要时间)

Pong是一个复杂的环境，需要更长的训练时间：

```bash
# 完整训练 (需要1-2小时)
python train.py --game pong --episodes 2000

# 如果只想测试，可以用更少的episodes
python train.py --game pong --episodes 100

# 测试
python test.py --game pong --model models/pong_best.pth --episodes 3
```

**提示**: 使用GPU会显著加快训练。

## 例子 5: 超参数调优

### 修改学习率

编辑 `train.py` 中对应游戏的config字典：

```python
def train_cartpole(args):
    config = {
        'learning_rate': 0.005,  # 改为 0.005 (原来是 0.001)
        # ... 其他参数
    }
```

然后重新训练：
```bash
python train.py --game cartpole --episodes 500
```

### 调整探索策略

编辑epsilon衰减参数：

```python
epsilon_start = 1.0
epsilon_end = 0.05      # 改为 0.05 (原来是 0.01)
epsilon_decay = 0.99    # 改为 0.99 (原来是 0.995)
```

更激进的衰减会加快学习但可能不稳定。

## 例子 6: 比较不同设置

训练多个版本进行对比：

```bash
# 版本1: 标准参数
python train.py --game cartpole --episodes 500

# 版本2: 更高的学习率 (需要修改train.py)
# ... 修改config中的learning_rate
python train.py --game cartpole --episodes 500

# 版本3: 更多的网络层 (需要修改train.py)
# ... 修改config中的hidden_sizes
python train.py --game cartpole --episodes 500

# 对比结果
python evaluate.py --all
```

## 例子 7: 批量运行所有实验

```bash
# 按顺序运行所有实验 (需要几小时)
python run_experiments.py --game all

# 或分别运行
python run_experiments.py --game cartpole
python run_experiments.py --game frozenlake
python run_experiments.py --game pong
```

## 例子 8: 自定义训练循环

创建一个自定义脚本 `my_training.py`：

```python
import gymnasium as gym
from agents import DQNAgent
from utils import Logger, Plotter

# 创建环境
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 创建智能体
agent = DQNAgent(state_size, action_size, {
    'gamma': 0.99,
    'learning_rate': 0.001,
    'batch_size': 64,
})

# 创建日志记录器
logger = Logger(experiment_name='custom_cartpole')

# 训练循环
for episode in range(100):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state, epsilon=0.1)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.train_step(state, action, reward, next_state, done)
        
        episode_reward += reward
        state = next_state
    
    logger.log_episode(episode, episode_reward, 0)
    print(f"Episode {episode}: {episode_reward}")

# 保存
agent.save('my_model.pth')
logger.save()
env.close()
```

运行：
```bash
python my_training.py
```

## 例子 9: 计算训练统计

创建 `compute_stats.py`：

```python
import json
import numpy as np
from pathlib import Path

log_dir = Path('results/logs')

for log_file in log_dir.glob('*.json'):
    with open(log_file) as f:
        data = json.load(f)
    
    rewards = [ep['reward'] for ep in data['episodes']]
    
    print(f"\n{data['experiment_name']}:")
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Average reward: {np.mean(rewards):.2f}")
    print(f"  Std dev: {np.std(rewards):.2f}")
    print(f"  Min: {np.min(rewards):.2f}, Max: {np.max(rewards):.2f}")
    
    if len(rewards) >= 100:
        last_100 = rewards[-100:]
        print(f"  Last 100 avg: {np.mean(last_100):.2f}")
```

运行：
```bash
python compute_stats.py
```

## 例子 10: 评估稳定性

验证模型的稳定性（多次运行）：

```bash
# 重复测试
for i in {1..10}; do
    python test.py --game cartpole --model models/cartpole_best.pth --episodes 1
done
```

或创建脚本 `test_stability.py`：

```python
from agents import DQNAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
agent.load('models/cartpole_best.pth')

results = []
for run in range(10):
    state, _ = env.reset()
    reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state, epsilon=0.0)
        next_state, r, terminated, truncated, _ = env.step(action)
        reward += r
        done = terminated or truncated
        state = next_state
    
    results.append(reward)
    print(f"Run {run+1}: {reward}")

print(f"\nAverage: {sum(results)/len(results):.2f}")
print(f"Std Dev: {(sum((x - sum(results)/len(results))**2 for x in results) / len(results)) ** 0.5:.2f}")

env.close()
```

## 故障排除

### 导入错误

```
ModuleNotFoundError: No module named 'agents'
```

解决方案：确保在项目根目录运行脚本

```bash
cd d:/demo-project-rl
python train.py --game cartpole
```

### CUDA相关错误

```
RuntimeError: CUDA out of memory
```

解决方案：
1. 减小batch_size
2. 减小hidden_sizes
3. 使用CPU：编辑dqn_agent.py，强制使用CPU

### 环境错误

```
ModuleNotFoundError: No module named 'gymnasium'
```

解决方案：
```bash
pip install -r requirements.txt
```

## 总结

这个项目提供了：
- ✓ 三个不同复杂度的游戏
- ✓ 两种主要的RL算法实现
- ✓ 完整的训练/测试框架
- ✓ 结果可视化工具
- ✓ 易于扩展的代码结构

祝你使用愉快！🎮
