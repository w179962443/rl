# CartPole实验配置

本目录包含CartPole游戏的实验配置和脚本。

## 环境说明

- **游戏**: CartPole-v1
- **状态空间**: 4维连续空间 (位置, 速度, 角度, 角速度)
- **动作空间**: 2个离散动作 (左, 右)
- **目标**: 保持杆子直立尽可能长的时间
- **成功标准**: 连续100轮平均分数 >= 195

## 算法

使用Deep Q-Network (DQN)算法

## 超参数

```python
gamma = 0.99                # 折扣因子
learning_rate = 0.001       # 学习率
batch_size = 64             # 批次大小
memory_size = 10000         # 经验回放缓冲区大小
target_update_freq = 10     # 目标网络更新频率
hidden_sizes = [128, 128]   # 隐藏层大小
epsilon_start = 1.0         # 初始探索率
epsilon_end = 0.01          # 最终探索率
epsilon_decay = 0.995       # 探索率衰减
```

## 训练

```bash
python train.py --game cartpole --episodes 500
```

## 测试

```bash
python test.py --game cartpole --model models/cartpole_best.pth --episodes 10 --render
```

## 预期结果

- 训练轮数: ~500轮
- 平均奖励: 195+
- 训练时间: ~5-10分钟 (CPU)
