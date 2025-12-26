# Pong 实验配置

本目录包含 Pong 游戏的实验配置和脚本。

## 环境说明

- **游戏**: ALE/Pong-v5
- **状态空间**: 210x160x3 RGB 图像
- **动作空间**: 6 个离散动作
- **目标**: 击败 AI 对手
- **成功标准**: 平均分数 >= 18

## 算法

使用 Deep Q-Network (DQN)算法

## 超参数

```python
gamma = 0.99                # 折扣因子
learning_rate = 0.0001      # 学习率
batch_size = 32             # 批次大小
memory_size = 50000         # 经验回放缓冲区大小
target_update_freq = 100    # 目标网络更新频率
hidden_sizes = [512, 256]   # 隐藏层大小
epsilon_start = 1.0         # 初始探索率
epsilon_end = 0.1           # 最终探索率
epsilon_decay = 0.9995      # 探索率衰减
```

## 训练

```bash
python train.py --game pong --episodes 2000
```

**注意**: Pong 训练需要较长时间，建议使用 GPU 加速。

## 测试

```bash
python test.py --game pong --model models/pong_best.pth --episodes 5 --render
```

## 预期结果

- 训练轮数: ~2000 轮
- 平均奖励: 18+
- 训练时间: ~2-4 小时 (GPU) / ~8-12 小时 (CPU)

## 优化建议

1. 使用卷积神经网络处理图像
2. 实现帧堆叠(frame stacking)
3. 使用图像预处理(灰度化、裁剪)
4. 考虑使用 Double DQN 或 Dueling DQN
