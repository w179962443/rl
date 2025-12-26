# FrozenLake 实验配置

本目录包含 FrozenLake 游戏的实验配置和脚本。

## 环境说明

- **游戏**: FrozenLake-v1
- **状态空间**: 16 个离散状态 (4x4 网格)
- **动作空间**: 4 个离散动作 (上, 下, 左, 右)
- **目标**: 从起点(S)到达目标(G)，避开洞(H)
- **难度**: 地面是滑的(is_slippery=True)
- **成功标准**: 成功率 >= 70%

## 地图

```
SFFF
FHFH
FFFH
HFFG
```

- S: 起点
- F: 冰面(可安全通过)
- H: 洞(掉入则失败)
- G: 目标(到达则成功)

## 算法

使用 Q-Learning 算法(表格方法)

## 超参数

```python
learning_rate = 0.1         # 学习率
gamma = 0.99                # 折扣因子
epsilon_start = 1.0         # 初始探索率
epsilon_end = 0.01          # 最终探索率
epsilon_decay = 0.9995      # 探索率衰减
```

## 训练

```bash
python train.py --game frozenlake --episodes 10000
```

## 测试

```bash
python test.py --game frozenlake --model models/frozenlake_best.pkl --episodes 100 --render
```

## 预期结果

- 训练轮数: ~10000 轮
- 成功率: 70-80%
- 训练时间: ~1-2 分钟

## 挑战

FrozenLake 是一个具有挑战性的环境因为:

1. 地面是滑的，动作有随机性
2. 稀疏奖励(只有到达目标才有奖励)
3. 需要长期规划

## 改进建议

1. 调整学习率和探索策略
2. 使用资格迹(eligibility traces)
3. 尝试其他算法如 SARSA
