# RL Games

用强化学习算法玩各种游戏。

## 项目结构

```
agents/          # RL 算法实现
  dqn_agent.py         - Double DQN
  dueling_dqn_agent.py - Double Dueling DQN
  reinforce_agent.py   - REINFORCE (策略梯度)
  a2c_agent.py         - Advantage Actor-Critic
  qlearning_agent.py   - 表格型 Q-Learning
envs/            # 游戏环境
  flappybird_env.py    - Flappy Bird (Pygame)
  dino_env.py          - Chrome 恐龙 (Pygame)
  snake_env.py         - 贪吃蛇
  gym_env.py           - Gymnasium 环境封装 (CartPole, LunarLander 等)
utils/           # 日志与可视化
train.py         # 统一训练入口
evaluate.py      # 统一评估入口
config.py        # 游戏与超参数配置
```

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
# 使用默认算法训练
python train.py --game flappybird
python train.py --game dino
python train.py --game cartpole

# 指定算法
python train.py --game flappybird --agent reinforce
python train.py --game dino --agent dqn
python train.py --game cartpole --agent a2c

# 自定义参数
python train.py --game flappybird --episodes 5000 --eval-every 100

# 从检查点恢复
python train.py --game flappybird --resume outputs/flappybird_dqn/models/best.pth
```

训练过程中每隔 N 轮自动运行评估，结果保存在 `outputs/<game>_<agent>/logs/` 下。

## 评估

```bash
python evaluate.py --game flappybird --model outputs/flappybird_dqn/models/best.pth
python evaluate.py --game dino --model outputs/dino_dueling_dqn/models/best.pth --render
python evaluate.py --game cartpole --model outputs/cartpole_dqn/models/best.pth --episodes 20
```

## 支持的游戏

| 游戏 | 状态维度 | 动作数 | 默认算法 |
|------|---------|--------|---------|
| flappybird | 7 | 2 | DQN |
| dino | 14 | 3 | Dueling DQN |
| snake | 100 | 4 | DQN |
| cartpole | 4 | 2 | DQN |
| lunarlander | 8 | 4 | DQN |
| frozenlake | 16 | 4 | Q-Learning |

## 支持的算法

| 算法 | 类型 | 适用场景 |
|------|------|---------|
| `dqn` | Value-based | 通用，离散动作空间 |
| `dueling_dqn` | Value-based | 稀疏动作场景效果更好 |
| `reinforce` | Policy-based | 简单环境，策略梯度基线 |
| `a2c` | Actor-Critic | 兼顾策略和价值估计 |
| `qlearning` | Tabular | 离散状态空间 (如 FrozenLake) |
