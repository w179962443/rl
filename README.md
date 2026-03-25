# RL Games

用强化学习算法玩各种游戏。

## 项目结构

```
agents/          # RL 算法实现
  dqn_agent.py         - Double DQN
  dueling_dqn_agent.py - Double Dueling DQN
  reinforce_agent.py   - REINFORCE (策略梯度)
  a2c_agent.py         - Advantage Actor-Critic
  ppo_agent.py         - Proximal Policy Optimization
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

## 训练 & 评估

训练结果保存在 `outputs/<game>_<agent>/`，模型在 `models/`，日志在 `logs/`。

### Flappy Bird

```bash
python train.py --game flappybird
python train.py --game flappybird --agent ppo
python train.py --game flappybird --agent reinforce
python train.py --game flappybird --agent a2c

python evaluate.py --game flappybird --model outputs/flappybird_dqn/models/best.pth
python evaluate.py --game flappybird --agent ppo --model outputs/flappybird_ppo/models/best.pth --render
```

### Dino

```bash
python train.py --game dino
python train.py --game dino --agent ppo
python train.py --game dino --agent dqn

python evaluate.py --game dino --model outputs/dino_dueling_dqn/models/best.pth
python evaluate.py --game dino --agent ppo --model outputs/dino_ppo/models/best.pth --render
```

### Snake

```bash
python train.py --game snake
python train.py --game snake --agent ppo
python train.py --game snake --agent dueling_dqn

python evaluate.py --game snake --model outputs/snake_dqn/models/best.pth
python evaluate.py --game snake --agent ppo --model outputs/snake_ppo/models/best.pth --render
```

### CartPole

```bash
python train.py --game cartpole
python train.py --game cartpole --agent ppo
python train.py --game cartpole --agent a2c

python evaluate.py --game cartpole --model outputs/cartpole_dqn/models/best.pth
python evaluate.py --game cartpole --agent ppo --model outputs/cartpole_ppo/models/best.pth --render
```

### LunarLander

```bash
python train.py --game lunarlander
python train.py --game lunarlander --agent ppo
python train.py --game lunarlander --agent dueling_dqn

python evaluate.py --game lunarlander --model outputs/lunarlander_dqn/models/best.pth
python evaluate.py --game lunarlander --agent ppo --model outputs/lunarlander_ppo/models/best.pth --render
```

### FrozenLake

```bash
python train.py --game frozenlake
python train.py --game frozenlake --agent ppo
python train.py --game frozenlake --agent dqn

python evaluate.py --game frozenlake --model outputs/frozenlake_qlearning/models/best.pth
python evaluate.py --game frozenlake --agent ppo --model outputs/frozenlake_ppo/models/best.pth --render
```

### 通用参数

```bash
# 自定义训练轮数和评估频率
python train.py --game cartpole --episodes 1000 --eval-every 50

# 从检查点恢复训练
python train.py --game flappybird --resume outputs/flappybird_dqn/models/best.pth

# 评估多轮
python evaluate.py --game lunarlander --model outputs/lunarlander_ppo/models/best.pth --episodes 20
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

| 算法 | 类型 | 特点 |
|------|------|------|
| `dqn` | Value-based | Double DQN + 经验回放，通用离散动作 |
| `dueling_dqn` | Value-based | Dueling 架构，稀疏奖励场景更优 |
| `reinforce` | Policy-based | Monte-Carlo 策略梯度，带基线 |
| `a2c` | Actor-Critic | 在线 n-step 更新，Actor-Critic 共享网络 |
| `ppo` | Actor-Critic | 裁剪代理目标 + GAE，样本效率高，训练稳定 |
| `qlearning` | Tabular | 表格型 Q-Learning，适合离散状态空间 |

## PPO 算法说明

PPO（Proximal Policy Optimization）是目前最常用的策略梯度算法之一，核心特点：

- **裁剪代理目标**：限制每次更新的策略变化幅度（`clip_ratio=0.2`），避免过大更新导致训练崩溃
- **GAE 优势估计**：Generalized Advantage Estimation 平衡偏差与方差（`gae_lambda=0.95`）
- **多轮 mini-batch 更新**：每个 episode 的轨迹数据复用多次（`n_epochs=4`），提升样本效率
- **熵正则化**：鼓励探索，防止策略过早收敛

主要超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `clip_ratio` | 0.2 | 策略更新裁剪范围 |
| `gae_lambda` | 0.95 | GAE 平滑系数 |
| `n_epochs` | 4 | 每个 episode 的更新轮数 |
| `entropy_coef` | 0.01 | 熵奖励系数 |
| `value_coef` | 0.5 | 价值损失权重 |
