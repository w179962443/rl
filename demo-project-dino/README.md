# Chrome Dinosaur — 强化学习项目

> 用 **Double Dueling DQN** 训练一个能玩 Chrome 恐龙跑酷游戏的 AI

---

## 游戏说明

- 恐龙自动向右奔跑，速度随分数线性增加
- 障碍物有两种：**仙人掌**（地面）和**翼龙**（空中）
- 动作空间（3 个离散动作）：

  | 动作 | 编号 | 说明                  |
  | ---- | ---- | --------------------- |
  | 不动 | 0    | 继续奔跑              |
  | 跳跃 | 1    | 跳过仙人掌 / 高空翼龙 |
  | 俯身 | 2    | 低头躲过低空翼龙      |

- 得分越高越好，最高速 18 px/frame

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 手动游玩
python test.py --manual
#   空格 / ↑ 键 : 跳跃
#   ↓ 键       : 俯身
#   ESC / Q    : 退出

# 开始训练（约 5000 episode 可明显学会跳跃，10000+ 可稳定躲翼龙）
python train.py

# 查看 AI 演示（需先训练）
python test.py

# 继续已有训练
python train.py --resume models/best_model.pth

# 批量测试（无界面，输出统计）
python test.py --episodes 50 --no_render

# 一键菜单（Windows）
start.bat
```

---

## 项目文件

```
demo-project-dino/
├── game.py          # 游戏核心逻辑（Pygame）
├── dqn_model.py     # Double Dueling DQN 实现
├── train.py         # 训练脚本
├── test.py          # 测试 / 演示脚本
├── start.bat        # Windows 一键启动菜单
├── requirements.txt
├── models/          # 训练后生成（保存模型）
└── logs/            # 训练后生成（训练曲线图）
```

---

## 算法设计

### 状态空间（14 维）

| 维度 | 含义                                                   |
| ---- | ------------------------------------------------------ |
| 0    | 恐龙 Y 坐标（归一化）                                  |
| 1    | 恐龙垂直速度（归一化）                                 |
| 2    | 是否俯身                                               |
| 3    | 当前速度（归一化）                                     |
| 4~8  | 最近障碍物 1：距离、宽、高、Y坐标、类型(0仙人掌/1翼龙) |
| 9~13 | 最近障碍物 2（同上）                                   |

### 奖励函数

| 情况       | 奖励  |
| ---------- | ----- |
| 每存活一步 | +0.1  |
| 碰撞死亡   | -10.0 |

### 网络结构

```
输入(14) → FC(256) → ReLU → FC(256) → ReLU
                                         ├─ Value Stream  → FC(128) → V(s)
                                         └─ Adv Stream   → FC(128) → A(s,a)
                          Q(s,a) = V(s) + A(s,a) - mean(A)
```

### 关键技术

- **Double DQN**：用 policy_net 选动作，target_net 估 Q 值，避免高估
- **Dueling DQN**：分离状态价值 V 与优势函数 A，提升无动作时的学习效率
- **Experience Replay**：100k 经验池，batch=64
- **ε-贪心衰减**：1.0 → 0.01，衰减系数 0.9995/step

---

## 训练进度参考

| 阶段     | Episode 范围 | 行为               |
| -------- | ------------ | ------------------ |
| 探索期   | 0~1000       | 随机跳跃，频繁死亡 |
| 学习跳跃 | 1000~3000    | 开始躲避仙人掌     |
| 稳定跑动 | 3000~6000    | 能持续跑较长距离   |
| 应对翼龙 | 6000+        | 学会俯身躲翼龙     |

---

## 训练参数

```bash
python train.py --help

  --episodes     训练总回合数（默认 8000）
  --hidden_size  神经网络隐藏层宽度（默认 256）
  --lr           学习率（默认 3e-4）
  --render_every 每隔多少 episode 渲染一次（默认 500）
  --save_every   每隔多少 episode 保存一次（默认 200）
  --resume       从指定 .pth 文件继续训练
  --model_dir    模型保存目录（默认 models/）
  --log_dir      日志 / 图表目录（默认 logs/）
```
