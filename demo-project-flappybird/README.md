# Flappy Bird - 强化学习 AI

这是一个使用深度 Q 学习(DQN)算法训练的 Flappy Bird 游戏 AI 项目。AI 会通过不断训练，学会越来越好地玩 Flappy Bird 游戏。

---

## 🚀 快速开始（3 步搞定）

### 第 1 步：创建并激活环境

```powershell
conda create -n flappy-bird python=3.10 -y
conda activate flappy-bird
```

### 第 2 步：安装依赖

```powershell
pip install pygame torch numpy matplotlib
```

如果下载慢，可以使用国内镜像：

```powershell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pygame torch numpy matplotlib
```

### 第 3 步：开始训练

```powershell
python train.py
```

就这么简单！AI 会自动开始学习玩游戏。

---

## 📖 详细使用说明

### ▶️ 训练 AI

```powershell
# 确保激活了环境
conda activate flappy-bird

# 开始训练
python train.py
```

**训练过程中会：**

- ✅ 每 10 回合打印一次日志
- ✅ 每 50 回合渲染一次游戏画面
- ✅ 每 100 回合保存一次检查点
- ✅ 自动保存最佳模型到 `models/best_model.pth`
- ✅ 生成训练曲线图到 `logs/` 目录

**按 `Ctrl+C` 可随时停止训练。**

---

### 🎮 测试 AI

```powershell
# 自动测试10局
python test.py

# 测试更多局
python test.py --episodes 20

# 加速观看（提高帧率）
python test.py --fps 120
```

---

### 🕹️ 交互模式（AI 和人类切换）

```powershell
python test.py --interactive
```

**控制键：**

- `SPACE` - 跳跃（人类模式）
- `A` - 切换 AI/人类模式
- `R` - 重新开始游戏
- `ESC` - 退出

---

### 🎯 手动玩游戏

```powershell
python game.py
```

按 `SPACE` 键跳跃！看看你能得多少分。

---

## 📁 项目结构

```
demo-project-flappybird/
├── game.py              # Flappy Bird游戏环境
├── dqn_model.py         # DQN强化学习模型
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── requirements.txt     # 依赖包列表
├── README.md           # 本文件
├── models/             # 保存的模型（训练后生成）
└── logs/               # 训练日志和图表（训练后生成）
```

---

## ⚙️ 训练参数调整

如果想修改训练参数，编辑 `train.py` 文件的 `main()` 函数：

```python
trainer = Trainer(
    num_episodes=5000,    # 训练回合数（默认5000）
    max_steps=10000,      # 每回合最大步数
    render_every=50,      # 每50回合渲染一次（设为更大值可加速训练）
    save_every=100,       # 每100回合保存一次检查点
    log_every=10          # 每10回合打印日志
)
```

---

## 🔧 常见问题解决

### ❌ 问题 1: 找不到 conda 命令

**解决方法**:

1. 确保已安装 [Anaconda](https://www.anaconda.com/download) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. 重启终端或电脑
3. 检查 conda 是否在系统 PATH 中

---

### ❌ 问题 2: pip 安装依赖失败或很慢

**解决方法**: 使用国内镜像源

```powershell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pygame torch numpy matplotlib
```

或者使用阿里云镜像：

```powershell
pip install -i https://mirrors.aliyun.com/pypi/simple/ pygame torch numpy matplotlib
```

---

### ❌ 问题 3: PyTorch 安装太慢

**解决方法**: 安装 CPU 版本（更小更快，本项目不需要 GPU）

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pygame numpy matplotlib
```

---

### ❌ 问题 4: 运行 test.py 提示找不到模型文件

**错误信息**: `[错误] 未找到训练好的模型！`

**解决方法**: 必须先运行 `python train.py` 进行训练，生成模型后才能测试。

---

### ❌ 问题 5: 训练时出现 CUDA 相关错误

**解决方法**: 项目会自动检测并使用 CPU，不影响使用。如果仍有问题，确保安装的是 CPU 版本的 PyTorch。

---

## 📊 训练效果预期

典型的训练进度：

| 训练回合  | 预期表现               | 平均分数 |
| --------- | ---------------------- | -------- |
| 0-100     | 随机探索，基本撞墙     | 0-1      |
| 100-500   | 开始学习，偶尔通过管道 | 1-5      |
| 500-1000  | 能稳定通过几个管道     | 5-15     |
| 1000-3000 | 性能持续提升           | 15-50    |
| 3000+     | 达到较高水平           | 50-100+  |

**注意**: 实际效果可能因随机性而有所不同。

---

## 🎓 技术栈

- **游戏引擎**: Pygame
- **深度学习**: PyTorch
- **强化学习**: DQN (Deep Q-Network)
- **数值计算**: NumPy
- **数据可视化**: Matplotlib

---

## 🧠 DQN 算法特点

- ✅ 经验回放 (Experience Replay)
- ✅ 目标网络 (Target Network)
- ✅ Double DQN
- ✅ ε-greedy 探索策略
- ✅ 平滑 L1 损失函数

---

## 📝 许可证

MIT License

---

## 🙏 致谢

- Flappy Bird 游戏灵感来自 Dong Nguyen 的原作
- DQN 算法基于 DeepMind 的论文 "Playing Atari with Deep Reinforcement Learning"

---

## 💡 提示

- 训练时间较长，建议让程序运行几小时到一天
- 可以随时按 `Ctrl+C` 停止训练，模型会自动保存
- 想看 AI 表现，可以先训练几百回合就测试一下
- 如果分数一直很低，可以增加训练回合数或调整超参数

**祝你训练顺利！🎉**
