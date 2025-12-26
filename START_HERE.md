# 项目使用指导

## 🎯 从这里开始

### 第一步：阅读合适的文档（5-10 分钟）

根据你的情况选择：

| 你的情况               | 读这个                             |
| ---------------------- | ---------------------------------- |
| 我是初学者，想快速上手 | [QUICKSTART.md](QUICKSTART.md)     |
| 我想知道如何安装       | [INSTALL.md](INSTALL.md)           |
| 我想看具体例子         | [EXAMPLES.md](EXAMPLES.md)         |
| 我想深入了解代码       | [DEVELOPMENT.md](DEVELOPMENT.md)   |
| 我说中文，想快速了解   | [README_CN.md](README_CN.md)       |
| 我想知道项目包含什么   | [项目完成总结.md](项目完成总结.md) |
| 我想看完整说明         | [README.md](README.md)             |

### 第二步：验证环境（2 分钟）

```bash
# 进入项目目录
cd demo-project-rl

# 验证一切是否就绪
python validate_setup.py
```

**预期输出**: 显示所有检查通过 ✓

### 第三步：快速体验（1 分钟）

```bash
# 运行演示脚本 - 会在CartPole上训练50个episode
python demo.py
```

**你将看到**:

- 前 10 个 episode 的进度
- 训练过程中奖励的增长
- 最终的平均奖励

### 第四步：开始你的第一次训练（5-10 分钟）

```bash
# CartPole - 最简单的环境，推荐首先尝试
python train.py --game cartpole --episodes 500

# 你将看到:
# - 实时进度显示
# - 每10个episode的统计信息
# - 最后生成的图表保存到 results/plots/
# - 模型保存到 models/
```

### 第五步：查看结果（2 分钟）

```bash
# 分析训练结果
python evaluate.py --analyze results/logs/cartpole_500ep.json

# 你将看到:
# - 训练统计信息
# - 平均奖励
# - 最后100个episode的性能
```

---

## 📚 按任务查找文档

### 🔵 安装和设置

**我如何安装这个项目？**
→ 参考 [INSTALL.md](INSTALL.md)

```bash
pip install -r requirements.txt
python validate_setup.py
```

**我遇到安装错误**
→ 查看 [INSTALL.md](INSTALL.md#常见安装问题) 的故障排除部分

**我如何使用 GPU 加速？**
→ [INSTALL.md](INSTALL.md#方法C-GPU加速-CUDA)

---

### 🔵 快速开始

**我只想快速体验一下**
→ [QUICKSTART.md](QUICKSTART.md) 或者：

```bash
python demo.py  # 1分钟快速演示
```

**我想立即开始训练**
→ [QUICKSTART.md#2-运行第一个实验---cartpole](QUICKSTART.md)

```bash
python train.py --game cartpole --episodes 500
```

**我想了解完整的工作流程**
→ [DEVELOPMENT.md#工作流程](DEVELOPMENT.md)

---

### 🔵 具体示例

**我想看 10 个详细的使用示例**
→ [EXAMPLES.md](EXAMPLES.md)

**我想学习如何调整超参数**
→ [EXAMPLES.md#例子-5-超参数调优](EXAMPLES.md)

**我想对比不同的实验**
→ [EXAMPLES.md#例子-6-比较不同设置](EXAMPLES.md)

**我想自定义训练循环**
→ [EXAMPLES.md#例子-8-自定义训练循环](EXAMPLES.md)

---

### 🔵 三个游戏说明

**CartPole - 倒立摆**

- 文件：[experiments/cartpole/README.md](experiments/cartpole/README.md)
- 难度：⭐ 简单
- 训练时间：5-10 分钟
- 算法：DQN
- 推荐：首先学习！

**FrozenLake - 冰湖**

- 文件：[experiments/frozenlake/README.md](experiments/frozenlake/README.md)
- 难度：⭐⭐ 中等
- 训练时间：1-2 分钟
- 算法：Q-Learning
- 推荐：学习表格方法

**Pong - 乒乓球**

- 文件：[experiments/pong/README.md](experiments/pong/README.md)
- 难度：⭐⭐⭐ 困难
- 训练时间：2-4 小时
- 算法：DQN
- 推荐：有经验后尝试

---

### 🔵 开发和扩展

**我想添加新的算法**
→ [DEVELOPMENT.md#扩展指南](DEVELOPMENT.md)

**我想添加新的游戏环境**
→ [DEVELOPMENT.md#添加新游戏环境](DEVELOPMENT.md)

**我想改进神经网络**
→ [DEVELOPMENT.md#改进神经网络](DEVELOPMENT.md)

**我想理解代码的设计**
→ [DEVELOPMENT.md#代码结构](DEVELOPMENT.md)

---

### 🔵 问题解决

**我的模型训练太慢**
→ [QUICKSTART.md#常见问题](QUICKSTART.md)

**内存不足怎么办**
→ [QUICKSTART.md#常见问题](QUICKSTART.md)

**模型性能不好**
→ [QUICKSTART.md#常见问题](QUICKSTART.md)

**导入错误或其他错误**
→ [EXAMPLES.md#故障排除](EXAMPLES.md)

---

## 📖 标准使用流程

### 流程 1：学习强化学习

```
1. 阅读 QUICKSTART.md (5分钟)
   ↓
2. 运行 demo.py (1分钟)
   ↓
3. 运行 train.py --game cartpole --episodes 100 (2分钟快速测试)
   ↓
4. 理解代码：agents/dqn_agent.py, train.py
   ↓
5. 运行完整训练：python train.py --game cartpole --episodes 500 (10分钟)
   ↓
6. 分析结果：python evaluate.py --analyze results/logs/*.json
   ↓
7. 观察可视化图表在 results/plots/ 中
```

### 流程 2：进行研究实验

```
1. 理解所有游戏：阅读 experiments/ 下的README
   ↓
2. 制定实验计划
   ↓
3. 修改 train.py 中的超参数
   ↓
4. 运行多个实验：python train.py --game X --episodes N
   ↓
5. 比较结果：python evaluate.py --all
   ↓
6. 分析和可视化结果
   ↓
7. 得出结论
```

### 流程 3：扩展功能

```
1. 阅读 DEVELOPMENT.md
   ↓
2. 分析现有代码结构
   ↓
3. 实现新功能（新算法/新环境）
   ↓
4. 在 train.py/test.py 中集成
   ↓
5. 进行完整测试
   ↓
6. 记录文档
```

---

## 🎯 常见任务快速参考

### 任务：快速演示

```bash
python demo.py
```

**预期时间**：1 分钟

### 任务：训练第一个模型

```bash
python train.py --game cartpole --episodes 500
```

**预期时间**：5-10 分钟  
**最终输出**：

- `models/cartpole_best.pth` - 最好的模型
- `results/logs/cartpole_500ep.json` - 训练日志
- `results/plots/cartpole_training.png` - 训练曲线

### 任务：测试模型

```bash
python test.py --game cartpole --model models/cartpole_best.pth --episodes 20
```

**预期时间**：30 秒

### 任务：查看训练结果

```bash
python evaluate.py --analyze results/logs/cartpole_500ep.json
```

**预期时间**：1 分钟

### 任务：对比多个实验

```bash
python evaluate.py --compare \
    results/logs/exp1.json \
    results/logs/exp2.json \
    results/logs/exp3.json
```

**预期时间**：2 分钟

### 任务：可视化 FrozenLake 策略

```bash
python train.py --game frozenlake --episodes 10000
python visualize_frozenlake.py
```

**预期时间**：2 分钟 (训练) + 30 秒 (可视化)

### 任务：运行所有实验

```bash
python run_experiments.py --game all
```

**预期时间**：几小时 (取决于 GPU)

---

## 💡 学习路径建议

### 完全初学者 (推荐 4 小时)

```
1. 阅读 QUICKSTART.md (15分钟)
2. 运行 demo.py (1分钟)
3. 运行 train.py --game cartpole --episodes 500 (10分钟)
4. 理解DQN算法：阅读 DEVELOPMENT.md (30分钟)
5. 查看源代码：agents/dqn_agent.py (30分钟)
6. 运行 train.py --game frozenlake --episodes 10000 (2分钟)
7. 理解Q-Learning：阅读 agents/qlearning_agent.py (30分钟)
8. 观察和分析结果 (1小时)
```

### 有编程基础的用户 (推荐 2 小时)

```
1. 快速阅读 QUICKSTART.md (5分钟)
2. 运行 demo.py (1分钟)
3. 阅读 EXAMPLES.md (30分钟)
4. 运行几个实验 (20分钟)
5. 深入理解代码 (30分钟)
6. 尝试修改超参数 (20分钟)
```

### 有 RL 经验的用户 (推荐 30 分钟)

```
1. 浏览 README.md (5分钟)
2. 查看代码结构 (10分钟)
3. 运行实验，对比结果 (15分钟)
```

---

## 📞 快速帮助

**问：我应该从哪里开始？**  
答：查看 [QUICKSTART.md](QUICKSTART.md) 或运行 `python demo.py`

**问：如何安装？**  
答：`pip install -r requirements.txt`，详见 [INSTALL.md](INSTALL.md)

**问：如何训练？**  
答：`python train.py --game cartpole --episodes 500`

**问：如何测试模型？**  
答：`python test.py --game cartpole --model models/cartpole_best.pth`

**问：如何查看结果？**  
答：`python evaluate.py --all`，图表保存在 `results/plots/`

**问：我遇到了问题**  
答：查看 [EXAMPLES.md#故障排除](EXAMPLES.md) 或 [QUICKSTART.md#常见问题](QUICKSTART.md)

---

## ✅ 验证清单

在开始之前，确保：

- [ ] Python 版本 >= 3.8
- [ ] 已安装依赖：`pip install -r requirements.txt`
- [ ] 验证通过：`python validate_setup.py` 显示 ✓
- [ ] 演示成功：`python demo.py` 运行完成

---

## 🎓 推荐学习资源

### 强化学习理论

- 📖 Sutton & Barto 《强化学习导论》
- 🎥 David Silver UCL 强化学习课程

### 深度学习工具

- 📚 PyTorch 官方教程：https://pytorch.org/tutorials/
- 📄 PyTorch 文档：https://pytorch.org/docs/

### RL 环境

- 🏠 Gymnasium 官网：https://gymnasium.farama.org/
- 📖 Gymnasium 文档

### 经典论文

- DQN: "Human-level control through deep reinforcement learning"
- A3C: "Asynchronous Methods for Deep RL"
- PPO: "Proximal Policy Optimization"

---

## 🚀 下一步行动

### 现在就开始（5 分钟）

```bash
python validate_setup.py  # 验证环境
python demo.py           # 运行演示
```

### 进行第一个实验（15 分钟）

```bash
python train.py --game cartpole --episodes 500
python evaluate.py --analyze results/logs/cartpole_500ep.json
```

### 深入学习（1 小时+）

1. 阅读 [EXAMPLES.md](EXAMPLES.md) 的详细示例
2. 修改超参数进行实验
3. 阅读源代码理解实现
4. 尝试新的想法

---

**现在就开始吧！** 🚀

有任何问题，查看合适的文档或运行：

```bash
python validate_setup.py  # 诊断问题
```

祝你学习愉快！🎉
