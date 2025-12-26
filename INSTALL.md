# 安装指南

## 完整的安装步骤

### 第 1 步: 验证 Python 版本

```bash
python --version
```

需要 Python 3.8+

### 第 2 步: 安装依赖包

#### 方法 A: 快速安装 (推荐)

```bash
pip install -r requirements.txt
```

#### 方法 B: 手动安装核心包

```bash
pip install torch
pip install gymnasium
pip install gymnasium[atari]  # 对于Pong (Atari环境)
pip install gymnasium[accept-rom-license]
pip install numpy
pip install matplotlib
pip install tqdm
```

#### 方法 C: GPU 加速 (CUDA)

如果你有 NVIDIA GPU，安装 GPU 版本的 PyTorch 会显著加快训练速度：

```bash
# PyTorch CUDA 12.1 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 11.8 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

检查 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 第 3 步: 验证安装

```bash
python validate_setup.py
```

正常输出应该显示所有项目都通过验证 ✓

## 常见安装问题

### 问题 1: pip 命令不存在

**解决方案**: 使用 `python -m pip`

```bash
python -m pip install -r requirements.txt
```

### 问题 2: PyTorch 安装失败

**解决方案**: 根据你的操作系统和 GPU 情况访问 https://pytorch.org/get-started/locally/

### 问题 3: 磁盘空间不足

**解决方案**: Atari 环境较大，可以先安装最小版本：

```bash
pip install torch gymnasium numpy matplotlib
```

先用 CartPole 和 FrozenLake 测试，后来再安装完整的 Atari 支持。

### 问题 4: 权限错误

**解决方案**: 使用用户级安装

```bash
pip install --user -r requirements.txt
```

或使用虚拟环境（推荐）

## 使用虚拟环境 (推荐)

### Windows (使用 venv)

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行项目
python train.py --game cartpole

# 退出虚拟环境
deactivate
```

### Windows (使用 Conda)

```bash
# 创建Conda环境
conda create -n rl-games python=3.10

# 激活环境
conda activate rl-games

# 安装依赖
pip install -r requirements.txt

# 运行项目
python train.py --game cartpole

# 停用环境
conda deactivate
```

### macOS/Linux

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行项目
python train.py --game cartpole

# 退出虚拟环境
deactivate
```

## 最小化安装

如果磁盘空间有限，可以先安装最小依赖：

```bash
pip install torch numpy matplotlib gymnasium
```

这样可以运行 CartPole 和 FrozenLake，Pong 需要额外的 Atari 包。

## 验证安装成功

```bash
# 方法1: 运行完整验证
python validate_setup.py

# 方法2: 快速演示
python demo.py

# 方法3: 手动检查
python -c "import torch; print('PyTorch:', torch.__version__); import gymnasium; print('Gymnasium:', gymnasium.__version__)"
```

## 升级依赖

升级到最新版本：

```bash
pip install --upgrade -r requirements.txt
```

## 卸载

完全卸载项目和相关包：

```bash
pip uninstall torch gymnasium gymnasium[atari] numpy matplotlib tqdm
```

## 故障诊断

### 测试 PyTorch

```python
import torch
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("GPU数量:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
```

### 测试 Gymnasium

```python
import gymnasium as gym
env = gym.make('CartPole-v1')
print("Gymnasium版本:", gym.__version__)
print("CartPole环境:", env)
env.close()
```

### 测试 NumPy 和 Matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt
print("NumPy版本:", np.__version__)
print("Matplotlib版本:", plt.matplotlib.__version__)
```

## 系统要求

### 最低配置

- **CPU**: 2 核心 2.0 GHz+
- **RAM**: 4 GB
- **磁盘**: 5 GB (含 Atari 环境)
- **操作系统**: Windows 10+, macOS 10.13+, Linux (Ubuntu 18.04+)

### 推荐配置

- **CPU**: 4 核心+ / **GPU**: NVIDIA GPU (CUDA 支持)
- **RAM**: 8 GB+
- **磁盘**: 10 GB+
- **操作系统**: 同上

### GPU 支持

支持以下 GPU：

- NVIDIA GPU (CUDA 11.8+)
- AMD GPU (ROCm 支持)
- MacBook Pro with Apple Silicon (MPS)

## 网络要求

首次安装需要下载包，需要网络连接。

如果在离线环境中工作，可以：

1. 在有网络的机器上下载轮文件
2. 使用 `pip install --no-index --find-links=/path/to/wheels -r requirements.txt`

## 下一步

安装完成后，查看快速开始指南：

```bash
cat QUICKSTART.md
```

或直接运行演示：

```bash
python demo.py
```

祝安装顺利！🎉
