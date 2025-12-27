#!/usr/bin/env python3
"""快速修复代码风格 - 只修复，不检查"""

import subprocess
import sys

tools = ["black", "autopep8", "isort", "flake8"]

print("📦 安装工具...")
for tool in tools:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", tool], capture_output=True
    )

print("\n🔧 修复代码...")
subprocess.run(
    [sys.executable, "-m", "isort", ".", "--skip-gitignore"], capture_output=True
)
subprocess.run(
    [sys.executable, "-m", "black", ".", "--exclude='.git|models'"], capture_output=True
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "autopep8",
        "--in-place",
        "-r",
        ".",
        "--exclude=.git,models",
    ],
    capture_output=True,
)

print("✅ 完成！")
