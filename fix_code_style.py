#!/usr/bin/env python3
"""
自动修复代码风格问题的脚本
支持：flake8, PEP 8, 导入排序等
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并报告结果"""
    print(f"\n{'=' * 60}")
    print(f"🔧 {description}...")
    print(f"{'=' * 60}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✅ {description} 完成！")
        else:
            print(f"⚠️  {description} 出现警告或错误")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ {description} 失败: {e}")
        return False


def install_tools():
    """安装所需的代码格式化工具"""
    print("📦 检查并安装代码格式化工具...")
    tools = ["black", "autopep8", "isort", "flake8"]

    for tool in tools:
        try:
            __import__(tool.replace("-", "_"))
            print(f"✅ {tool} 已安装")
        except ImportError:
            print(f"📥 安装 {tool}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", tool], capture_output=True
            )


def fix_python_files():
    """修复所有Python文件"""
    py_files = list(Path(".").rglob("*.py"))

    # 排除.git和models目录
    py_files = [
        f
        for f in py_files
        if ".git" not in str(f)
        and "models" not in str(f)
        and "__pycache__" not in str(f)
    ]

    if not py_files:
        print("❌ 没有找到Python文件")
        return False

    print(f"\n📝 找到 {len(py_files)} 个Python文件")

    # 1. 使用isort整理导入
    run_command(
        f"{sys.executable} -m isort . --skip-gitignore --skip .git", "整理导入顺序"
    )

    # 2. 使用black格式化代码
    run_command(
        f"{sys.executable} -m black . --exclude '.git|models|__pycache__'",
        "格式化代码(Black)",
    )

    # 3. 使用autopep8修复PEP 8问题
    run_command(
        f"{sys.executable} -m autopep8 --in-place --aggressive --aggressive -r . --exclude=.git,models,__pycache__",
        "修复PEP 8问题",
    )


def check_style():
    """检查代码风格"""
    print(f"\n{'=' * 60}")
    print("🔍 检查代码风格...")
    print(f"{'=' * 60}")

    run_command(
        f"{sys.executable} -m flake8 . --exclude=.git,models,__pycache__ --max-line-length=120 --count",
        "Flake8检查",
    )


def main():
    """主函数"""
    print(
        """
╔═══════════════════════════════════════════════════════╗
║           自动代码风格修复工具                          ║
║     修复 flake8, PEP 8, 导入排序等代码问题            ║
╚═══════════════════════════════════════════════════════╝
    """
    )

    # 安装工具
    install_tools()

    # 修复代码
    fix_python_files()

    # 检查结果
    check_style()

    print(f"\n{'=' * 60}")
    print("✅ 代码风格修复完成！")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
