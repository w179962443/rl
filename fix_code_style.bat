@echo off
REM 自动修复代码风格 - 双击运行即可

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║           自动代码风格修复工具                          ║
echo ║     修复 flake8, PEP 8, 导入排序等代码问题            ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

echo 📦 安装必要的工具...
python -m pip install -q black autopep8 isort flake8

echo.
echo 🔧 整理导入顺序...
python -m isort . --skip-gitignore --skip .git

echo.
echo 🔧 格式化代码...
python -m black . --exclude=".git|models|__pycache__"

echo.
echo 🔧 修复PEP 8问题...
python -m autopep8 --in-place --aggressive --aggressive -r . --exclude=.git,models,__pycache__

echo.
echo 🔍 检查修复结果...
python -m flake8 . --exclude=.git,models,__pycache__ --max-line-length=120 --count

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║              ✅ 代码风格修复完成！                      ║
echo ╚═══════════════════════════════════════════════════════╝
echo.
pause
