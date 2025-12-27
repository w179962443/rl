@echo off
REM 快速修复代码风格 - 不检查，更快

echo.
echo ╔═══════════════════════════════════════════════════════╗
echo ║              快速代码风格修复                          ║
echo ╚═══════════════════════════════════════════════════════╝
echo.

python -m pip install -q black autopep8 isort flake8 >nul 2>&1

echo 🔧 处理中...
python -m isort . --skip-gitignore >nul 2>&1
python -m black . --exclude=".git|models" >nul 2>&1
python -m autopep8 --in-place -r . --exclude=.git,models >nul 2>&1

echo.
echo ✅ 快速修复完成！
echo.
pause
