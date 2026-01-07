@echo off
chcp 65001 >nul
title Flappy Bird AI - 启动脚本
color 0A

:MENU
cls
echo ╔════════════════════════════════════════════════════════════════╗
echo ║           Flappy Bird - 强化学习AI 启动脚本                   ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo  [1] 安装环境
echo  [2] 训练AI
echo  [3] 测试AI（自动模式）
echo  [4] 测试AI（交互模式）
echo  [5] 手动玩游戏
echo  [0] 退出
echo.
echo ================================================================
set /p choice="请选择操作 (0-5): "

if "%choice%"=="1" goto INSTALL
if "%choice%"=="2" goto TRAIN
if "%choice%"=="3" goto TEST
if "%choice%"=="4" goto TEST_INTERACTIVE
if "%choice%"=="5" goto PLAY
if "%choice%"=="0" goto EXIT
goto MENU

:INSTALL
cls
echo ════════════════════════════════════════════════════════════════
echo  正在检查和安装环境...
echo ════════════════════════════════════════════════════════════════
echo.

REM 检查conda是否安装
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Conda！
    echo.
    echo 请先安装 Anaconda 或 Miniconda:
    echo https://www.anaconda.com/download
    echo 或
    echo https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    goto MENU
)

echo [✓] Conda 已安装
echo.

REM 检查flappy-bird环境是否存在
conda env list | findstr "flappy-bird" >nul 2>nul
if %errorlevel% equ 0 (
    echo [!] 检测到 flappy-bird 环境已存在
    echo.
    set /p recreate="是否要删除并重新创建? (y/n): "
    if /i "!recreate!"=="y" (
        echo.
        echo [→] 正在删除旧环境...
        call conda deactivate 2>nul
        call conda env remove -n flappy-bird -y
        echo [✓] 旧环境已删除
    ) else (
        echo.
        echo [→] 跳过环境创建，准备安装依赖包...
        goto INSTALL_PACKAGES
    )
)

echo.
echo [→] 正在创建 flappy-bird conda 环境 (Python 3.10)...
call conda create -n flappy-bird python=3.10 -y
if %errorlevel% neq 0 (
    echo [错误] 环境创建失败！
    pause
    goto MENU
)
echo [✓] 环境创建成功

:INSTALL_PACKAGES
echo.
echo [→] 正在激活环境...
call conda activate flappy-bird
if %errorlevel% neq 0 (
    echo [错误] 环境激活失败！
    pause
    goto MENU
)
echo [✓] 环境已激活

echo.
echo [→] 正在安装依赖包...
echo.
pip install pygame torch numpy matplotlib
if %errorlevel% neq 0 (
    echo.
    echo [错误] 依赖包安装失败！
    pause
    goto MENU
)

echo.
echo ════════════════════════════════════════════════════════════════
echo [✓] 环境安装完成！
echo ════════════════════════════════════════════════════════════════
echo.
echo 环境名称: flappy-bird
echo Python版本: 3.10
echo 已安装: pygame, torch, numpy, matplotlib
echo.
pause
goto MENU

:TRAIN
cls
echo ════════════════════════════════════════════════════════════════
echo  启动训练...
echo ════════════════════════════════════════════════════════════════
echo.

REM 检查conda环境
conda env list | findstr "flappy-bird" >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 flappy-bird 环境！
    echo 请先运行 [1] 安装环境
    echo.
    pause
    goto MENU
)

echo [→] 正在激活 flappy-bird 环境...
call conda activate flappy-bird

echo [→] 正在启动训练...
echo.
echo ----------------------------------------------------------------
python train.py
echo ----------------------------------------------------------------
echo.
echo 训练结束！
echo.
pause
goto MENU

:TEST
cls
echo ════════════════════════════════════════════════════════════════
echo  启动测试（自动模式）...
echo ════════════════════════════════════════════════════════════════
echo.

REM 检查conda环境
conda env list | findstr "flappy-bird" >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 flappy-bird 环境！
    echo 请先运行 [1] 安装环境
    echo.
    pause
    goto MENU
)

REM 检查模型文件
if not exist "models\best_model.pth" (
    echo [错误] 未找到训练好的模型！
    echo 请先运行 [2] 训练AI
    echo.
    pause
    goto MENU
)

echo [→] 正在激活 flappy-bird 环境...
call conda activate flappy-bird

echo [→] 正在加载模型并测试...
echo.
echo ----------------------------------------------------------------
python test.py --episodes 10
echo ----------------------------------------------------------------
echo.
echo 测试结束！
echo.
pause
goto MENU

:TEST_INTERACTIVE
cls
echo ════════════════════════════════════════════════════════════════
echo  启动测试（交互模式）...
echo ════════════════════════════════════════════════════════════════
echo.

REM 检查conda环境
conda env list | findstr "flappy-bird" >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 flappy-bird 环境！
    echo 请先运行 [1] 安装环境
    echo.
    pause
    goto MENU
)

REM 检查模型文件
if not exist "models\best_model.pth" (
    echo [错误] 未找到训练好的模型！
    echo 请先运行 [2] 训练AI
    echo.
    pause
    goto MENU
)

echo [→] 正在激活 flappy-bird 环境...
call conda activate flappy-bird

echo [→] 正在启动交互模式...
echo.
echo 控制说明:
echo   SPACE - 跳跃（人类模式）
echo   A     - 切换 AI/人类 模式
echo   R     - 重新开始
echo   ESC   - 退出
echo.
echo ----------------------------------------------------------------
python test.py --interactive
echo ----------------------------------------------------------------
echo.
pause
goto MENU

:PLAY
cls
echo ════════════════════════════════════════════════════════════════
echo  手动玩游戏...
echo ════════════════════════════════════════════════════════════════
echo.

REM 检查conda环境
conda env list | findstr "flappy-bird" >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未找到 flappy-bird 环境！
    echo 请先运行 [1] 安装环境
    echo.
    pause
    goto MENU
)

echo [→] 正在激活 flappy-bird 环境...
call conda activate flappy-bird

echo [→] 正在启动游戏...
echo.
echo 按 SPACE 键跳跃，关闭窗口退出
echo.
echo ----------------------------------------------------------------
python game.py
echo ----------------------------------------------------------------
echo.
pause
goto MENU

:EXIT
cls
echo.
echo 感谢使用 Flappy Bird AI！
echo.
timeout /t 2 >nul
exit
