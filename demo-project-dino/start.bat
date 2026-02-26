@echo off
chcp 65001 > nul
title Chrome Dino RL — 启动菜单

:menu
cls
echo ╔══════════════════════════════════════════╗
echo ║    Chrome Dinosaur Reinforcement Learning  ║
echo ╠══════════════════════════════════════════╣
echo ║  1. 手动游玩（键盘控制）                   ║
echo ║  2. 开始 AI 训练（从头）                  ║
echo ║  3. 继续训练（加载最佳模型）               ║
echo ║  4. AI 演示（加载已训练模型）              ║
echo ║  5. 批量测试（无界面统计）                 ║
echo ║  0. 退出                                  ║
echo ╚══════════════════════════════════════════╝
set /p choice=请输入选项: 

if "%choice%"=="1" goto manual
if "%choice%"=="2" goto train_new
if "%choice%"=="3" goto train_resume
if "%choice%"=="4" goto demo
if "%choice%"=="5" goto batch_test
if "%choice%"=="0" exit
goto menu

:manual
echo.
echo [手动游玩] 空格/↑ 跳跃，↓ 俯身，ESC 退出
python test.py --manual
pause
goto menu

:train_new
echo.
echo [训练] 从头开始训练...
python train.py
pause
goto menu

:train_resume
echo.
echo [继续训练] 加载 models/best_model.pth ...
python train.py --resume models/best_model.pth
pause
goto menu

:demo
echo.
echo [AI 演示] 加载最佳模型进行演示...
python test.py --model models/best_model.pth --episodes 10 --slow
pause
goto menu

:batch_test
echo.
echo [批量测试] 运行 50 回合统计...
python test.py --model models/best_model.pth --episodes 50 --no_render
pause
goto menu
