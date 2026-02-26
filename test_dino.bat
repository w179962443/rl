@echo off
chcp 65001 > nul
title Chrome Dino — AI Test / Demo

echo ╔══════════════════════════════════════════╗
echo ║    Chrome Dinosaur — AI 演示 / 测试       ║
echo ╚══════════════════════════════════════════╝
echo.
echo 用法示例：
echo   test_dino.bat                  — 加载最佳模型演示 10 回合
echo   test_dino.bat --episodes 30    — 演示 30 回合
echo   test_dino.bat --manual         — 手动游玩
echo   test_dino.bat --no_render      — 批量无界面测试
echo.

cd demo-project-dino

if "%1"=="--manual" (
    python test.py --manual
) else (
    python test.py %*
)

cd ..
pause
