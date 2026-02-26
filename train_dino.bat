@echo off
chcp 65001 > nul
title Chrome Dino — RL Training

echo ╔══════════════════════════════════════════╗
echo ║    Chrome Dinosaur — 开始 DQN 训练        ║
echo ╚══════════════════════════════════════════╝
echo.
echo 训练说明：
echo   - 约 5000 episode 后 AI 开始稳定躲仙人掌
echo   - 约 10000 episode 后开始应对翼龙
echo   - 模型保存在 demo-project-dino/models/
echo   - 训练曲线保存在 demo-project-dino/logs/
echo.

cd demo-project-dino
python train.py %*
cd ..
pause
