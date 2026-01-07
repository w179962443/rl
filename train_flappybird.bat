@echo off
REM Quick training script for Flappy Bird

echo ========================================
echo Training Flappy Bird with DQN
echo ========================================

python train.py --game flappybird --episodes 1000 --render-every 100

pause
