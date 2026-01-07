@echo off
REM Test Flappy Bird AI

echo ========================================
echo Testing Flappy Bird AI
echo ========================================

python test.py --game flappybird --model experiments/flappybird/models/best_model.pth --episodes 5 --render

pause
