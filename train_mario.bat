@echo off
REM Quick training script for Super Mario Bros

echo ========================================
echo Training Super Mario Bros with CNN-DQN
echo World 1-1
echo ========================================

python train.py --game mario --episodes 10000 --render-every 100 --world 1 --stage 1

pause
