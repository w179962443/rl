@echo off
REM Test Super Mario Bros AI

echo ========================================
echo Testing Super Mario Bros AI
echo ========================================

python test.py --game mario --model experiments/mario/models/best_score_model.pth --episodes 5 --render --world 1 --stage 1

pause
