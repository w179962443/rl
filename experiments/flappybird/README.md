# Flappy Bird Reinforcement Learning Experiment

This directory contains the Flappy Bird game trained using Deep Q-Network (DQN).

## Structure

- `train.py`: Training script for Flappy Bird
- `test.py`: Testing script for trained models
- `models/`: Directory for saved models
- `logs/`: Directory for training logs and plots

## Features

- **Training with visualization**: Renders game every N episodes
- **Automatic checkpointing**: Saves model every N episodes
- **Training history**: Saves and loads training progress
- **Plot generation**: Generates training curves (score, loss, epsilon)
- **Best model tracking**: Automatically saves the best performing model

## Usage

Training is integrated into the main training script. Use:

```bash
python train.py --game flappybird --episodes 1000
```

Testing:

```bash
python test.py --game flappybird --model experiments/flappybird/models/best_model.pth --episodes 10
```
