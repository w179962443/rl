# LunarLander-v2 Experiment

## Overview

LunarLander-v2 is a classic reinforcement learning environment where an agent must learn to land a lunar module safely on a landing pad. The agent controls the main engine and directional thrusters to navigate gravitational forces and achieve a soft landing.

## Environment Description

### Game Rules

- The lander starts at the top of the screen with random initial force
- Goal: Land safely on the landing pad between the two flags
- The landing pad is always at coordinates (0,0)
- Episode ends when:
  - The lander crashes (any body part touches ground outside the landing pad)
  - The lander goes outside the viewport
  - The lander lands successfully
  - Maximum steps reached

### State Space (8 dimensions)

The observation is an 8-dimensional continuous vector:

1. **x position**: Horizontal coordinate
2. **y position**: Vertical coordinate
3. **x velocity**: Horizontal speed
4. **y velocity**: Vertical speed
5. **angle**: Lander orientation
6. **angular velocity**: Rotation speed
7. **left leg contact**: Boolean (1.0 if leg touching ground)
8. **right leg contact**: Boolean (1.0 if leg touching ground)

### Action Space (4 discrete actions)

- **0**: Do nothing
- **1**: Fire left orientation engine
- **2**: Fire main engine (downward thrust)
- **3**: Fire right orientation engine

### Reward Structure

- **Landing on pad**: +100 to +140 points (depends on how well centered and gentle)
- **Crash**: -100 points
- **Each leg touching ground**: +10 points
- **Fire main engine**: -0.3 points per frame
- **Fire side engine**: -0.03 points per frame
- **Moving away from pad**: Small negative reward
- **Successful episode**: Total reward > 200 is considered solved

## Algorithm: Deep Q-Network (DQN)

### Why DQN?

- Continuous state space → Neural network needed
- Discrete action space → Q-learning applicable
- Well-suited for value-based learning

### Hyperparameters

```python
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ = 10
HIDDEN_SIZES = [256, 256]
```

## Training

### Command

```bash
python train.py --game lunarlander --episodes 1000 --render
```

### Expected Training Time

- ~20-40 minutes on CPU for 1000 episodes
- ~10-20 minutes on GPU

### Training Progress

- **Episodes 0-200**: Random exploration, negative rewards
- **Episodes 200-500**: Learning basic hovering and thrust control
- **Episodes 500-800**: Improving landing accuracy
- **Episodes 800-1000**: Consistent successful landings

### Expected Results

- **Target**: Average reward > 200 over 100 consecutive episodes
- **Good performance**: Average reward 220-250
- **Excellent performance**: Average reward > 250

## Testing

### Command

```bash
python test.py --game lunarlander --model models/lunarlander_best.pth --episodes 10 --render
```

### Success Criteria

- Lander touches down gently on the pad
- Both legs make contact
- Final reward > 200

## Difficulty Analysis

### Challenges

1. **Precise Control**: Requires fine-tuned thrust control
2. **Physics**: Must account for momentum and gravity
3. **Fuel Efficiency**: Using engines costs points
4. **Landing Accuracy**: Must land centered on pad

### Compared to Other Games

- **Easier than**: Pong (simpler state space, clearer rewards)
- **Harder than**: CartPole (more dimensions, more complex dynamics)
- **Similar to**: Snake (medium complexity, clear success criteria)

## Tips for Better Performance

1. **Longer Training**: 1000+ episodes often needed for consistent performance
2. **Larger Network**: Try [512, 256] hidden layers for complex policy
3. **Experience Replay**: Large memory buffer (100K) helps stabilize learning
4. **Target Network**: Update every 10 episodes to reduce oscillation
5. **Reward Shaping**: The environment already provides good shaped rewards

## Visualization

The rendered environment shows:

- Blue lander with flames when engines fire
- Yellow landing pad between flags
- Black background representing space
- Real-time physics simulation

## References

- [Gymnasium LunarLander Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- Original implementation based on Oleg Klimov's work
- Classic benchmark in RL research
