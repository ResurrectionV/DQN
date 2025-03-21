# DQN Training on a Custom GridWorld Environment

This repository provides an implementation of a Deep Q-Network (DQN) for a custom grid-world environment using [OpenAI Gym](https://gym.openai.com/) and [PyTorch](https://pytorch.org/). The project demonstrates how reinforcement learning techniques can be applied in a configurable environment and integrates TensorBoard for detailed logging of training metrics.

## Overview

This project includes:
- **GridWorldEnv:** A custom Gym environment with configurable grid size and blocked actions.
- **DQN Algorithm:** An implementation using an MLP Q-Network with experience replay and soft target updates.
- **TensorBoard Logging:** Tracks key training metrics:
  - **Loss per optimization step**
  - **Cumulative reward per episode**
  - **Episode duration (time per episode)**
- **Visualization:** A generated figure (`training_performance.png`) displays episode durations over training.

## Features

- **Customizable Environment:**  
  Modify grid size, discount factor, and other hyperparameters via command-line arguments.
  
- **DQN with Experience Replay:**  
  Uses a replay buffer and soft target network updates to stabilize training.
  
- **TensorBoard Integration:**  
  Log and visualize:
  - **Loss:** The loss computed at each optimization step.
  - **Reward:** The cumulative reward per episode.
  - **Time per Episode:** The wall-clock time for each episode.
  
- **Performance Visualization:**  
  Generates a plot (`training_performance.png`) of episode durations.

## Requirements

- Python 3.6+
- [PyTorch](https://pytorch.org/)
- [Gym](https://github.com/openai/gym)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard) (install with `pip install tensorboard`)
- [tqdm](https://github.com/tqdm/tqdm)

Install the required packages with:

```bash
pip install torch gym numpy matplotlib tensorboard tqdm
```

## Usage

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/dqn-gridworld.git
cd dqn-gridworld
```

Run the training script with default parameters:
```bash
python dqn_training.py
```

### Command-Line Arguments

- `--grid_size`: Size of the grid environment (default: 4)
- `--gamma`: Discount factor (default: 0.9)
- `--episodes`: Number of training episodes (default: 20)
- `--batch_size`: Batch size for optimization (default: 32)
- `--memory_capacity`: Capacity of the replay buffer (default: 10000)
- `--hidden_size`: Hidden layer size in the Q-network (default: 128)
- `--learning_rate`: Learning rate for AdamW optimizer (default: 1e-4)
- `--max_steps`: Maximum steps per episode (default: 2000)
- `--tau`: Soft update coefficient for the target network (default: 0.005)
- `--output_dir`: Directory to save the training performance figure (default: current directory)
- `--log_dir`: Directory to store TensorBoard logs (default: `./runs`)

### Monitoring Training with TensorBoard

After starting the training script, launch TensorBoard to monitor training metrics:

```bash
tensorboard --logdir=./runs
```

