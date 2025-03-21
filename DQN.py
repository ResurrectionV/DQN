#!/usr/bin/env python3
"""
dqn_training.py

A DQN implementation on a custom grid-world Gym environment.
Use command-line arguments to control hyperparameters such as grid size,
discount factor, number of episodes, and so on.

Usage:
    python dqn_training.py --grid_size 4 --episodes 20 --max_steps 2000 --output_dir . --log_dir ./runs
"""

import argparse
import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import math
from itertools import count
import os
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter

# Custom Gym Environment renamed to GridWorldEnv
class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=4, gamma=0.9):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.gamma = gamma
        self.action_space = gym.spaces.Discrete(4)  # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([grid_size - 1, grid_size - 1]),
            dtype=np.intc
        )
        # Define blocked actions only for grid_size==4; otherwise, none are blocked.
        if grid_size == 4:
            self.blocked_actions = {
                (0, 0): [1],  # From (0, 0), going down is blocked.
                (0, 1): [1],
                (0, 2): [1],
                (1, 0): [0],  # From (1, 0), going up is blocked.
                (1, 1): [0],
                (1, 2): [0],
            }
        else:
            self.blocked_actions = {}
        self.state = None

    def reset(self):
        # Agent starts at the bottom-left corner (grid_size-1, 0)
        self.state = np.array([self.grid_size - 1, 0])
        return self.state

    def step(self, action):
        new_state = np.array(self.state, copy=True)
        if action == 0:  # Up
            new_state[0] -= 1
        elif action == 1:  # Down
            new_state[0] += 1
        elif action == 2:  # Left
            new_state[1] -= 1
        elif action == 3:  # Right
            new_state[1] += 1

        # Verify within boundaries and not blocked by environment rules.
        if (0 <= new_state[0] < self.grid_size and 0 <= new_state[1] < self.grid_size and 
            (action not in self.blocked_actions.get(tuple(self.state), []))):
            self.state = new_state

        # Reward of 10 only if the agent reaches the top-left corner.
        reward = 10 if (self.state[0] == 0 and self.state[1] == 0) else 0
        done = (self.state[0] == 0 and self.state[1] == 0)
        return self.state, reward, done

    def step2(self, action, state):
        """
        Alternate step function that takes an external state input.
        (Retained from your original code for completeness.)
        """
        new_state = np.array(state, dtype=np.intc, copy=True)
        if action == 0:
            new_state[0] -= 1
        elif action == 1:
            new_state[0] += 1
        elif action == 2:
            new_state[1] -= 1
        elif action == 3:
            new_state[1] += 1

        if (0 <= new_state[0] < self.grid_size and 0 <= new_state[1] < self.grid_size and
            (action not in self.blocked_actions.get(tuple(state), []))):
            reward = 10 if (new_state[0] == 0 and new_state[1] == 0) else 0
            done = (new_state[0] == 0 and new_state[1] == 0)
            return new_state, reward, done
        else:
            reward = 10 if (state[0] == 0 and state[1] == 0) else 0
            done = (state[0] == 0 and state[1] == 0)
            return np.array(state, dtype=np.intc), reward, done

# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def reset(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

# Q-Network (A simple Multi-Layer Perceptron)
class Q_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Epsilon-Greedy Action Selection
def select_action(state, policy_net, steps_done, n_actions):
    eps_threshold = 0.005 + 0.1 * math.exp(-1. * steps_done / 200)
    if random.random() > eps_threshold:
        with torch.no_grad():
            # Add batch dimension for network input.
            q_values = policy_net(state.unsqueeze(0))
            action = q_values.argmax(dim=1).item()
            return action
    else:
        return random.randrange(n_actions)

# Optimize the model using a batch sampled from replay memory.
def optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma, device):
    if len(memory) < batch_size:
        return None
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.stack(batch.action).to(device)
    reward_batch = torch.stack(batch.reward).to(device)
    
    non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool)
    if any(non_final_mask):
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(device)
    else:
        non_final_next_states = torch.tensor([]).to(device)
    
    if action_batch.dim() == 1:
        action_batch = action_batch.unsqueeze(-1)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        if non_final_next_states.size(0) > 0:
            best_actions = policy_net(non_final_next_states).argmax(dim=1, keepdim=True)
            next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, best_actions).squeeze()
    
    expected_state_action_values = (next_state_values * gamma) + reward_batch.squeeze()
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()

# Build a directional map (i.e. optimal action at each state) from the trained policy.
def build_dir_map(env, policy_net, device):
    dir_map = np.zeros((env.grid_size, env.grid_size), dtype=int)
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            state = torch.tensor([i, j], dtype=torch.float32).to(device)
            with torch.no_grad():
                q_values = policy_net(state.unsqueeze(0))
                action = q_values.argmax(dim=1).item()
                dir_map[i, j] = action
    return dir_map

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a TensorBoard writer.
    writer = SummaryWriter(log_dir=args.log_dir)
    
    env = GridWorldEnv(grid_size=args.grid_size, gamma=args.gamma)
    memory = ReplayBuffer(args.memory_capacity)
    
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    policy_net = Q_Net(n_observations, args.hidden_size, n_actions).to(device)
    target_net = Q_Net(n_observations, args.hidden_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.learning_rate, amsgrad=True)
    
    episode_durations = []
    steps_done = 0
    global_step = 0
    print("Beginning Double DQN Training Phase ...")
    
    for i_episode in tqdm(range(args.episodes), desc="Training Episodes"):
        total_reward = 0
        start_time = time.time()
        state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
        for t in count():
            action = select_action(state, policy_net, steps_done, n_actions)
            steps_done += 1
            global_step += 1
            
            observation, reward, done = env.step(action)
            total_reward += reward
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
            
            if not done:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device)
            else:
                next_state = None
            
            memory.push(state, torch.tensor([action], device=device), reward_tensor, next_state, done)
            state = next_state if next_state is not None else state
            
            loss_val = optimize_model(memory, args.batch_size, policy_net, target_net, optimizer, args.gamma, device)
            if loss_val is not None:
                writer.add_scalar("Loss/step", loss_val, global_step)
                # Log the base learning rate from AdamW.
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("LearningRate", current_lr, global_step)
            
            # Soft update of target network.
            tau = args.tau
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
            target_net.load_state_dict(target_net_state_dict)
            
            if done or t >= args.max_steps:
                episode_duration = time.time() - start_time
                episode_durations.append(t + 1)
                writer.add_scalar("Reward/episode", total_reward, i_episode)
                writer.add_scalar("Time/episode", episode_duration, i_episode)
                break
    
    # plt.figure()
    # plt.plot(episode_durations)
    # plt.xlabel('Episode')
    # plt.ylabel('Duration (timesteps)')
    # plt.title('Training Performance')
    
    # output_path = os.path.join(args.output_dir, "training_performance.png")
    # plt.savefig(output_path)
    # print(f"Figure saved to {output_path}")
    # plt.show()
    
    dir_map = build_dir_map(env, policy_net, device)
    print("Directional map of the trained policy (action indices):")
    print(dir_map)
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DQN Training on a Custom Grid Environment")
    parser.add_argument('--grid_size', type=int, default=4, help='Size of the grid environment (default: 4)')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor (default: 0.9)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes (default: 10)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for optimization (default: 32)')
    parser.add_argument('--memory_capacity', type=int, default=10000, help='Replay memory capacity (default: 10000)')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size in the Q-network (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--max_steps', type=int, default=2000, help='Maximum steps per episode (default: 2000)')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target network (default: 0.005)')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the training performance figure (default: current directory)')
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory to store TensorBoard logs (default: ./runs)')
    
    args = parser.parse_args()
    main(args)
