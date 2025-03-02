
#Imports
import gymnasium as gym
import ale_py
from environment_functions import AtariPreprocessingWrapper
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import os
import json  # For saving logs as JSON
from collections import deque
import numpy as np
# ----------------------
# Setup
# ----------------------
# Where to store training logs.
LOGS_PATH = "logs/training_stats.json"
gamma = 0.99
# Register ALE environments for Gymnasium.
gym.register_envs(ale_py)
# Check for GPU availability.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# JSON-based logging functions.
def save_checkpoint(file_path, data):
    """Saves training logs to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_checkpoint(file_path):
    """Loads training logs from JSON file if available."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None





# ----------------------
# DQN Network Definition
# ----------------------
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        # Compute conv output size: assuming input (4,84,84)
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc_layers(x)

# ----------------------
# Replay Buffer
# ----------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ----------------------
# Helper Function to Convert Observations
# ----------------------
def preprocess_observation(obs):
    """Converts a numpy observation (stacked frames) into a torch tensor."""
    return torch.from_numpy(obs).float()

# ----------------------
# Display Training Stats
# ----------------------
def display_stats(reward_history, loss_history, max_q_value_history):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(reward_history)
    plt.title("Reward Tracking")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 3, 2)
    plt.plot(loss_history, color='orange')
    plt.title("Q Value Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.subplot(1, 3, 3)
    plt.plot(max_q_value_history, color='green')
    plt.title("Max Q Value")
    plt.xlabel("Episode")
    plt.ylabel("Q Value")
    
    plt.tight_layout()
    plt.show()

# ----------------------
# Reduced Action Space Mapping
# ----------------------
# Here we assume raw actions: 0 = NOOP, 1 = FIRE, 2 = RIGHT, 3 = LEFT.
# The reduced action space used by the agent (indices 0,1,2) maps to:
action_map = {0: 0, 1: 2, 2: 3}  # FIRE is not in the reduced set; we trigger it separately when needed.

# ----------------------
# Training Loop with Ball-Out Handling (FIRE action when ball is lost)
# ----------------------
def train_DQN(policy_net, target_net, num_episodes=5, batch_size=1028, max_steps=10000, learning_rate=0.001, eps=1):
    env = gym.make("Breakout-v4", render_mode="rgb_array")
    env = AtariPreprocessingWrapper(env)
    replay_buffer = ReplayBuffer(400000)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    # Load saved weights if available.
    try:
        policy_net.load_state_dict(torch.load("model_weights.pth"))
    except:
        pass    
    
    target_net.load_state_dict(policy_net.state_dict())

    logs = load_checkpoint(LOGS_PATH)
    if logs is not None:
        start_episode = logs.get("episode", 0)
        reward_history = logs.get("reward_history", [])
        loss_history = logs.get("loss_history", [])
        max_q_value_history = logs.get("max_q_value_history", [])
        print(f"Loaded training logs from episode {start_episode}.")
    else:
        start_episode = 0
        reward_history = []
        loss_history = []
        max_q_value_history = []

    # Start the game by taking the FIRE action.
    state, info = env.reset()
    state, reward, terminated, truncated, info = env.step(1)
    done = terminated or truncated
    lives = info.get("lives", None)

    # Record an initial game and display current stats.
    record_breakout_game(policy_net)
    display_stats(reward_history, loss_history, max_q_value_history)

    for episode in range(start_episode, num_episodes):
        state, info = env.reset()
        state, reward, terminated, truncated, info = env.step(1)  # FIRE to launch ball.
        done = terminated or truncated
        lives = info.get("lives", lives)
        episode_reward = 0
        steps = 0
        episode_losses = []
        max_q_value = 0

        # Convert state (numpy) to tensor.
        state = preprocess_observation(state)

        while not done and steps < max_steps:
            # Epsilon-greedy action selection.
            if random.uniform(0, 1) <= eps:
                reduced_action = random.choice([0, 1, 2])
            else:
                with torch.no_grad():
                    q_values = policy_net(state.unsqueeze(0).to(device))
                    reduced_action = torch.argmax(q_values, dim=1).item()
                    max_q_value = max(max_q_value, q_values[0, reduced_action].item())
            # Map reduced action to raw environment action.
            action = action_map[reduced_action]

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_lives = info.get("lives", lives)
            # If a life is lost (ball is out) and the game is not done, force a FIRE action.
            if current_lives is not None and current_lives < lives and not done:
                next_state, fire_reward, terminated, truncated, info = env.step(1)
                reward += fire_reward - 5  # Optional penalty adjustment.
                done = terminated or truncated
                current_lives = info.get("lives", current_lives)
            lives = current_lives
            episode_reward += reward

            # Convert next_state (numpy array) to tensor.
            next_state_tensor = preprocess_observation(next_state)
            # Save transition in the replay buffer.
            replay_buffer.add((state, reduced_action, reward, next_state_tensor, done))
            state = next_state_tensor
            steps += 1
            if steps%4==0:
                if len(replay_buffer) >= max(batch_size,100):
                    batch = replay_buffer.sample(batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    states = torch.stack(states).to(device)
                    actions = torch.LongTensor(actions).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)
                    next_states = torch.stack(next_states).to(device)
                    dones = torch.FloatTensor(dones).to(device)

                    q_current = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_targets = rewards + gamma * target_net(next_states).max(1)[0] * (1 - dones)
                    loss = nn.MSELoss()(q_current, q_targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    episode_losses.append(loss.item())

        # Update target network periodically.
        breaker = 8
        if episode % breaker == 1:
            target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon
        eps = max(0.05, eps * (0.999 ** (episode)))
        reward_history.append(episode_reward)
        loss_history.append(sum(episode_losses) / len(episode_losses) if episode_losses else 0)
        max_q_value_history.append(max_q_value)

        # Save logs and model every 10 episodes.
        if episode % 10 == 0:
            logs = {
                "episode": episode,
                "reward_history": reward_history,
                "loss_history": loss_history,
                "max_q_value_history": max_q_value_history
            }
            save_checkpoint(LOGS_PATH, logs)
            torch.save(policy_net.state_dict(), "model_weights.pth")
            print(f"Training logs saved at episode {episode}.")
            print(f"Episode {episode+1}: Reward = {episode_reward}, Avg Loss = {loss_history[-1]:.4f}, Max Q Value = {max_q_value:.4f}")
        if episode%100==0:
            record_breakout_game(policy_net)
    env.close()

# ----------------------
# Record a Game (Evaluation)
# ----------------------
def record_breakout_game(policy_net, max_steps=10000, output_file="breakout.gif"):
    env = gym.make("Breakout-v4", render_mode="rgb_array")
    env = AtariPreprocessingWrapper(env)
    policy_net.eval()
    frames = []
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(1)
    frames.append(env.render())
    lives = info.get("lives", None)
    done = terminated or truncated
    steps = 0

    # Convert observation to tensor.
    observation = preprocess_observation(observation)

    while not done and steps < max_steps:
        with torch.no_grad():
            q_values = policy_net(observation.unsqueeze(0).to(device))
            reduced_action = torch.argmax(q_values, dim=1).item()
        action = action_map[reduced_action]
        observation_np, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_lives = info.get("lives", lives)
        if current_lives is not None and current_lives < lives and not done:
            observation_np, fire_reward, terminated, truncated, info = env.step(1)
            reward += fire_reward
            done = terminated or truncated
            current_lives = info.get("lives", current_lives)
        lives = current_lives

        # Convert new observation to tensor.
        observation = preprocess_observation(observation_np)
        frames.append(env.render())
        steps += 1

    env.close()
    """print("Number of frames captured:", len(frames))
    if frames:
        plt.imshow(frames[0])
        plt.show()"""

    imageio.mimsave(output_file, frames, fps=30)
    print(f"Saved game recording to {output_file}")

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    num_actions = 3  # Reduced action space indices: 0 (NOOP), 1 (RIGHT), 2 (LEFT)
    input_shape = (4, 84, 84)
    policy_net = DQN(input_shape, num_actions).to(device)
    target_net = DQN(input_shape, num_actions).to(device)
    # Train for a specified number of episodes.
    train_DQN(policy_net, target_net, num_episodes=100000)
    record_breakout_game(policy_net)
