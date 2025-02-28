import gymnasium as gym
import ale_py
import cv2
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

# Check for GPU availability.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

gamma = 0.995

# Register ALE environments for Gymnasium.
gym.register_envs(ale_py)

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

# Where to store training logs.
LOGS_PATH = "logs/training_stats.json"

# Preprocessing function: Grayscale + Resize to 84×84 + Normalize + Add channel dimension
def preprocess_observation(obs):
    # obs is shape (210, 160, 3) in Breakout
    # Convert to grayscale
    obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
    # Convert to float and normalize
    obs_tensor = torch.FloatTensor(obs_resized) / 255.0
    # Add a channel dimension => shape (1, 84, 84)
    obs_tensor = obs_tensor.unsqueeze(0)
    return obs_tensor

class Res_Block(nn.Module):
    def __init__(self, channels):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        return self.relu(x + out)

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # Input shape is now (1, 84, 84) due to grayscale + resizing.
        self.reduce_size = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # e.g. output shape: (16, 42, 42)
        )
        # Stack 8 residual blocks on the 16-channel output.
        self.res_blocks = nn.Sequential(*[Res_Block(16) for _ in range(8)])
        # Convolutional layers for feature extraction.
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # e.g. (32, 21, 21)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # e.g. (64, 11, 11)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # e.g. (64, 6, 6)
            nn.ReLU(),
        )
        # Flatten and add Fully Connected Layers.
        self.flatten = nn.Flatten()
        # Calculate final spatial dims if you need to be precise, or use a test input.
        # We'll assume (64, 6, 6) => 64 * 6 * 6 = 2304
        self.fc = nn.Sequential(
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        out = self.reduce_size(x)
        out = self.res_blocks(out)
        out = self.conv2(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def display_stats(reward_history, loss_history, max_q_value_history):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(reward_history, marker='o')
    plt.title("Reward Tracking")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 3, 2)
    plt.plot(loss_history, marker='o', color='orange')
    plt.title("Q Value Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.subplot(1, 3, 3)
    plt.plot(max_q_value_history, marker='o', color='green')
    plt.title("Max Q Value")
    plt.xlabel("Episode")
    plt.ylabel("Q Value")
    
    plt.tight_layout()
    plt.show()

# Mapping for the reduced action space.
action_map = {0: 0, 1: 2, 2: 3}

def train_DQN(policy_net, target_net, num_episodes=5, batch_size=128, max_steps=3000, learning_rate=0.003, eps=1):
    env = gym.make("Breakout-v4", render_mode="rgb_array")
    replay_buffer = ReplayBuffer(100000)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
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

    # Reset environment and preprocess first frame.
    state, info = env.reset()
    state, reward, terminated, truncated, info = env.step(1)
    done = terminated or truncated
    lives = info.get("lives", None)
    # Preprocess (grayscale + resize).
    state = preprocess_observation(state)

    record_breakout_game(policy_net)  # Record an initial game.
    display_stats(reward_history, loss_history, max_q_value_history)

    for episode in range(start_episode, num_episodes):
        state, info = env.reset()
        state, reward, terminated, truncated, info = env.step(1)
        done = terminated or truncated
        lives = info.get("lives", lives)
        # Preprocess observation.
        state = preprocess_observation(state)

        episode_reward = 0
        steps = 0
        episode_losses = []
        max_q_value = -100

        while not done and steps < max_steps:
            # For inference, move state to GPU on the fly.
            if random.uniform(0, 1) <= eps:
                reduced_action = random.choice([0, 1, 2])
            else:
                with torch.no_grad():
                    q_values = policy_net(state.unsqueeze(0).to(device))
                    reduced_action = torch.argmax(q_values, dim=1).item()
                    max_q_value = max(max_q_value, q_values[:, reduced_action].item())
            action = action_map[reduced_action]

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_lives = info.get("lives", lives)
            if current_lives is not None and current_lives < lives and not done:
                next_state, fire_reward, terminated, truncated, info = env.step(1)
                reward += fire_reward-5
                done = terminated or truncated
                current_lives = info.get("lives", current_lives)
            lives = current_lives
            episode_reward += reward

            # Preprocess next state
            next_state_tensor = preprocess_observation(next_state)
            replay_buffer.add((state, reduced_action, reward, next_state_tensor, done))
            state = next_state_tensor
            steps += 1
        
        if len(replay_buffer) >= batch_size:
            for q in range(5):
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                # Convert entire batch to GPU.
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
                if episode>100:
                    breaker=15
                else:
                    breaker=2
                # Update target network occasionally
                if episode % breaker  == 0:
                    target_net.load_state_dict(policy_net.state_dict())
        else:
            episode_losses = [0]

    # Epsilon decay
    eps = max(0.3, eps * (0.999 ** episode**(1/2)))
    reward_history.append(episode_reward)
    loss_history.append(sum(episode_losses) / len(episode_losses))
    max_q_value_history.append(max_q_value)

    # Save logs and model
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

    #if episode in [10, 20, 50, 100, 200]:
        #   record_breakout_game(policy_net)
        #  display_stats(reward_history, loss_history, max_q_value_history)

    env.close()

def record_breakout_game(policy_net, max_steps=3000, output_file="breakout.gif"):
    env = gym.make("Breakout-v4", render_mode="rgb_array")
    policy_net.eval()
    frames = []
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(1)
    frames.append(env.render())
    lives = info.get("lives", None)

    # Preprocess and move to GPU for evaluation
    observation = preprocess_observation(observation).to(device)
    done = terminated or truncated
    steps = 0

    while not done and steps < max_steps:
        with torch.no_grad():
            q_values = policy_net(observation.unsqueeze(0))
            reduced_action = torch.argmax(q_values, dim=1).item()
        action = action_map[reduced_action]
        observation_cpu, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_lives = info.get("lives", lives)
        if current_lives is not None and current_lives < lives and not done:
            observation_cpu, fire_reward, terminated, truncated, info = env.step(1)
            reward += fire_reward
            done = terminated or truncated
            current_lives = info.get("lives", current_lives)
        lives = current_lives

        # Preprocess for next step
        observation = preprocess_observation(observation_cpu).to(device)
        frames.append(env.render())
        steps += 1

    env.close()
    print("Number of frames captured:", len(frames))
    if frames:
        plt.imshow(frames[0])
        plt.show()

    imageio.mimsave(output_file, frames, fps=30)
    print(f"Saved game recording to {output_file}")

if __name__ == "__main__":
    num_actions = 3
    policy_net = DQN(num_actions).to(device)
    target_net = DQN(num_actions).to(device)
    train_DQN(policy_net, target_net, num_episodes=100000)
    record_breakout_game(policy_net)
