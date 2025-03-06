# ----------------------
# Imports
# ----------------------
import os
import json
import random
from collections import deque
import time
from pathlib import Path

import gymnasium as gym
import ale_py
import imageio
import numpy as np
import matplotlib.pyplot as plt
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Eigene Hilfsfunktionen für Atari-Umgebungen
from environment_functions import AtariPreprocessingWrapper
from utils.config_manager import ConfigManager
import torch.serialization
torch.serialization.add_safe_globals([deque])
# ----------------------
# Setup and Configuration
# ----------------------
config_manager = ConfigManager('config/dqn_config.yaml')
config = config_manager.params
LOGS_PATH = config['logging']['logs_path']
warmup_steps=config['training']['warmup_steps']
# TensorBoard Writer with improved run management
writer = SummaryWriter(log_dir=config_manager.get_tensorboard_dir())

# Register ALE-Umgebungen für Gymnasium
gym.register_envs(ale_py)

# Gerät (GPU falls verfügbar)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
    
    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)
# ----------------------
# JSON-basierte Logging-Funktionen
# ----------------------
def save_checkpoint(file_path, data):
    """Speichert Trainingslogs als JSON."""
    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Tensor in Liste umwandeln
        elif isinstance(obj, list):
            return [convert_tensor(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_tensor(v) for k, v in obj.items() }
        return obj

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(convert_tensor(data), f)

def load_checkpoint(file_path):
    """Lädt Trainingslogs aus einer JSON-Datei, falls vorhanden."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

# ----------------------
# Replay Buffer
# ----------------------


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        capacity: maximale Anzahl gespeicherter Transitions
        alpha: exponentieller Faktor, der bestimmt, wie stark Prioritäten gewichtet werden (0 = uniform, 1 = volle Priorisierung)
        """
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def add(self, experience, priority=None):
        """
        Fügt eine Transition hinzu. Wird kein Priority-Wert angegeben,
        so wird der maximale bisherige Wert verwendet, um sicherzustellen, dass neue Erfahrungen häufig gewählt werden.
        """
        if len(self.buffer) == 0:
            max_prio = 1.0
        else:
            max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if priority is None:
            priority = max_prio
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Zieht eine Stichprobe von Transitions, wobei die Auswahlwahrscheinlichkeit
        proportional zu (priority^alpha) ist.
        beta: Korrekturfaktor für die importance-sampling Gewichte
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Berechne die importance-sampling Gewichte
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalisierung, damit max(weights)==1
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Aktualisiert die Prioritäten für die angegebenen Indizes.
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


# ----------------------
# DQN-Netzwerkdefinition
# ----------------------
class DuelingNoisyDQN(nn.Module):
    def __init__(self, input_dim, output_dim, filename="model_weights.pth"):
        super(DuelingNoisyDQN, self).__init__()
        self.input_dim = input_dim
        channels, _, _ = input_dim

        # Convolutional Layers mit ReLU-Aktivierung
        self.l1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Berechne die Ausgabedimension der Conv-Schicht
        conv_output_size = self.conv_output_dim()
        lin1_output_size = 512

        # Fully Connected Layers
        self.value_net_1 = NoisyLinear(conv_output_size, lin1_output_size)
        self.value_net_2 = NoisyLinear(lin1_output_size, 1)
            
        self.advantage_net_1 = NoisyLinear(conv_output_size, lin1_output_size)
        self.advantage_net_2 = NoisyLinear(lin1_output_size, output_dim)
  

        # Dateiname zum Speichern des Modells
        self.filename = filename

    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))
    
    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], -1)
        values = self.value_net_1(x)
        values = self.value_net_2(values)
        advantages = self.advantage_net_1(x)
        advantages = self.advantage_net_2(advantages)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

# ----------------------
# Hilfsfunktionen
# ----------------------
def preprocess_observation(obs):
    """Konvertiert einen NumPy-Frame in einen Torch-Tensor."""
    return torch.from_numpy(obs).float()

def display_stats(reward_history, loss_history, max_q_value_history, q_values_per_game):
    """Zeigt Trainingsstatistiken in mehreren Diagrammen an."""
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 4, 1)
    plt.plot(reward_history)
    plt.title("Reward Tracking")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 4, 2)
    plt.plot(loss_history, color='orange')
    plt.title("Q Value Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    
    plt.subplot(1, 4, 3)
    plt.plot(max_q_value_history, color='green')
    plt.title("Max Q Value")
    plt.xlabel("Episode")
    plt.ylabel("Q Value")
    
    plt.subplot(1, 4, 4)
    plt.plot(q_values_per_game, color='red')
    plt.title("Q values per step")
    plt.xlabel("Step")
    plt.ylabel("Q Value")
    
    plt.tight_layout()
    plt.show()

# Mapping des reduzierten Aktionsraums (0: NOOP, 1: RIGHT, 2: LEFT)
action_map = {0: 0, 1: 2, 2: 3}  # FIRE wird separat behandelt.
reverse_action_map = {0: 0, 1: 2, 2: 1}

def augment_data(state, reduced_action, reward, next_state_tensor, done):
    """Datenaugmentation durch horizontales Spiegeln des Zustands."""
    return (torch.flip(state, dims=[2]),
            reverse_action_map[reduced_action],
            reward,
            torch.flip(next_state_tensor, dims=[2]),
            done)

# ----------------------
# Trainings- und Evaluationsfunktionen
# ----------------------
def record_breakout_game(policy_net, max_steps=4000, output_file="breakout.gif"):
    """Zeichnet ein Spiel auf und speichert es als GIF."""
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
    q_values_per_game = []
    
    observation = preprocess_observation(observation)
    policy_net.eval()
    while not done and steps < max_steps:
        with torch.no_grad():
            q_values = policy_net(observation.unsqueeze(0).to(device))
            reduced_action = torch.argmax(q_values, dim=1).item()
            q_values_per_game.append(q_values.max(1)[0].cpu())
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

        observation = preprocess_observation(observation_np)
        frames.append(env.render())
        steps += 1
    policy_net.train()
    env.close()
    imageio.mimsave(output_file, frames, fps=10)
    print(f"Saved game recording to {output_file}")

    return q_values_per_game

def log_model_histograms(model, episode):
    """Loggt Gewichts- und Gradienten-Histogramme für das Modell."""
    for name, param in model.named_parameters():
        writer.add_histogram(f"{name}/weights", param.data.cpu(), episode)
        if param.grad is not None:
            writer.add_histogram(f"{name}/grads", param.grad.cpu(), episode)

def train_DQN(policy_net, target_net):
    """Trainiert das DQN-Modell über mehrere Episoden und loggt umfassend in TensorBoard."""
    env = gym.make(config['environment']['name'], 
                  render_mode=config['environment']['render_mode'])
    env = AtariPreprocessingWrapper(env)
    replay_buffer = PrioritizedReplayBuffer(config['training']['replay_buffer_size'])

    optimizer = optim.Adam(policy_net.parameters(), 
                          lr=config['training']['learning_rate'])
    
    # Load weights if available
    checkpoint_dir = config_manager.get_checkpoint_dir()
    model_path = checkpoint_dir / "latest_model.pth"
    if model_path.exists():
        policy_net.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}")

    target_net.load_state_dict(policy_net.state_dict())

    # Trainingslogs laden, falls vorhanden.
    logs = load_checkpoint(LOGS_PATH)
    if logs is not None:
        start_episode = logs.get("episode", 0)
        reward_history = logs.get("reward_history", [])
        loss_history = logs.get("loss_history", [])
        max_q_value_history = logs.get("max_q_value_history", [])
        q_values_per_game = logs.get("q_values_per_game", [])
        update_steps = logs.get("update_steps", 0)

        print(f"Loaded training logs from episode {start_episode}.")
    else:
        start_episode = 0
        update_steps = 0
        reward_history = []
        loss_history = []
        max_q_value_history = []
        q_values_per_game = []

    # Spielstart: FIRE-Aktion, um den Ball zu starten.
    state, info = env.reset()
    state, reward, terminated, truncated, info = env.step(1)
    done = terminated or truncated
    lives = info.get("lives", None)
    
    # Erste Aufzeichnung des Spiels (Evaluation)
    q_values_per_game = record_breakout_game(policy_net)
    display_stats(reward_history, loss_history, max_q_value_history, q_values_per_game)

    breaker = 0

    session_start_time = time.time()
    last_save_time = session_start_time
    #eps=1.0
    for episode in range(start_episode, config['training']['num_episodes']):
        # Check session duration
        current_time = time.time()
        session_duration = current_time - session_start_time
        
        # Check if we need to end the session
        if session_duration >= config['run_management']['max_session_duration']:
            print(f"\nSession duration limit reached ({session_duration/3600:.1f} hours)")
            print(f"Saving checkpoint and ending session. Resume by setting resume_from: {config_manager.run_id} in config.")
            # Save final checkpoint
            checkpoint_path = config_manager.get_checkpoint_dir() / f"model_ep_{episode}.pth"
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward_history': reward_history,
                'loss_history': loss_history,
                'max_q_value_history': max_q_value_history,
                'q_values_per_game': q_values_per_game,
                'replay_buffer': replay_buffer.buffer
            }, checkpoint_path)
            return  # End the training session
            
        # Periodic saving
        if current_time - last_save_time >= config['run_management']['save_frequency_seconds']:
            checkpoint_path = config_manager.get_checkpoint_dir() / "latest.pth"
            torch.save({
                'episode': episode,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward_history': reward_history,
                'loss_history': loss_history,
                'max_q_value_history': max_q_value_history,
                'q_values_per_game': q_values_per_game,
                'replay_buffer': replay_buffer.buffer
            }, checkpoint_path)
            last_save_time = current_time
            print(f"\nIntermediate checkpoint saved at episode {episode}")

        episode_start_time = time.time()
        state, info = env.reset()
        state, reward, terminated, truncated, info = env.step(1)  # FIRE-Aktion
        done = terminated or truncated
        lives = info.get("lives", lives)
        episode_reward = 0
        steps = 0
        episode_losses = []
        max_q_value = 0

        state = preprocess_observation(state)

        while not done and steps < config['training']['max_steps']:
            with torch.no_grad():
                q_values = policy_net(state.unsqueeze(0).to(device))
                reduced_action = torch.argmax(q_values, dim=1).item()
                max_q_value = max(max_q_value, q_values[0, reduced_action].item())
            action = action_map[reduced_action]

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            current_lives = info.get("lives", lives)
            # Wenn ein Leben verloren wurde, erzwinge FIRE
            if current_lives is not None and current_lives < lives and not done:
                next_state, fire_reward, terminated, truncated, info = env.step(1)
                reward += fire_reward -5# Strafanpassung optional
                done = terminated or truncated
                current_lives = info.get("lives", current_lives)
            lives = current_lives
            episode_reward += reward

            next_state_tensor = preprocess_observation(next_state)
            # Speichere Transitionen im Replay Buffer (inklusive Datenaugmentation)
            replay_buffer.add((state, reduced_action, reward, next_state_tensor, done))
            #replay_buffer.add(augment_data(state, reduced_action, reward, next_state_tensor, done))
            state = next_state_tensor
            steps += 1

            # Trainingsupdate alle 4 Schritte
            # Innerhalb des Trainingsloops, z.B. alle 2 Schritte:
            beta_start = 0.4
            beta_frames = warmup_steps  # Anzahl der Schritte, über die beta linear ansteigt
            beta = min(1.0, beta_start + update_steps * (1.0 - beta_start) / beta_frames)

            if steps % 2 == 0 and len(replay_buffer) >= max(config['training']['batch_size'], 5000):
                batch, indices, weights = replay_buffer.sample(config['training']['batch_size'], beta)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.stack(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.stack(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                weights = torch.FloatTensor(weights).to(device)

                q_values = policy_net(states)
                q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    # Double DQN: Aktionselektion mit policy_net, Bewertung mit target_net
                    action_predicted = policy_net(next_states).max(1)[1].unsqueeze(1)
                    q_targets_next = target_net(next_states).gather(1, action_predicted).squeeze(1)
                    q_targets = rewards + config['training']['gamma'] * q_targets_next * (1 - dones)
                
                # Berechne den TD-Error und gewichte den Loss
                td_errors = torch.abs(q_current - q_targets).detach()
                loss = (weights * (q_current - q_targets) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_losses.append(loss.item())
                
                # Aktualisiere die Prioritäten im Replay Buffer (kleines epsilon zur Stabilität)
                new_priorities = td_errors.cpu().numpy() + 1e-6
                replay_buffer.update_priorities(indices, new_priorities)

                # (Der Rest deines Update- und Zielnetzwerk-Codes bleibt unverändert.)

                TARGET_UPDATE_FREQUENCY = 10000 if episode > 300 else 500
                breaker += 1
                update_steps += 1
                if update_steps % TARGET_UPDATE_FREQUENCY == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    print(f"Target-Netzwerk Update bei Update-Schritt {update_steps}")

        # Epsilon-Decay: lineare Abnahme von 1.0 auf 0.1 über die Episoden
        # Time-based decay using total environment steps rather than episodes
        total_steps = update_steps  # Since you update every 4 steps
        """eps_start = 1.0
        eps_end = 0.1
        eps_decay_steps = warmup_steps  # 1M steps for full decay
        eps = eps_end + (eps_start - eps_end) * max(0, (eps_decay_steps - total_steps) / eps_decay_steps)"""
        reward_history.append(episode_reward)
        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
        loss_history.append(avg_loss)
        max_q_value_history.append(max_q_value)
        episode_duration = time.time() - episode_start_time

        # Logge Scalar-Metriken für die aktuelle Episode
        writer.add_scalar("Episode/Reward", episode_reward, episode)
        writer.add_scalar("Episode/AverageLoss", avg_loss, episode)
        writer.add_scalar("Episode/MaxQ", max_q_value, episode)
        #writer.add_scalar("Episode/Epsilon", eps, episode)
        writer.add_scalar("Episode/Steps", steps, episode)
        writer.add_scalar("Episode/BufferSize", len(replay_buffer), episode)
        writer.add_scalar("Episode/UpdateSteps", update_steps, episode)
        writer.add_scalar("Episode/Duration_sec", episode_duration, episode)

        # Logge Histogramme der Gewichte und Gradienten alle 50 Episoden
        if episode % 50 == 0:
            log_model_histograms(policy_net, episode)

        # Speichere alle 10 Episoden Trainingslogs und Modellgewichte
        if episode % 10 == 0:
            logs = {
                "episode": episode,
                "reward_history": reward_history,
                "loss_history": loss_history,
                "max_q_value_history": max_q_value_history,
                "q_values_per_game": q_values_per_game,
                "update_steps": update_steps
            }
            save_checkpoint(LOGS_PATH, logs)
            torch.save(policy_net.state_dict(), "model_weights.pth")
            print(f"Training logs und Modell bei Episode {episode} gespeichert.")
            print(f"Episode {episode+1}: Reward = {episode_reward}, Avg Loss = {avg_loss:.4f}, Max Q Value = {max_q_value:.4f}, Steps = {steps}, Buffer Size = {len(replay_buffer)}, Update Steps = {update_steps}")

        if episode % 100 == 0:
            q_values_per_game = record_breakout_game(policy_net)

    env.close()

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    num_actions = config['model']['num_actions']
    input_shape = tuple(config['model']['input_shape'])
    
    policy_net = DuelingNoisyDQN(input_shape, num_actions).to(device)
    target_net = DuelingNoisyDQN(input_shape, num_actions).to(device)
    
    # Load checkpoint if resuming
    if config['run_management']['resume_from']:
        checkpoint_dir = Path(config['logging']['base_log_dir']) / config['run_management']['resume_from'] / "checkpoints"
        checkpoint_path = checkpoint_dir / "latest.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Resumed from checkpoint at episode {checkpoint['episode']}")
        else:
            print(f"Warning: Could not find checkpoint at {checkpoint_path}")
    
    train_DQN(policy_net, target_net)
    
    # Zeichne ein finales Spiel auf
    record_breakout_game(policy_net)
    print("Training completed.")
