
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
LOGS_PATH = "logs/training_stats.json"
def display_stats(reward_history, loss_history, max_q_value_history,q_values_per_game):
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

def save_checkpoint(file_path, data):
    """Saves training logs to JSON file."""
    # Konvertiere alle Tensors in Listen oder skalare Werte
    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Konvertiere Tensor in eine Python-Liste
        elif isinstance(obj, list):
            return [convert_tensor(x) for x in obj]  # Rekursiv für Listen
        elif isinstance(obj, dict):
            return {k: convert_tensor(v) for k, v in obj.items()}  # Für Dictionaries
        return obj  # Falls es kein Tensor ist, einfach zurückgeben

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(convert_tensor(data), f)  # Umgewandelte Daten speichern

def load_checkpoint(file_path):
    """Loads training logs from JSON file if available."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None


def display_st():
    logs = load_checkpoint(LOGS_PATH)
    if logs is not None:
        start_episode = logs.get("episode", 0)
        reward_history = logs.get("reward_history", [])
        loss_history = logs.get("loss_history", [])
        max_q_value_history = logs.get("max_q_value_history", [])
        try:
            q_values_per_game=logs.get("q_values_per_game", [])
        except:
            pass
        print(f"Loaded training logs from episode {start_episode}.")
    else:
        start_episode = 0
        reward_history = []
        loss_history = []
        max_q_value_history = []
    display_stats(reward_history, loss_history, max_q_value_history,q_values_per_game)

if __name__ == "__main__":
    display_st()