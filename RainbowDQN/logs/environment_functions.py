import gymnasium as gym
import cv2
from collections import deque
import numpy as np

# ----------------------
# Preprocessing with Frame Stacking
# ----------------------

class AtariPreprocessingWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip=4, stack_size=4):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.frame_buffer = deque(maxlen=2)  # For max-pooling over frames
        self.stack_size = stack_size
        self.stacked_frames = deque(maxlen=stack_size)

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.frame_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        # Max-pool the last two frames.
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[-1])
        processed_frame = self._preprocess(max_frame)
        self.stacked_frames.append(processed_frame)
        stacked_state = self._get_stacked_state()
        return stacked_state, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frame_buffer.clear()
        self.stacked_frames.clear()
        self.frame_buffer.append(obs)
        processed_frame = self._preprocess(obs)
        # Fill the stack with the initial frame.
        for _ in range(self.stack_size):
            self.stacked_frames.append(processed_frame)
        return self._get_stacked_state(), info

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=0)  # Shape: (1,84,84)

    def _get_stacked_state(self):
        return np.concatenate(self.stacked_frames, axis=0)  # Shape: (4,84,84)