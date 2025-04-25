import gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# ------------------------------------------------------------
# Utility: frame preprocessing
# ------------------------------------------------------------
import cv2

def _preprocess_frame(frame: np.ndarray):
    """Convert RGB (240x256x3) frame to 84x84 grayscale uint8."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame 


# ------------------------------------------------------------
# Deep Q‑Network – architecture from the 2015 Nature DQN paper
# ------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 84 → 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20 → 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9 → 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# Agent for evaluation – loads trained weights and selects argmax‑Q
# ------------------------------------------------------------
class Agent(object):
    """DQN Agent for Super Mario Bros (COMPLEX_MOVEMENT)."""

    WEIGHT_PATH = "mario_dqn.pth"  # update if you use a different filename

    def __init__(self):
        # Evaluation must run on CPU according to leaderboard rules
        self.device = torch.device("cpu")

        self.skip_count = 0
        self.last_action = 0

        # Action space has size 12 in COMPLEX_MOVEMENT
        self.action_space = gym.spaces.Discrete(12)
        n_actions = self.action_space.n

        # Build network and load weights – fall back to random if missing
        self.policy_net = DQN(n_actions).to(self.device)
        try:
            state_dict = torch.load(self.WEIGHT_PATH, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.policy_net.eval()
            self.use_network = True
        except FileNotFoundError:
            print(f"[student_agent] WARNING: '{self.WEIGHT_PATH}' not found. "
                  "Agent will act randomly.")
            self.use_network = False

        # Frame history (deque of 4 uint8 84×84 frames)
        self.frames: deque[np.ndarray] = deque(maxlen=4)

    def act(self, observation):
        """Return action for a single environment step."""
        if not self.use_network:
            return self.action_space.sample()

        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        # Preprocess incoming RGB frame
        processed = _preprocess_frame(observation)  # uint8 84×84
        self.frames.append(processed)

        # If we don't have 4 frames yet (start of episode), pad with copies
        while len(self.frames) < 4:
            self.frames.append(processed)

        # Build (4,84,84) tensor
        state = np.stack(self.frames, axis=0)  # uint8
        state = (torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0)

        with torch.no_grad():
            q_values = self.policy_net(state)
            action = int(q_values.argmax(dim=1).item())

        self.last_action = action
        self.skip_count = 4 - 1

        return action
