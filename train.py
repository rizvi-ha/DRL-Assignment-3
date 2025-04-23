import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gym_super_mario_bros import make as make_mario
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from tqdm import tqdm

# ------------------------------------------------------------
# Hyper‑parameters
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("[train.py] FATAL: CUDA not available – training on CPU is far too slow.")
    exit(1)

BATCH_SIZE   = 8192 
BUFFER_CAP   = 100_000
GAMMA        = 0.99
LR           = 1e-3
TARGET_UPD   = 2_000        # steps between target network syncs
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 300_000     # linear decay steps
TOTAL_STEPS  = 500_000
SAVE_EVERY   = 10_000
WEIGHT_PATH  = "mario_dqn.pth"

# ------------------------------------------------------------
# Environment helper – handles grayscale, resize, framestack on the fly
# ------------------------------------------------------------

def make_env():
    env = make_mario("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)  # (H,W,1)
    env = ResizeObservation(env, 84)               # (84,84,1)
    env = FrameStack(env, 4)                       # (4,84,84,1)
    return env

# utility to squeeze trailing channel‑dim if present
def squeeze_obs(obs):
    arr = np.array(obs, dtype=np.uint8)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)  # -> (4,84,84)
    return arr

# ------------------------------------------------------------
# Replay Buffer – contiguous arrays, no Python loops in sample
# ------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.state      = np.empty((capacity, 4, 84, 84), dtype=np.uint8)
        self.action     = np.empty((capacity,),           dtype=np.int16)
        self.reward     = np.empty((capacity,),           dtype=np.float32)
        self.next_state = np.empty((capacity, 4, 84, 84), dtype=np.uint8)
        self.done       = np.empty((capacity,),           dtype=np.bool_)

    def push(self, s, a, r, n, d):
        self.state[self.ptr]      = s
        self.action[self.ptr]     = a
        self.reward[self.ptr]     = r
        self.next_state[self.ptr] = n
        self.done[self.ptr]       = d
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, k):
        idx = np.random.randint(0, self.size, size=k)
        s  = torch.as_tensor(self.state[idx],      device=DEVICE, dtype=torch.uint8).float() / 255.
        n  = torch.as_tensor(self.next_state[idx], device=DEVICE, dtype=torch.uint8).float() / 255.
        a  = torch.as_tensor(self.action[idx],     device=DEVICE, dtype=torch.long)
        r  = torch.as_tensor(self.reward[idx],     device=DEVICE)
        d  = torch.as_tensor(self.done[idx],       device=DEVICE).float()
        return s, a, r, n, d

    def __len__(self):
        return self.size

# ------------------------------------------------------------
# DQN architecture
# ------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------

env = make_env()
n_actions = env.action_space.n

policy_net = DQN(n_actions).to(DEVICE)
target_net = DQN(n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())

optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
replay_buf = ReplayBuffer(BUFFER_CAP)

eps   = EPS_START
state = squeeze_obs(env.reset())  # (4,84,84)

episode     = 0
cum_reward  = 0
log_f = open("log.txt", "w")
curr_loss = []

progress = tqdm(range(TOTAL_STEPS), desc="Training", miniters=100)
for step in progress:

    # ---------- act ----------
    if np.random.rand() < eps:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            s = torch.as_tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0) / 255.
            action = int(policy_net(s).argmax(1).item())

    # ---------- env step ----------
    next_state, reward, done, info = env.step(action)
    next_state = squeeze_obs(next_state)
    cum_reward += reward

    replay_buf.push(state, action, reward, next_state, done)
    state = next_state

    # ε‑decay
    eps = max(EPS_END, EPS_START - step / EPS_DECAY)

    # ---------- learn ----------
    if len(replay_buf) > 20_000 and step % 4 == 0:
        s_batch, a_batch, r_batch, n_batch, d_batch = replay_buf.sample(BATCH_SIZE)
        q_pred = policy_net(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = target_net(n_batch).max(1)[0]
            q_target = r_batch + GAMMA * q_next * (1 - d_batch)
        loss = nn.functional.smooth_l1_loss(q_pred, q_target)
        curr_loss.append(loss)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # ---------- target update ----------
    if step % TARGET_UPD == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # ---------- episode end ----------
    if done:
        avg_loss = np.mean(curr_loss)
        log_f.write(f"Episode {episode}\tReward {cum_reward}\tEpsilon {eps}\tAvg loss: {avg_loss}\n"); log_f.flush()
        episode += 1
        state = squeeze_obs(env.reset())
        cum_reward = 0
        curr_loss = []

    # ---------- checkpoint ----------
    if step % SAVE_EVERY == 0 and step > 0:
        torch.save(policy_net.state_dict(), WEIGHT_PATH)
        print(f"[train.py] Saved checkpoint at step {step}")

# final save
torch.save(policy_net.state_dict(), WEIGHT_PATH)
log_f.close()
print("[train.py] Training complete – weights saved to", WEIGHT_PATH)

