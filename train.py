import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros import make as make_mario
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from tqdm import tqdm

# TorchRL imports
from tensordict import TensorDict
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler

# ------------------------------------------------------------
# Hyper‑parameters
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("[train.py] FATAL: CUDA not available – training on CPU is far too slow.")
    exit(1)

BATCH_SIZE   = 64
BUFFER_CAP   = 60_000
GAMMA        = 0.9
LR           = 2.5e-4
TARGET_UPD   = 10_000       # steps between target network syncs
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 3_000_000    # linear decay steps
TOTAL_STEPS  = 5_000_000
SAVE_EVERY   = 500_000
WEIGHT_PATH  = "mario_dqn.pth"
PRELOAD      = False
WARMUP       = 2000         # minimum buffer size before learning starts

# ------------------------------------------------------------
# Environment helper
# ------------------------------------------------------------
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_r = 0.0
        for _ in range(self._skip):
            obs, r, done, info = self.env.step(action)
            total_r += r
            if done:
                break
        return obs, total_r, done, info


def make_env():
    env = make_mario("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=4)                   # 4 frames per action
    env = GrayScaleObservation(env, keep_dim=False)  # (H,W,1)
    env = ResizeObservation(env, 84)               # (84,84,1)
    env = FrameStack(env, 4)                       # (4,84,84,1)
    return env


def squeeze_obs(obs):
    arr = np.array(obs, dtype=np.uint8)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr

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
if PRELOAD:
    print("Loading previous weights from", WEIGHT_PATH)
    policy_net.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))

target_net = DQN(n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())

def sync_target():
    target_net.load_state_dict(policy_net.state_dict())

optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
# GPU-based replay buffer
storage = LazyTensorStorage(max_size=BUFFER_CAP, device=DEVICE)
replay_buf = TensorDictReplayBuffer(
    storage=storage,
    batch_size=BATCH_SIZE,
    sampler=RandomSampler(),
)

# initialize
eps   = EPS_START
state = squeeze_obs(env.reset())  # (4,84,84)
episode     = 0
cum_reward  = 0
log_f = open("log.txt", "w")
progress = tqdm(range(TOTAL_STEPS), desc="Training", miniters=100)

for step in progress:
    # ---------- act ----------
    if torch.rand(1, device=DEVICE).item() < eps:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            s = torch.tensor(state, device=DEVICE, dtype=torch.float32).unsqueeze(0).div_(255)
            action = int(policy_net(s).argmax(1).item())

    # ---------- env step ----------
    next_state, reward, done, _ = env.step(action)
    next_state = squeeze_obs(next_state)
    cum_reward += reward

    # push to GPU buffer as TensorDict
    replay_buf.add(TensorDict({
        "state":       torch.tensor(state, device=DEVICE, dtype=torch.uint8),
        "action":      torch.tensor(action, device=DEVICE, dtype=torch.long),
        "reward":      torch.tensor(reward, device=DEVICE, dtype=torch.float32),
        "next_state":  torch.tensor(next_state, device=DEVICE, dtype=torch.uint8),
        "done":        torch.tensor(done, device=DEVICE, dtype=torch.bool)
    }, batch_size=[]))
    state = next_state

    # ε‑decay
    eps = max(EPS_END, EPS_START - step / EPS_DECAY)

    # ---------- learn ----------
    if len(replay_buf) > WARMUP and step % 4 == 0:
        batch = replay_buf.sample()
        s_batch = batch["state"].float() / 255             # [B,4,84,84]
        a_batch = batch["action"].unsqueeze(-1)  # [B,1]
        r_batch = batch["reward"]            # [B]
        n_batch = batch["next_state"].float() / 255        # [B,4,84,84]
        d_batch = batch["done"].float()      # [B]

        # current Q
        q_pred = policy_net(s_batch).gather(1, a_batch).squeeze(-1)
        # double dqn target
        with torch.no_grad():
            next_actions = policy_net(n_batch).argmax(1, keepdim=True)
            q_next = target_net(n_batch).gather(1, next_actions).squeeze(-1)
            q_target = r_batch + GAMMA * q_next * (1.0 - d_batch)
        loss = nn.functional.smooth_l1_loss(q_pred, q_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(policy_net.parameters(), 1.0)
        optimizer.step()

    # ---------- target update ----------
    if step % TARGET_UPD == 0:
        sync_target()

    # ---------- episode end ----------
    if done:
        log_f.write(f"Episode {episode}\tReward {cum_reward}\tEpsilon {eps:.3f}\tSteps {step}\n")
        log_f.flush()
        episode += 1
        state = squeeze_obs(env.reset())
        cum_reward = 0

    # ---------- checkpoint ----------
    if step % SAVE_EVERY == 0 and step > 0:
        torch.save(policy_net.state_dict(), WEIGHT_PATH)
        print(f"[train.py] Saved checkpoint at step {step}")

# final save
torch.save(policy_net.state_dict(), WEIGHT_PATH)
log_f.close()
print("[train.py] Training complete – weights saved to", WEIGHT_PATH)
