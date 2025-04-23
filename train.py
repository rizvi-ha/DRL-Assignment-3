import gym
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from gym_super_mario_bros import make as make_mario
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from tqdm import tqdm

# ---------- hyper-parameters ----------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE == "cpu":
    print("Warning: running on CPU, which is slow")
    print("Please use GPU for training.")
    exit(1)

BATCH_SIZE   = 128
BUFFER_CAP   = 100_000
GAMMA        = 0.99
LR           = 1e-4
TARGET_UPD   = 10_000        # steps
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 8e5           # steps
TOTAL_STEPS  = 2_000_000
SAVE_EVERY   = 100_000
WEIGHT_PATH  = "mario_dqn.pth"
# --------------------------------------

# Open log file
log_file = open("log.txt", "w")

# ==== util ====
import cv2
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs

Transition = namedtuple("Transition", "state action reward next_state done")

class ReplayBuffer:
    def __init__(self, cap): self.buf, self.cap = deque(maxlen=cap), cap
    def push(self,*args): self.buf.append(Transition(*args))
    def sample(self,k):
        batch = np.random.choice(len(self.buf),k,replace=False)
        data  = [self.buf[i] for i in batch]
        s,a,r,n,d = map(np.array, zip(*data))
        s = torch.as_tensor(s, dtype=torch.float32, device=DEVICE) / 255.
        n = torch.as_tensor(n, dtype=torch.float32, device=DEVICE) / 255.
        a = torch.as_tensor(a, dtype=torch.int64,  device=DEVICE)
        r = torch.as_tensor(r, dtype=torch.float32, device=DEVICE)
        d = torch.as_tensor(d, dtype=torch.float32, device=DEVICE)
        return s,a,r,n,d
    def __len__(self): return len(self.buf)

class DQN(nn.Module):
    def __init__(self, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,32,8,4), nn.ReLU(),
            nn.Conv2d(32,64,4,2), nn.ReLU(),
            nn.Conv2d(64,64,3,1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7,512), nn.ReLU(),
            nn.Linear(512,n_act))
    def forward(self,x): return self.net(x)

# ==== main training ====
env = JoypadSpace(make_mario("SuperMarioBros-v0"), COMPLEX_MOVEMENT)
n_actions = env.action_space.n
policy, target = DQN(n_actions).to(DEVICE), DQN(n_actions).to(DEVICE)
target.load_state_dict(policy.state_dict())
optimiser = optim.Adam(policy.parameters(), lr=LR)
replay = ReplayBuffer(BUFFER_CAP)

eps = EPS_START
state_q = deque(maxlen=4)    # for stacking

def stack_state(frame_deque):
    return np.stack(frame_deque, axis=0).astype(np.uint8)

obs = env.reset()
obs = preprocess(obs)
for _ in range(4): state_q.append(obs)          # warm-start
state = stack_state(state_q)
episode = 0
total_reward = 0

pbar = tqdm(range(TOTAL_STEPS), desc="Training")
for step in pbar:
    # epsilon-greedy
    if np.random.random() < eps:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)/255.
            action = int(policy(s).argmax(1).item())

    next_obs, reward, done, info = env.step(action)
    total_reward += reward
    next_proc = preprocess(next_obs)
    state_q.append(next_proc)
    next_state = stack_state(state_q)

    replay.push(state, action, reward, next_state, done)
    state = next_state
    eps = max(EPS_END, EPS_START - step / EPS_DECAY)

    # learn
    if len(replay) > 20_000 and step % 4 == 0:
        s,a,r,n,d = replay.sample(BATCH_SIZE)
        q = policy(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = target(n).max(1)[0]
            target_q = r + GAMMA * q_next * (1 - d)
        loss = nn.functional.smooth_l1_loss(q, target_q)
        optimiser.zero_grad(); loss.backward(); optimiser.step()

    # target network update
    if step % TARGET_UPD == 0:
        target.load_state_dict(policy.state_dict())

    # reset episode
    if done:
        obs = env.reset()
        state_q.clear()
        frame = preprocess(obs)
        for _ in range(4): state_q.append(frame)
        state = stack_state(state_q)

        # log
        log_file.write(f"Episode {episode} | Reward: {total_reward}\n")
        log_file.flush()

        episode += 1
        pbar.set_description(f"Ep {episode}")
        pbar.set_postfix(reward=total_reward, refresh=False)
        total_reward = 0

    # checkpoint
    if step % SAVE_EVERY == 0 and step > 0:
        torch.save(policy.state_dict(), WEIGHT_PATH)
        print(f"Saved checkpoint at {step} steps")

torch.save(policy.state_dict(), WEIGHT_PATH)
log_file.close()
print("Training complete → mario_dqn.pth saved")
