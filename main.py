import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import time
#%%
env = gym.make("MountainCar-v0")
env.reset()
#%%
(-0.9) + (-0.9) * 3
#%%
# d = act(env.reset()[0])
# env.step(np.array(d))
# torch.tensor([2])
# env.step(np.array(1))
# random.sample(memory, batch_size)[0]
#%%
count_step = 0

def act(state):
    global count_step
    
    count_step = count_step + 1
    if count_step <= 1000:
        return np.random.randint(3)
    action_idx = net(torch.tensor([state]), 'online')
    return torch.argmax(action_idx).item()
    
def env_step(act):
    result = env.step(np.array(act))
    # time_limit = result(1)[3]['TimeLimit.truncated']
    next_state = result[0][0]
    speed = result[0][1]
    reward = next_state + next_state * 10
    done = result[2]
    if done == True:
        done = 1 
    else:
        done = 0
                      
    return next_state, reward, done
#%%
memory = deque(maxlen=100000)
batch_size = 32

def cache(state, next_state, action, reward, done):
    state = torch.tensor([state])
    action = torch.tensor([action])
    next_state = torch.tensor([next_state])
    reward = torch.tensor([reward])
    done = torch.tensor([done])
    memory.append((state, next_state, action, reward, done,))
#%%
def recall():
    batch = random.sample(memory, batch_size)
    state, next_state, action, reward, done = map(torch.stack, zip(*batch))
    return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
#%%
class CarNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Linear(input_dim, output_dim),)

        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
#%%
net = CarNet(1, 3).float()
d = net(torch.tensor([2.5]), 'online')
print(torch.argmax(d))
d
#%%
gamma = 0.9

def td_estimate(state, action):
    global net
    current_Q = net(state, model="online")[np.arange(0, batch_size), action] 
    return current_Q

@torch.no_grad()
def td_target(reward, next_state, done):
    next_state_Q = net(next_state, model="online")
    best_action = torch.argmax(next_state_Q, axis=1)
    next_Q = net(next_state, model="target")[np.arange(0, batch_size), best_action] 
    return (reward + (1 - done.float()) * gamma * next_Q).float()
#%%
optimizer = torch.optim.Adam(net.parameters(), lr=0.00025)
loss_fn = torch.nn.SmoothL1Loss()

def update_Q_online(td_estimate, td_target):
    global optimizer
    global loss_fn
    
    loss = loss_fn(td_estimate, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def sync_Q_target():
    net.target.load_state_dict(net.online.state_dict())
#%%   
burnin = 100
learn_every = 3 
sync_every = 1e4 
td_estimate2 = 0

def learn():
    global count_step
    global sync_every
    global save_every
    global learn_every
    global td_estimate2

    if count_step % sync_every == 0:
        sync_Q_target()

    if count_step < burnin:
        return None, None

    if count_step % learn_every != 0:
        return None, None

    state, next_state, action, reward, done = recall()
    td_est = td_estimate(state, action)
    td_tgt = td_target(reward, next_state, done)
    loss = update_Q_online(td_est, td_tgt)
    return (td_est.mean().item(), loss)
#%%
episodes = 80000
for e in range(episodes):
    state = env.reset()[0]
    while True:
        action = act(state)
        next_state, reward, done = env_step(action)
        cache(state, next_state, action, reward, done)
        q, loss = learn()
        
        state = next_state
        if done == 1:
            break
print("Done")
#%%
env.reset()[0]
#%%
i = 0
state1 = env.reset()[0]
while True:
    i = i + 1
    act1 = net(torch.tensor([state1]), 'target')
    actidx = torch.argmax(act1).item()
    state1 = env.step(np.array(actidx))[0][0]
    print(state1, actidx)
    if i == 300:
        break
