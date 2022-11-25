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
# import gym_super_mario_bros
import time
import sys
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
total_reward = 0

def act(state):
    global count_step
    
    count_step = count_step + 1
    if count_step <= 100:
        return np.random.randint(3)
    action_idx = net(torch.tensor([state]), 'online')
    return torch.argmax(action_idx).item()
    
def env_step(act, current_state, current_speed=None):
    global total_reward
    
    result = env.step(np.array(act))
    # time_limit = result(1)[3]['TimeLimit.truncated']
    next_state = result[0][0]
    next_speed = result[0][1]
    # reward = (next_state * 0) + (speed * 100)
    # reward = 3 + (next_state * 1)
    # if current_state < next_state:
    #     reward = 1 + (next_state * 1)
    reward = 0
    reward = (reward + 1) if (current_state) < (next_state) else (reward + 0)
    reward = (reward + 1) if current_speed < next_speed else (reward + 0)
    reward = (1 + (next_state * 1)) + reward
    total_reward = total_reward + reward
    
    done = result[3]
    if done == True:
        done = 1
    else:
        done = 0
                      
    return next_state , reward, done, speed
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
        self.online = nn.Sequential(nn.Linear(input_dim, output_dim), )
        self.target = copy.deepcopy(self.online)
        
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        input = input.to("cuda")
        if model == "online":
            return self.online(input * 2)
        elif model == "target":
            return self.target(input * 2)
#%%
net = (CarNet(1, 3).float()).to("cuda")
# d = net(torch.tensor([2.5]), 'online')
# d
# (0 + (1) * 0.9 * 0)
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
    # return (reward + (1 - done.float()) * gamma * next_Q).float()
    reward = reward.to("cuda")
    return (reward * gamma * next_Q).float()
#%%
optimizer = torch.optim.Adam(net.parameters(), lr=0.00025)
loss_fn = torch.nn.SmoothL1Loss()

ls = 500
def update_Q_online(td_estimate, td_target):
    # global optimizer
    # global loss_fn
    global ls
    loss = loss_fn(td_estimate, td_target)
    # if loss.item() < ls:
    #     ls = loss.item()
    #     print(ls)
    # print(td_estimate[0])
    # print(td_target[0])
    # print('*' * 10)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def sync_Q_target():
    net.target.load_state_dict(net.online.state_dict())
#%%   
burnin = 500
learn_every = 3 
sync_every = 1e4 
sync_every = 500
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
episodes = 40000
for e in range(episodes):
    state = env.reset()[0][0]
    speed = env.reset()[0][1]
    while True:
        action = act(state)
        next_state, reward, done, next_speed = env_step(action, state, speed)
        cache(state, next_state, action, reward, done)
        q, loss = learn()
        state = next_state
        speed = next_speed
        if done == 1:
            break
print("Done")
#%%
i = 0
state1 = env.reset()[0][0]
while True:
    i = i + 1
    act1 = net(torch.tensor([state1]), 'target')
    actidx = torch.argmax(act1).item()
    state1, spd = env.step(np.array(actidx))[0][0:2]
    print(state1, spd, actidx)
    if i == 205:
        break
# torch.tensor([next_state])
# net(state[0], model="online")[np.arange(0, batch_size), action[0]]
#%%
env.reset()[0]
#%%
env.step(2)[3]
#%%
net(torch.tensor([state1]), 'target')