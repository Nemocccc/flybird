import torch.nn.functional as F
import gym
from gym.envs.registration import register

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from aiflybird import FlappyBirdClient

import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy
import time
import datetime
import os
import io
import ray
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class ActorCritic(nn.Module):
    def __init__(self, state_dim,action_dim):
        super().__init__()

        self.critic = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
              )
        self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim),
              )

    def forward(self,x):
        logits=self.actor(x)
        v=self.critic(x)
        return logits,v

class Distributions():
    def __init__(self, action_space=None):
        self.action_space=action_space

    def argmax(self,logits):
        return int(torch.argmax(logits,dim=1).cpu().item())

    def sample(self, logits):
        distribution = Categorical(logits=logits)
        return int(distribution.sample().cpu().item())

    def entropy(self, logits):
        distribution = Categorical(logits=logits)
        return distribution.entropy().float()

    def logprob(self, logits, value_data):
        if torch.is_tensor(value_data):
            action=value_data
        else:
            action=torch.tensor(value_data)

        # print("logits.shape",logits.shape,"action",action.shape)

        distribution = Categorical(logits=logits)

        return distribution.log_prob(action).unsqueeze(1).float()


class Agent:
    def __init__(self, state_dim, action_dim, is_training_mode):
        self.is_training_mode = is_training_mode
        self.device = torch.device('cpu')


        self.distributions = Distributions(action_dim)
        self.actor_model = ActorCritic(state_dim, action_dim).to(self.device)

        if is_training_mode:
            self.actor_model.train()
        else:
            self.actor_model.eval()



    def act(self,state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
            logit,value = self.actor_model(state)

            # We don't need sample the action in Test Mode
            # only sampling the action in Training Mode in order to exploring the actions
            if self.is_training_mode:
                # Sample the action
                action = self.distributions.sample(logit)
            else:
                action = self.distributions.argmax(logit)

            logprob=self.distributions.logprob(logit,action)


        return action,logprob.flatten(),value.flatten()

    def set_weights(self, weights):
        if weights is not None:
            try:
                self.actor_model.load_state_dict(weights,strict=False)
            except Exception as e:
                print(f"set weight result {e}")

    def load_weights(self,path='agent.pth'):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.actor_model.load_state_dict(state_dict)
        except Exception as e:
            # 文件加载失败时的处理
            print(f"Error loading weights from file: {e}")
            return

def run_episode(env,agent):
    state=env.reset()
    done = False
    eps_time=0
    total_reward=0
    while not done:

        action, logprob, value = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        eps_time += 1
        total_reward += reward

        state = next_state

        if done:
            states = env.reset()

            print('t_reward: {} \t time: {} \t '.format( total_reward,eps_time))


env = FlappyBirdClient(port=11111)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent=Agent(state_dim,action_dim,False)

agent.load_weights("agent.pth")

run_episode(env,agent)



