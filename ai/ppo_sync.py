import torch.nn.functional as F
import gym
from gym.envs.registration import register

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from gym_examples import FlappyBirdClient

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





class PPO_Dataset(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.returns = []
        self.logprobs=[]
        self.values=[]

    def merge_exp(self,experiences):
        for exp in experiences:
            for key,value in exp.items():
                if key=="states":
                    self.states.extend(value)
                if key=="actions":
                    self.actions.extend(value)
                if key=="returns":
                    self.returns.extend(value)
                if key=="logprobs":
                    self.logprobs.extend(value)
                if key=="values":
                    self.values.extend(value)


    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype=np.float32), \
            np.array(self.actions[idx], dtype=np.float32),\
            np.array(self.returns[idx], dtype=np.float32),\
            np.array(self.logprobs[idx],dtype=np.float32),\
            np.array(self.values[idx],dtype=np.float32)

    def get_all(self):
        return self.states, self.actions, self.returns,self.logprobs,self.values

    def save_all(self, states, actions,  returns,logprobs,values):
        self.states = states
        self.actions = actions
        self.returns = returns
        self.logprobs=logprobs
        self.values=values

    def save_eps(self, state, action,Return,logprob,value):
        self.states.append(state)
        self.actions.append(action)
        if Return is not None:
            self.returns.append(Return)
        self.logprobs.append(logprob)
        self.values.append(value)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.returns[:]
        del self.logprobs[:]
        del self.values[:]


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





class Learner():
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip,
                 entropy_coef, vf_loss_coef,clip_coef,max_grad_norm,
                 minibatch, PPO_epochs, learning_rate):
        self.clip_coef=clip_coef
        self.max_grad_norm=max_grad_norm
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.minibatch = minibatch
        self.PPO_epochs = PPO_epochs
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.learner_model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = Adam(self.learner_model.parameters(), lr=learning_rate)


        self.distributions = Distributions(action_dim)

        if is_training_mode:
            self.learner_model.train()
        else:
            self.learner_model.eval()
            self.load_weights()


    # Loss for PPO
    def get_loss(self, logprobs, values, old_logprobs, old_values, returns,entropy):
        # Don't use old value in backpropagation
        Old_values = old_values.detach()

        # Finding the ratio (pi_theta / pi_theta__old):
        # logprobs = self.distributions.logprob(probs, actions)
        # Old_logprobs = self.distributions.logprob(old_probs, actions).detach()

        # print(logprobs.shape,old_logprobs.shape,entropy)

        Old_logprobs=old_logprobs.detach()

        # Getting general advantages estimator
        Returns = returns.detach()
        Advantages = (Returns-Old_values).detach()
        Advantages = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach()


        ratios = (logprobs - Old_logprobs).exp()

        pg_loss1 = -Advantages * ratios
        pg_loss2 = -Advantages * torch.clamp(ratios, 1-self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Getting entropy from the action probability
        dist_entropy = entropy.mean()

        # Getting critic loss by using Clipped critic value
        vpredclipped = Old_values + torch.clamp(values - Old_values, -self.value_clip,
                                                self.value_clip)  # Minimize the difference between old value and new value
        vf_losses1 = (Returns - values).pow(2) * 0.5  # Mean Squared Error
        vf_losses2 = (Returns - vpredclipped).pow(2) * 0.5  # Mean Squared Error
        critic_loss = torch.max(vf_losses1, vf_losses2).mean()

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = pg_loss+(critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef)
        return loss

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, returns,old_logprobs,old_values):

        # print(states.shape,actions.shape,returns.shape,old_logprobs.shape,old_values.shape)

        logits, values = self.learner_model(states)
        # old_probs, old_values = self.actor_old(states), self.critic_old(states)

        entropy=self.distributions.entropy(logits=logits).mean()
        logprobs=self.distributions.logprob(logits=logits,value_data=actions)
        # print(probs.shape,old_probs.shape,next_values.shape,values.shape)

        loss = self.get_loss(logprobs, values, old_logprobs, old_values, returns, entropy)


        # === Do backpropagation ===

        self.optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(self.learner_model.parameters(), max_norm=self.max_grad_norm, norm_type=2)

        self.optimizer.step()

        # === backpropagation has been finished ===

    # Update the model
    def update_ppo(self,train_dataset):
        if not self.is_training_mode:
            return

        batch_size = int(len(train_dataset) / self.minibatch)

        # print("batch_size=",batch_size)
        dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for states, actions,  returns,logprobs,values in dataloader:
                self.training_ppo(states.float().to(device), actions.float().to(device), \
                                  returns.float().to(device),logprobs.float().to(device),\
                                  values.float().to(device))


        # Copy new weights into old policy:
        # self.actor_old.load_state_dict(self.actor.state_dict())
        # self.critic_old.load_state_dict(self.critic.state_dict())

    def get_weights(self):
        self.learner_model.to("cpu")
        state_dict=self.learner_model.state_dict()
        self.learner_model.to("cuda")
        return state_dict

    def save_weights(self):
        if not self.is_training_mode:
            return
        torch.save(self.learner_model.state_dict(), 'agent.pth')

    def load_weights(self):
        try:
            state_dict = torch.load('agent.pth', map_location=self.device)
            self.learner_model.load_state_dict(state_dict)
        except Exception as e:
            # 文件加载失败时的处理
            print(f"Error loading weights from file: {e}")
            return

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

    def load_weights(self):
        try:
            state_dict = torch.load('agent.pth', map_location=self.device)
            self.actor_model.load_state_dict(state_dict)
        except Exception as e:
            # 文件加载失败时的处理
            print(f"Error loading weights from file: {e}")
            return




@ray.remote
class Runner():
    def __init__(self, env_name, lam,gamma,training_mode, render, n_update, tag):
        # self.env = gym.make(env_name)

        self.env = FlappyBirdClient()

        self.states = self.env.reset()

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.agent = Agent(self.state_dim, self.action_dim, training_mode)

        self.render = render
        self.tag = tag
        self.training_mode = training_mode
        self.n_update = n_update


        self.last_model_load_time = time.time()

        self.load_model_start_time= self.last_model_load_time

        self.load_model_cnt = 0

        self.lam=lam
        self.gamma=gamma

        self.total_reward=0
        self.i_episode=0
        self.eps_time=0


    def set_weights(self, weights):
        if weights is not None:
            try:
                self.agent.set_weights(weights)
            except Exception as e:
                print(f"set weight result {e}")

    def run_episode(self):

        # self.agent.load_weights()
        #
        # self.load_model_cnt += 1
        print(
            f"Process {self.tag} loaded new model, load cnt is {self.load_model_cnt},load time is {time.time() - self.load_model_start_time} !")
        self.last_model_load_time = time.time()  # 更新载入时间

        # self.agent.clear_all()
        gc.collect()

        # print("actor 3")

        dones = []
        rewards = []
        returns = []
        values = []
        actions = []
        states = []
        logprobs = []

        for _ in range(self.n_update):

            action, logprob, value = self.agent.act(self.states)
            next_state, reward, done, _ = self.env.step(action)

            self.eps_time += 1
            self.total_reward += reward

            # print("actor 4")

            # if self.training_mode:
            #     self.agent.save_eps(self.states.tolist(), action,None,logprob,value)

            states.append(list(self.states))
            actions.append(action)
            logprobs.append(logprob)
            values.append(value)
            rewards.append(reward)
            dones.append(float(done))

            self.states = next_state

            if self.render and self.tag == 0:
                self.env.render()

            if done:
                self.states = self.env.reset()
                self.i_episode += 1
                print('Episode {} \t t_reward: {} \t time: {} \t process no: {} \t'.format(self.i_episode, self.total_reward,
                                                                                           self.eps_time, self.tag))

                self.total_reward = 0
                self.eps_time = 0

        gae = 0
        for t in reversed(range(self.n_update)):
            if t == self.n_update - 1:
                _, _, next_value = self.agent.act(self.states)

            else:
                next_value = values[t + 1]
            done = dones[t]
            value = values[t]
            delta = rewards[t] + self.gamma * next_value * (1 - done) - value
            adv = gae = delta + self.gamma * self.lam * (1 - done) * gae
            Return = adv + value
            returns.insert(0, Return)



        samples =  {
        "states": states,
        "actions": actions,
        "returns": returns,
        "logprobs": logprobs,
        "values": values
        }

        # print(f"actor {self.tag} add data")
            # time.sleep(self.sleep_time)
        return samples


def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))


def main():
    ############## Hyperparameters ##############
    training_mode = True  # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    render = False  # If you want to display the image, set this to True. Turn this off if you run this in Google Collab

    n_episode = 100000  # How many episode you want to run

    clip_coef=0.2
    max_grad_norm=0.5
    policy_kl_range = 0.0008  # Recommended set to 0.03 for Continous
    policy_params = 20  # Recommended set to 5 for Continous
    value_clip = 1.0  # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef = 0.01  # How much randomness of action you will get. Because we use Standard Deviation for Continous, no need to use Entropy for randomness
    vf_loss_coef = 0.5  # Just set to 1

    n_agent = 4  # How many agent you want to run asynchronously
    n_update = 128  # How many episode before you update the Policy. Recommended set to 1024 for Continous
    minibatch = 4   # How many batch per update. size of batch = n_update / minibatch. Recommended set to 32 for Continous
    PPO_epochs = 4  # How many epoch per update. Recommended set to 10 for Continous

    gamma = 0.99  # Just set to 0.99
    lam = 0.95  # Just set to 0.95
    learning_rate = 2.5e-4  # Just set to 0.95
    #############################################

    env_name = 'CartPole-v1'
    # env = gym.make(env_name)
    env=FlappyBirdClient()

    # print(action_dim)
    # exit(0)
    train_dataset=PPO_Dataset()
    learner = Learner(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef,
                      vf_loss_coef,clip_coef,max_grad_norm,
                      minibatch, PPO_epochs, learning_rate)
    learner.load_weights()
    #############################################
    start = time.time()
    ray.init()


    # ModelPool=ModelPoolManager.remote(capacity=1)

    try:
        runners = [Runner.remote(env_name,lam,gamma, training_mode, render, n_update, i) for i in range(n_agent)]
        learner.save_weights()

        # ModelPool.add_model.remote(learner.get_weights())



            # time.sleep(1)

        for _ in range(1, n_episode + 1):

            experiences=[ray.get(runner.run_episode.remote()) for runner in runners]

            train_dataset.clear_memory()
            train_dataset.merge_exp(experiences)

            learner.update_ppo(train_dataset)

            for runner in runners:
                runner.set_weights.remote(learner.get_weights())

            learner.save_weights()

            gc.collect()

    except KeyboardInterrupt:
        print('\nTraining has been Shutdown \n')
    finally:
        ray.shutdown()

        finish = time.time()
        timedelta = finish - start
        print('Timelength: {}'.format(str(datetime.timedelta(seconds=timedelta))))


if __name__ == '__main__':
    main()