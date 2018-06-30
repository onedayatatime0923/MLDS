from agent_dir.agent import Agent
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import scipy.misc
from torch.autograd import *
from torch.distributions import Bernoulli , Categorical
import pylab as plt 
import gym
from itertools import count
from collections import deque

writer = SummaryWriter('runs/exp_policy') 

class CNN(nn.Module):
    def __init__(self,action_space):
        super(CNN, self).__init__()
        self.affine1 = nn.Linear(80*80, 256)
        self.affine2 = nn.Linear(256, action_space)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

def prepro(o,image_size=[80,80]):
    o = o / 255
    o = o[35:195]  # crop
    o = o[::2, ::2, 0]  # downsample by factor of 2
    o[o == 144] = 0  # erase background (background type 1)
    o[o == 109] = 0  # erase background (background type 2)
    o[o != 0] = 1  # everything else (paddles, ball) just set to 1
    return o.astype(np.float).ravel().reshape(1,-1)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """ 
        super(Agent_PG,self).__init__(env)
 
        ##################
        # YOUR CODE HERE #
        ##################
        self.env = gym.make("Pong-v0")
        self.gamma = 0.99  
        self.num_episode = 20000
        self.batch_size = 1
        self.action_space = self.env.action_space.n
        self.model = CNN(self.action_space).cuda() 

        if args.test_pg: 
            print('loading trained model')
            self.model.load_state_dict(torch.load('policy_gradient_model.pt'))
            self.model.eval() 
 
        self.optimizer = optim.Adam(self.model.parameters(),lr=1e-3)

        self.seed = 543 
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)   

    def init_game_setting(self):
        """ 
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.test_last_observation = prepro(self.env.reset()) 
  
    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)  # Convert from numpy
        probs = self.model(Variable(state).cuda())  # Make Variable enables gradients
        m = Categorical(probs)  # categorical distribution based on probablities
        action = m.sample()  # Sample from distribuiton
        self.model.saved_log_probs.append(m.log_prob(action))  # Save Log probablities
        return action.data[0]  # Return action sampled

    def finish_episode(self): 
        R = 0  # Discounted future rewards, set to 0 each time called
        policy_loss = []  # new loss calc for each game
        rewards = []  # New discounted rewards calc for each game
        for r in self.model.rewards[::-1]:  # Reverse loop through actual rewards
            if r == 0 : R = 0
            R = r + self.gamma * R  # Create R based on discounted future reward est
            rewards.insert(0, R)  # Insert R at the front for rewards
        rewards = torch.Tensor(rewards).cuda()  # Convert to pytorch tensor 
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)  # Standardized to unit normal

        optimizer.zero_grad()  # Zero gradients before backward pass 

        for log_prob, reward in zip(self.model.saved_log_probs, rewards):  # loop through log_prob & rewards
            loss = -log_prob * reward
            loss.backward() 
          
        self.optimizer.step()  # Make changes to weights
        
        self.model.rewards = []
        self.model.saved_log_probs = []

    def train(self):
        best_average_reward = -100
        running_reward = deque(maxlen=30)
        prev_x = None  # used in computing the difference frame

        for i_episode in count(1):
            reward_sum = 0
            done = False
            state = env.reset()

            while not done:  # Don't infinite loop while learning
                # preprocess the observation, set input to network to be difference image
                if i_episode % 20 == 0:
                    env.render()
                cur_x = prepro(state)
                x = cur_x - prev_x if prev_x is not None else np.zeros(size_in)
                prev_x = cur_x
                action = self.select_action(x)
                state, reward, done, _ = env.step(action)
                policy.rewards.append(reward)
                reward_sum += reward

            running_reward.append(reward_sum)

            self.finish_episode()
            
            average_reward = np.mean(running_reward)

            if average_reward > best_average_reward :
                best_average_reward = average_reward
                torch.save(self.model.state_dict(), 'policy_gradient_model.pt')

            print('Episode: {}\tScore: {}\tAverage score: {}'.format(i_episode, reward_sum, average_reward ) )
            writer.add_scalar('30 average reward', average_reward , i_episode) 