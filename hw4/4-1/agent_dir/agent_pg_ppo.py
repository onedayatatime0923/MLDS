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

writer = SummaryWriter('runs/exp_policy_v3') 

class CNN(nn.Module):
    def __init__(self,action_space):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(80*80, 256)
        self.fc2 = nn.Linear(256, action_space)

        self.saved_log_probs = []
        self.grad_log_probs = [] 
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action = self.fc2(x)
        return F.softmax(action, dim=1)
   
def prepro(o,image_size=[80,80]): 
    o = o[35:195]  # crop
    o = o[::2, ::2, 0]  # downsample by factor of 2
    o[o == 144] = 0  # erase background (background type 1)
    o[o == 109] = 0  # erase background (background type 2)
    o[o != 0] = 1  # everything else (paddles, ball) just set to 1
    return o.astype(np.float).ravel()

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
        #self.env = gym.make("Pong-v0")
        self.gamma = 0.99    
        self.action_space = self.env.action_space.n
        self.model = CNN(self.action_space).cuda() 

        if args.test_pg: 
            print('loading trained model')
            self.model.load_state_dict(torch.load('policy_gradient_model.pt'))
            self.model.eval() 
 
        self.optimizer = optim.Adam(self.model.parameters(),lr=1e-3)
 
        torch.manual_seed(543)
        self.env.seed(11037)   

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
        probs = self.model(Variable(state,requires_grad=True).cuda())  # Make Variable enables gradients
        m = Categorical(probs)  # categorical distribution based on probablities
        action = m.sample()  # Sample from distribuiton
        self.model.saved_log_probs.append(m.log_prob(action))  # Save Log probablities
        
        #m.log_prob(action).backward(retain_graph=True) 
        #self.model.grad_log_probs.append([x.grad.data.cpu().numpy() for x in self.model.parameters()])
        #self.model.zero_grad()
        
        return action.data[0]  # Return action sampled

    def finish_episode(self): 
        discounted_reward = 0   
        rewards = []   
        for r in self.model.rewards[::-1]:   
            if r != 0 : discounted_reward = 0
            discounted_reward = r + self.gamma * discounted_reward  
            rewards.insert(0, discounted_reward)  
        rewards = torch.Tensor(rewards)  
        self.model.rewards = []  
        rewards.cuda()
        self.optimizer.zero_grad()   
        for log_prob, reward in zip(self.model.saved_log_probs, rewards):  # loop through log_prob & rewards
            loss = -log_prob * reward 
            loss.backward()  
        self.optimizer.step()    
        self.model.saved_log_probs = []

    def train(self):
        best_average_reward = -100
        running_reward = deque(maxlen=30)
        prev_x = None  # used in computing the difference frame

        for i_episode in count(1):
            reward_sum = 0
            done = False
            state = self.env.reset()

            while not done:  
                cur_x = prepro(state)
                x = cur_x - prev_x if prev_x is not None else np.zeros(80*80)
                prev_x = cur_x
                action = self.select_action(x)
                state, reward, done, _ = self.env.step(action)
                self.model.rewards.append(reward)
                reward_sum += reward

            running_reward.append(reward_sum)

            self.finish_episode()
            
            average_reward = np.mean(running_reward)

            print('Episode: {}\tScore: {}\tAverage score: {}'.format(i_episode, reward_sum, average_reward ) )
            writer.add_scalar('30 average reward', average_reward , i_episode) 
            writer.add_scalar('Episode average reward', reward_sum , i_episode) 

            if average_reward > best_average_reward :
                best_average_reward = average_reward
                torch.save(self.model.state_dict(), 'policy_gradient_model_v3.pt')
                print('model saved') 