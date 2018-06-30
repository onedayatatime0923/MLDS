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

writer = SummaryWriter('runs/exp_policy') 

class CNN(nn.Module):
    def __init__(self,action_space):
        super(CNN, self).__init__()
        self.action_space = action_space
        self.affine1 = nn.Linear(80*80, 200)
        self.affine2 = nn.Linear(200, self.action_space)
 
    def forward(self, x):
        #print(self.action_space)
        #print(x.size())
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
'''
def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to graactcale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Graactcale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    y = np.dot(y[...,:3], [0.2126, 0.7152, 0.0722]) / 255
    #print(y)
    resized = scipy.misc.imresize(y, image_size) 
    return np.expand_dims(resized.astype(np.float32),axis=2).ravel()  
'''

def prepro(I):
    I = I / 255
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel().reshape(1,-1)

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
        self.model = CNN(self.action_space)
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model.load_state_dict(torch.load('policy_gradient_model.pt'))
            self.model.eval() 
        self.model.cuda() 
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.0001)
        torch.manual_seed(87)

    def init_game_setting(self):
        """ 
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.test_last_observation = prepro(self.env.reset()) 

    def train(self):  
            
        self.model.train()
        state_pool = []
        action_pool = []
        reward_pool = []
        steps = 0
        record_reward_list = []
        best_average_reward = - 1000

        for e in range(self.num_episode):

            observation = self.env.reset() 
            train_last_observation = None 
            record_reward = 0
            self.env.render()

            while True: 
                
                #preprocess the observation, set input to network to be difference image 

                current_observation = prepro(observation)  
                if train_last_observation is not None : x = current_observation - train_last_observation
                else : x = current_observation
                train_last_observation = current_observation

                x = torch.from_numpy(x).float()
                x = Variable( x , requires_grad = True ).cuda() 

                probs = self.model(x)
                m = Categorical(probs)
                action = m.sample()  

                state_pool.append(x)
                action_pool.append(float(action))

                action = action.cpu().data.numpy().astype(int)[0] 
                #print(action)
                observation, reward, done, info = self.env.step(action)
                self.env.render()
                record_reward += reward

                if done : reward = 0  
                reward_pool.append(reward)  

                steps += 1 
                 
                if done : 
                    record_reward_list.append(record_reward)
                    break

            # an episode finished 

            discount_rewards = np.zeros_like(reward_pool)
            running_add = 0
            for t in reversed(range(steps)): 
                if reward_pool[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
                running_add = running_add * self.gamma + reward_pool[t]
                discount_rewards[t] = running_add 
 
            # Normalize reward

            reward_mean = np.mean(discount_rewards)
            reward_std = np.std(discount_rewards)
            for i in range(steps):
                discount_rewards[i] = (discount_rewards[i] - reward_mean) / reward_std

            print(discount_rewards)

            # Gradient Desent

            self.optimizer.zero_grad()
 
            for i in range(steps):
                state = state_pool[i]
                action = Variable( torch.FloatTensor([action_pool[i]]) , requires_grad = True ).long().cuda()
                reward = discount_rewards[i]

                probs = self.model(state)
                m = Categorical(probs) 
                loss = -m.log_prob(action) * reward   
                loss.backward()    

            self.optimizer.step() 

            # print status

            if len(record_reward_list) > 30 :  
                average_reward = sum(record_reward_list[-30:]) / 30
            else : 
                average_reward = sum(record_reward_list) / len(record_reward_list)

            if average_reward > best_average_reward :
                best_average_reward = average_reward
                torch.save(self.model.state_dict(), 'policy_gradient_model.pt')

            print(str( e / self.batch_size ) + ' update reward = %f || 30 average reward =  %f' % (record_reward, average_reward))

            # tensorboard

            writer.add_scalar('30 average reward', average_reward , e) 

            # reset

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0 


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if test:
            self.model.eval()

        observation = prepro(observation)
        observation_delta = observation - self.test_last_observation
        self.test_last_observation = observation

        probs = self.model(x)
        m = Categorical(probs)
        action = m.sample()   
        action = action.data.numpy().astype(int)[0] + 2 

        return action 