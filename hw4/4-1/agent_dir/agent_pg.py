from agent_dir.agent import Agent
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/exp_policy') 

class CNN(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.fc1 = nn.Linear(6400, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, flatten):
        fc1 = self.relu(self.fc1(flatten))
        pred = self.sigmoid(self.fc2(fc1))

        return pred

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
    resized = scipy.misc.imresize(y, image_size) 
    return np.expand_dims(resized.astype(np.float32),axis=2).ravel()


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
        self.gamma = 0.99  

        self.model = CNN()
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.model.load_state_dict(torch.load(''))
            self.model.eval() 
        self.model.cuda() 
        self.optimizer = optim.Adam(model.parameters(),lr=0.0001)


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
        best_average_reward = -100000
        train_last_observation = None  # used in computing the difference frame
        observation = self.env.reset()
        img,act,rew = [], [], []
        record_reward = []
        reward_sum = 0
        episode_number = 0

        self.model.train()

        while True: 
            # preprocess the observation, set input to network to be difference image
            current_observation = prepro(observation)
            if train_last_observation is not None : x = current_observation - train_last_observation
            else : x = current_observation
            train_last_observation = current_observation

            # forward the policy network and sample an action from the returned probability

            pred = self.predict_action(x).data.cpu().numpy()[0]
            action = 2 if 0.5 < pred else 3  # roll the dice!

            # record various intermediates (needed later for backprop)
            img.append(x)  # observation
            y = 1 if action == 2 else 0  # a "fake label"
            act.append(y)

            # step the environment and get new measurements
            observation, reward, done, info = self.env.step(action)
            reward_sum += reward
            rew.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

            if done:  # an episode finished
                episode_number += 1

                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                episode_img = np.concatenate(img,axis=0)
                episode_reward = np.concatenate(rew,axis=0)
                episode_action = np.concatenate(act,axis=0)
                img,act,rew = [], [], [] # reset array memory


                # compute the discounted reward backwards through time
                discounted_episode_reward = self.discount_rewards(episode_reward, self.gamma)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                #discounted_episode_reward -= np.mean(discounted_episode_reward)
                #discounted_episode_reward /= np.std(discounted_episode_reward)
 
                states = Variable(torch.from_numpy(episode_img), requires_grad=False).cuda().float()
                action = Variable(torch.from_numpy(episode_action), requires_grad=False).cuda().float()
                rewards = Variable(torch.from_numpy(episode_reward), requires_grad=False).cuda().float()
                self.optimizer.zero_grad()

                probability = self.model(states)
                # loss = torch.sum((action - probability) * rewards) / episode_action.shape[0]
                loss = F.binary_cross_entropy(probability, action, rewards)
                loss.backward()
                self.optimizer.step()


                # boring book-keeping
                record_reward.append(reward_sum)
                if len(record_reward) > 30 : average_reward = sum(record_reward[-30:]) / 30
                else : average_reward = sum(record_reward) / len(record_reward)
                print('resetting env. episode reward was %f. running mean: %f' % (reward_sum, average_reward))
                writer.add_scalar('Average_reward', average_reward , len(record_reward))

                if average_reward > best_average_reward : 
                    torch.save(self.model.state_dict(), '')
                    print("model saved ")

                reward_sum = 0
                observation = self.env.reset()  # reset env
                train_last_observation = None

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

        pred = self.predict_action(observation_delta).data.cpu().numpy()[0]
        action = 2 if 0.5 < pred else 3
        return action

    # each step sum reward for all steps which it has influenced
    def discount_rewards(self, rewards, discount_factor):
        discount_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * discount_factor + rewards[t]
            discount_rewards[t] = running_add
        return discount_rewards

    def predict_action(self, observation): 
        observation = Variable(torch.from_numpy(observation), requires_grad=False).cuda().float()
        probability = self.model(observation)# replace with torch network
        return probability