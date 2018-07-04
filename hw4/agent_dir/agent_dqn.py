
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter 
import random, copy, time, math, sys, os
from agent_dir.agent import Agent
from atari_wrapper import make_wrap_atari

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
        ##################
        # YOUR CODE HERE #
        ##################

        self.env = env
        self.state_dim = env.get_observation_space().shape
        self.action_dim = env.get_action_space().n
        self.current_model = QNetwork(self.state_dim, self.action_dim).cuda()
        self.target_model = copy.deepcopy(self.current_model)
        #self.optimizer = torch.optim.RMSprop(self.current_model.parameters(), lr=args.learning_rate, eps=1E-6, weight_decay=0.9, momentum=0)
        #self.optimizer = torch.optim.RMSprop(self.current_model.parameters(), lr=args.learning_rate)
        self.optimizer = torch.optim.Adam(self.current_model.parameters(), lr= args.learning_rate)
        self.buffer = ReplayBuffer( env, args)

        self.args = args
        self.tb_setting('./runs/{}'.format(args.tensorboard_dir))
    def tb_setting(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for f in os.listdir(directory): 
            os.remove('{}/{}'.format(directory,f))
        self.writer = SummaryWriter(directory)
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass
    def train(self, print_every=3000, running_n = 30):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.current_model.train()
        criterion = nn.MSELoss()
        explore_epsilon = np.linspace(self.args.epsilon[0],self.args.epsilon[1],self.args.explore_step)
        exploitation_epsilon = np.ones((self.args.exploitation_step,))*self.args.epsilon[1]
        epsilon = np.hstack((explore_epsilon,exploitation_epsilon))
        start= time.time()
        
        episode_last = 0
        loss_batch = 0
        for e in range(self.args.explore_step + self.args.exploitation_step):
            if (e+1) % self.args.target_update_step ==0:
                self.target_model = copy.deepcopy(self.current_model)
            trajectory = self.buffer.collect_data(self.current_model, epsilon[e])
            state= Variable(torch.cat([i[0] for i in trajectory],0).cuda())
            action = Variable(torch.cat([i[1] for i in trajectory],0).cuda()).unsqueeze(1)
            state_n= Variable(torch.cat([i[2] for i in trajectory],0).cuda())
            reward = Variable(torch.FloatTensor([[i[3]] for i in trajectory]).cuda())
            done = Variable(torch.FloatTensor([[i[4]] for i in trajectory]).cuda())
            #print(state.size())
            #print(action.size())
            #print(state_n.size())
            #print(reward.size())
            #print(done.size())
            #print(done)
            #input()

            state_value= torch.gather(self.current_model(state),1, action)
            expected_value = torch.max(self.target_model(state_n).detach(),1)[0].unsqueeze(1) *( 1-done )* ( self.args.gamma)+ reward
            #print(reward)
            #print(state_value)
            #print(expected_value)

            loss = criterion(state_value, expected_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_batch += float(loss)
            print('\rTrain Step: {} | Loss: {:.4f} | Time: {}  '.format(
                e + 1,  float(loss) ,
                self.timeSince(start, (e - ((e-1)// print_every)* print_every)/ print_every )),end='')
            sys.stdout.flush()
            # record
            reward_ave = sum(self.buffer.reward[-running_n-1:-1]) / running_n
            self.writer.add_scalar('Train Reward', reward_ave, e)
            if (e+1) % print_every ==0:
                reward_ave = sum(self.buffer.reward[episode_last:-1]) / (len(self.buffer.reward)-1 - episode_last)
                print('\rTrain Step: {} | Episode: {:.0f} Total Episode: {:.0f} | Average Reward: {:.2f} | Loss: {:.4f} | Time: {}  '.format(
                    e + 1,  len(self.buffer.reward)-1- episode_last, len(self.buffer.reward)-1 , reward_ave, loss_batch/ print_every,
                    self.timeSince(start, 1)))
                episode_last = len(self.buffer.reward)-1
                loss_batch = 0
                self.test((e+1)// print_every)
                start= time.time()
    def test(self, epoch=0):
        self.current_model.eval()
        done = False
        state=np.array(self.env.reset())
        rewards = 0.0
        cnt = 0
        while not done:
            x = Variable(torch.FloatTensor(state).unsqueeze(0).cuda())
            action= torch.max(self.current_model(x),1)[1].cpu().data
            state_n , reward , done, _ = self.env.step(int(action))
            #if done: print(action_softmax)
            
            state = np.array(state_n)
            rewards += reward
            cnt += 1
            if( cnt >self.args.test_max_step):
                break
        print('======[testing score]======')
        print('reward: ', rewards)
        print('len', cnt)
        self.writer.add_scalar('Test Reward', rewards, epoch)
        self.writer.add_scalar('Test Length', cnt, epoch)
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        return self.env.get_random_action()
    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))
    def asMinutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

class QNetwork(nn.Module):
    def __init__(self, input_size, action_dim):
        super(QNetwork, self).__init__()
        self.LeakyReLU= nn.LeakyReLU(0.2, inplace= True)
        self.SELU = nn.SELU()
        # input size. 84 x 84 x 4
        self.cnn= nn.Sequential(
            # state size. 4 x 84 x 84
            nn.BatchNorm2d(input_size[2]),
            nn.Conv2d( input_size[2], 32, 8, 4, 2, bias=True),
            # state size. 32 x 21x 21
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d( 32, 64, 4, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            # state size. 64 x 10 x 10
            nn.ReLU(),
            nn.Conv2d( 64, 64 , 3, 1, 1, bias=True),
            nn.BatchNorm2d(64),
            # state size. (hidden_size*4) x 10 x 10
            nn.ReLU())
        self.linear = nn.Sequential(
            nn.Linear( 64* (input_size[0]// 8) * (input_size[1]// 8), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512 , action_dim),
            nn.BatchNorm1d(action_dim),
            nn.ReLU())
        self._initialize_weights()
    def forward(self,x ):
        # input is batch_size x 84 x 84 x 4
        x = x.permute(0,3,1,2).contiguous()
        x = self.cnn(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class ReplayBuffer():
    def __init__(self, env, args):
        # for game
        self.env = env
        self.state= torch.FloatTensor(self.env.reset()).unsqueeze(0)
        self.done = False

        self.buffer=[]

        # parameter
        self.action_dim = env.get_action_space().n
        self.buffer_size = args.buffer_size
        self.step = args.current_update_step
        self.batch_size = args.batch_size

        self.reward = [0]
    def collect_data(self, model, epsilon):
        model.eval()
        for s in range(self.step):
            if (self.done):
                self.state= torch.FloatTensor(self.env.reset()).unsqueeze(0)
                self.done = False
                self.reward.append(0)
            x = Variable(self.state.cuda())
            action = self.epsilon_greedy(model(x), epsilon)
            observation, reward, done, _ = self.env.step(int(action))
            reward_clamp = min(1, max(-1, reward))
            state_n = torch.FloatTensor(observation).unsqueeze(0)
            self.buffer.append([self.state, action, state_n, reward_clamp, done])

            self.reward[-1] += reward

            self.state= state_n
            self.done= done
        self.buffer= self.buffer[-self.buffer_size:]

        batch_size = min(self.batch_size, len(self.buffer))
        result = random.sample(self.buffer, batch_size)
        #print(len(self.buffer))
        return result
    def epsilon_greedy(self, action_value, epsilon):
        if random.random() < epsilon:
            #print(torch.max(action_value,1)[1].cpu().data.size())
            #print(torch.LongTensor([random.randint(0,self.action_dim-1)]).size())
            return torch.LongTensor([random.randint(0,self.action_dim-1)])
        else:
            #print(torch.max(action_value,1)[1].cpu().data.size())
            #print(torch.LongTensor([random.randint(0,self.action_dim-1)]).size())
            return torch.max(action_value,1)[1].cpu().data
