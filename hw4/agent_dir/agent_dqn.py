
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter 
import random, copy, time, math, sys, os
from agent_dir.agent import Agent

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
        self.current_model = QNetwork(self.state_dim, self.action_dim, args).cuda()
        self.target_model = copy.deepcopy(self.current_model)
        self.optimizer = torch.optim.RMSprop(self.current_model.parameters(),lr=args.learning_rate)
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
    def train(self, print_every=100):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        start= time.time()
        self.current_model.train()
        criterion = nn.MSELoss()
        epsilon = np.linspace(self.args.epsilon[0],self.args.epsilon[1],self.args.step_n)
        episodes = 0
        for e in range(self.args.step_n):
            if (e+1) % self.args.target_update_step ==0:
                self.target_model = copy.deepcopy(self.current_model)
            trajectory, episode = self.buffer.collect_data(self.current_model, epsilon[e])
            state= Variable(torch.cat([i[0] for i in trajectory],0).cuda())
            action = Variable(torch.cat([i[1] for i in trajectory],0).cuda())
            reward = Variable(torch.FloatTensor([i[2] for i in trajectory]).cuda())
            state_n= Variable(torch.cat([i[3] for i in trajectory],0).cuda())
            #print(state)
            #print(action)
            #print(reward)
            #print(state_n)
            #print(self.current_model(state))

            action_index = action.unsqueeze(1)
            state_value= torch.gather(self.current_model(state),1, action_index).squeeze(1)
            expected_value = torch.max(self.target_model(state_n).detach(),1)[0] + reward
            #print(reward)
            #print(state_value)
            #print(expected_value)
            #input()

            loss = criterion(state_value, expected_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            episodes += episode
            print('\rTrain Step: {} | Episode: {:.0f} | Loss: {:.2f} | Time: {}  '.format(
                e + 1,  episodes, float(loss) ,
                self.timeSince(start, (e - ((e-1)// print_every)* print_every)/ print_every )),end='')
            sys.stdout.flush()
            if (e+1) % print_every ==0:
                print()
                self.test((e+1)// print_every)
                torch.save(self.current_model, self.args.model_path)
                start= time.time()
    def test(self, epoch=0):
        self.current_model.eval()
        with torch.no_grad():
            done = False
            states=self.env.reset()
            rewards = 0.0
            cnt = 0
            while not done:
                x = Variable(torch.FloatTensor(states).unsqueeze(0).cuda())
                action= torch.max(self.current_model(x),1)[1].cpu().data
                state_n , reward , done, _ = self.env.step(int(action))
                '''
                if done:
                    print(self.current_model(x))
                    print(action)
                '''
                
                rewards += reward
                cnt += 1
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
    def __init__(self, input_size, action_dim, args):
        super(QNetwork, self).__init__()
        self.LeakyReLU= nn.LeakyReLU(0.2, inplace= True)
        self.SELU = nn.SELU()
        # input size. 84 x 84 x 4
        self.cnn= nn.Sequential(
            # state size. 4 x 84 x 84
            nn.Conv2d( input_size[2], 32, 8, 4, 2, bias=True),
            # state size. 32 x 21x 21
            nn.ReLU(),
            nn.Conv2d( 32, 64, 4, 2, 1, bias=True),
            # state size. 64 x 10 x 10
            nn.ReLU(),
            nn.Conv2d( 64, 64 , 3, 1, 1, bias=True),
            # state size. (hidden_size*4) x 10 x 10
            nn.ReLU())
        self.linear = nn.Sequential(
            nn.Linear( 64* (input_size[0]// 8) * (input_size[1]// 8), 256),
            nn.ReLU(),
            nn.Linear(256 , action_dim))
        self._initialize_weights()
    def forward(self,x ):
        # input is batch_size x 84 x 84 x 4
        x = x.permute(0,3,1,2)/255
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
        self.env = env
        self.action_dim = env.get_action_space().n
        self.done = True
        self.state= None

        self.buffer=[]

        self.buffer_size = args.buffer_size
        self.step = args.current_update_step
        self.batch_size = args.batch_size
    def collect_data(self, model, epsilon):
        episode=0
        for s in range(self.step):
            if (self.done):
                self.state= torch.FloatTensor(self.env.reset()).unsqueeze(0)
                self.done = False
            action = self.epsilon_greedy(model(Variable(self.state.cuda())), epsilon)
            observation, reward, done, _ = self.env.step(int(action))
            state_n = torch.FloatTensor(observation).unsqueeze(0)
            self.buffer.append([self.state.clone(), action, reward, state_n.clone()])
            self.state= state_n
            self.done= done
            if (self.done):
                episode +=1
        self.buffer= self.buffer[-self.buffer_size:]

        batch_size = min(self.batch_size, len(self.buffer))
        result = random.sample(self.buffer, batch_size)
        #print(len(self.buffer))
        return result, episode
    def epsilon_greedy(self, action_value, epsilon):
        if random.random() < epsilon:
            #print(torch.max(action_value,1)[1].cpu().data.size())
            #print(torch.LongTensor([random.randint(0,self.action_dim-1)]).size())
            return torch.LongTensor([random.randint(0,self.action_dim-1)])
        else:
            #print(torch.max(action_value,1)[1].cpu().data.size())
            #print(torch.LongTensor([random.randint(0,self.action_dim-1)]).size())
            return torch.max(action_value,1)[1].cpu().data

