
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import numpy as np
assert F

class Datamanager():
    def __init__(self):
        self.data={}
    def get_data(self,name,b_size,shuf=True):
        X=np.linspace(-5,5,300000).reshape((-1,1))
        Y=np.exp(np.sinc(X))
        X,Y=torch.from_numpy(X).double().cuda(),torch.from_numpy(Y).double().cuda()
        train_dataset = Data.TensorDataset(data_tensor=X[:], target_tensor=Y[:]) 
        self.data[name]=Data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=shuf)
    def get_Mnist(self,name,b_size):
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                    batch_size=b_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                    batch_size=b_size, shuffle=True)
        self.data[name]=[train_loader,test_loader]
    def get_CIFAR10(self,name,b_size):
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
                    batch_size=b_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
                    batch_size=b_size, shuffle=True)
        self.data[name]=[train_loader,test_loader]
    def load_np(self,name,path):
        self.data[name]=np.load(path)
    def train(self,model,trainloader,epoch,loss):
        start= time.time()

        model.train()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)   # optimize all cnn parameters
        if loss=='mse':
            loss_func = nn.MSELoss()
        elif loss=='cross_entropy':
            loss_func = nn.CrossEntropyLoss()

        total_loss=0
        
        for batch_index, (x, y) in enumerate(trainloader):
            x, y= Variable(x).cuda(), Variable(y).cuda() 
            output = model(x)
            loss = loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_index % 4 == 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)]\t |  Loss: {:.6f}'.format(
                        epoch, batch_index * len(x), len(trainloader.dataset),
                        100. * batch_index / len(trainloader), loss.data[0]),end='')

            total_loss+= loss.data[0]*len(x) # sum up batch loss
        elapsed= time.time() - start
        total_loss/= len(trainloader.dataset)
        print('\nTime: {}:{}'.format(int(elapsed/60),int(elapsed%60)))
        print('Total loss: {:.4f}'.format(total_loss))
        return total_loss
    def val(self,model,valloader,epoch):
        model.eval()
        test_loss = 0
        correct = 0
        for x, y in valloader:
            x, y = Variable(x, volatile=True).cuda(), Variable(y,volatile=True).cuda()
            output = model(x)
            test_loss += F.cross_entropy(output, y, size_average=False).data[0] # sum up batch loss
            #print('cross_entropy:',F.cross_entropy(output, y, size_average=False).data.size())
            pred = output.data.max(1,keepdim=True)[1] # get the index of the max log-probability
            #print('max:',output.data.max(1, keepdim=True).size())
            correct += pred.eq(y.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(valloader.dataset)
        print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(valloader.dataset),
            100. * correct / len(valloader.dataset)))
        return test_loss,100 * correct / len(valloader.dataset)
    def test(self,model,trainloader):
        model.eval()
        pred_x=[]
        pred_y=[]
        for x,y in trainloader:
            pred_x.extend(list(x.cpu().numpy()))
            pred_y.extend(list(model(Variable(x).cuda()).cpu().data.numpy()))
        return np.array([pred_x,pred_y])
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DNN(nn.Module):
    def __init__(self,args):
        super(DNN, self).__init__()
        print(args.unit)
        self.den=nn.ModuleList()
        self.den.append( nn.Sequential(
            nn.Linear(1, args.unit[0]),
            nn.ReLU(),
        ))
        for i in range(1,len(args.unit)-1):
            self.den.append( nn.Sequential(
                nn.Linear(args.unit[i-1], args.unit[i]),
                nn.ReLU(),
            ))
        self.den.append( nn.Sequential(
            nn.Linear(args.unit[-2], args.unit[-1]),
        ))
        print(self.den)
    def forward(self, x):
        for i in self.den:
            x= i(x)
        return x 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(                 # input shape (3, 32, 32)
            nn.Conv2d(3, 8, 5, 1, 2),              # output shape (8, 32, 32)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape (8, 16, 16)
        )
        self.conv2 = nn.Sequential(                 # input shape (8, 16, 16)
            nn.Conv2d(8, 16, 5, 1, 2),              # output shape (16, 16, 16)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape (16, 8, 8)
        )
        self.conv3 = nn.Sequential(                 # input shape (16, 8, 8)
            nn.Conv2d(16, 32, 3, 1, 1),              # output shape (32, 8, 8)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape (32, 4, 4)
        )
        self.conv4 = nn.Sequential(                 # input shape (32, 4, 4)
            nn.Conv2d(32, 64, 3, 1, 1),              # output shape (64, 4, 4)
            nn.ReLU(),
            #nn.AvgPool2d(kernel_size=2),            # output shape (5, 2, 2)
        )
        self.den1= nn.Sequential(
            nn.Linear(64* 4 *4, 512),
            nn.ReLU(),
        )
        self.den2= nn.Sequential(
            nn.Linear(512, 128),
        )
        self.den3= nn.Sequential(
            nn.Linear(128, 48),
        )        
        self.den4= nn.Sequential(
            nn.Linear(48, 10),
        ) 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x= self.den1(x)
        x= self.den2(x)
        x= self.den3(x)
        x= self.den4(x)
        return x 
