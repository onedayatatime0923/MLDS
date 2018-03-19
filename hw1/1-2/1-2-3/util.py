
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
from torch.autograd import grad
import numpy as np
assert F

class Datamanager():
    def __init__(self):
        self.data={}
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

        total_loss = 0
        norm = 0
        if(epoch<50): print('minimize loss')
        else: print('minimize gradient norm')
        for batch_index, (x, y) in enumerate(trainloader):
            x, y= Variable(x).cuda(), Variable(y).cuda() 
            output = model(x)
            if(epoch<50):
                loss = loss_func(output,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                loss = loss_func(output,y)
                loss_grads = grad(loss, model.parameters(),retain_graph=True,create_graph=True)
                gn2 = sum([grd.norm()**2 for grd in loss_grads])
                norm = gn2.cpu().data
                optimizer.zero_grad()
                gn2.backward()
                optimizer.step()
                if(batch_index%4==0):
                    print('\rTrain epoch: {}  , gn2 = {} '.format(epoch,gn2))
            if batch_index % 4 == 0 and epoch<50 :
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)]\t |  Loss: {:.6f}'.format(
                        epoch, batch_index * len(x), len(trainloader.dataset),
                        100. * batch_index / len(trainloader), loss.data[0]),end='')

            total_loss+= loss.data[0]*len(x) # sum up batch loss
       
        elapsed= time.time() - start
        total_loss/= len(trainloader.dataset)
        print('\nTime: {}:{}\t | '.format(int(elapsed/60),int(elapsed%60)),end='')
        print('Total loss: {:.4f}'.format(total_loss))
        #return total_loss
        return norm
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
        print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
            nn.Conv2d(1, 5, 3, 1, 1),              # output shape (5, 28, 28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape (5, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d( 5,  5, 3, 1, 1),              # output shape (5, 14, 14)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape ( 5, 7, 7)
        )
        self.conv3 = nn.Sequential(
	    nn.Conv2d( 5,  5, 3, 1, 1),              # output shape ( 5, 7, 7)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape ( 5, 3, 3)
        )
        self.den1= nn.Sequential(
            nn.Linear(5* 3 *3,  32),
            nn.ReLU(),
        )
        self.den2= nn.Sequential(
            nn.Linear( 32,  32),
            nn.ReLU(),
        )
        self.den3= nn.Sequential(
            nn.Linear( 32, 10),
        ) 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x= self.den1(x)
        x= self.den2(x)
        x= self.den3(x)
        return x 
