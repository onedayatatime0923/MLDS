
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable, grad
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh
assert F and Data

class Datamanager():
    def __init__(self):
        self.data={}
    def get_Mnist(self,name,b_size):
        train_dataset=datasets.MNIST('../../../dataset', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
        train_loader = torch.utils.data.DataLoader(
                    train_dataset,   
                    batch_size=b_size, shuffle=True)
        test_dataset=datasets.MNIST('../../../dataset', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ]))
        test_loader = torch.utils.data.DataLoader(test_dataset, 
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
    def train(self,model,trainloader,epoch,loss):
        start= time.time()

        model.train()
        optimizer = torch.optim.Adam(model.parameters())   # optimize all cnn parameters
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
        print('\nTime: {:0>2d}:{:0>2d}\t | Total loss: {:.4f}\n'.format(int(elapsed/60),int(elapsed%60),total_loss),end='')
        return total_loss
    def val(self,model,name,valloader):
        model.eval()
        loss_func = nn.CrossEntropyLoss(size_average=False)
        test_loss = 0
        correct = 0
        g_total=[torch.zeros_like(i) for i in list(model.parameters())]
        h=torch.zeros(self.count_parameters(model),self.count_parameters(model)).cuda()
        for batch_index ,(x, y) in enumerate(valloader):
            x, y = Variable(x, volatile=False).cuda(), Variable(y,volatile=False).cuda()
            output = model(x)
            loss = loss_func(output,y)
            # loss
            test_loss += float(loss)
            # accu
            pred = output.data.max(1,keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).long().cpu().sum()
            # grad
            g=torch.autograd.grad(loss, model.parameters(),retain_graph=True)
            for i in range(len(g_total)):
                g_total[i]+=g[i]
            # sharp
            if batch_index % 40 == 0:
                print('\rVal: [{}/{} ({:.0f}%)]'.format(
                         batch_index * len(x), len(valloader.dataset),
                        100. * batch_index / len(valloader)),end='')

        test_loss /= len(valloader.dataset)
        g_total = sum([grd.norm()**2 for grd in g_total]).data[0]/ len(valloader.dataset)
        h+=self.Hessian(loss,list(model.parameters())).data
        sharp=self.sharpness(h.cpu().numpy())#/len(valloader.dataset))
        print(sharp)

        print('{} set: Average loss: {:.4f} | Accuracy: {}/{} ({:.0f}%) | Gradien Norm: {} | Sharpness: {}'.format(
            name,test_loss, correct, len(valloader.dataset),
            100. * correct / len(valloader.dataset),g_total,sharp))
        return [test_loss,100 * correct / len(valloader.dataset),g_total,sharp]
    def test(self,model,trainloader):
        model.eval()
        pred_x=[]
        pred_y=[]
        for x,y in trainloader:
            pred_x.extend(list(x.cpu().numpy()))
            pred_y.extend(list(model(Variable(x).cuda()).cpu().data.numpy()))
        return np.array([pred_x,pred_y])
    def load_np(self,name,path):
        self.data[name]=np.load(path)
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def sharpness(self,h):
        val,_= eigsh(h, k=1)
        return val

    def Hessian(self,l,var):
        g=torch.cat([i.view(-1) for i in grad(l,var,retain_graph=True,create_graph= True)])
        h=[]
        for i in range(len(g)):
            e=torch.zeros(len(g)).cuda()
            e[i]=1
            e=Variable(e,requires_grad=True)
            t=torch.cat([i.contiguous().view(1,-1) for i in grad(g.dot(e),var,retain_graph=True,create_graph= True)],1)
            h.append(t)
        h=torch.cat(h,0)
        return (h)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
            nn.Conv2d(1, 3, 3, 1, 1),              # output shape (4, 28, 28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape (4, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3,3, 3, 1, 1),              # output shape (4, 14, 14)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape ( 4, 7, 7)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3,3, 3, 1, 1),              # output shape ( 4, 7, 7)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),            # output shape ( 4, 3, 3)
        )
        self.den1= nn.Sequential(
            nn.Linear(3*3*3, 32),
            nn.ReLU(),
        )
        self.den2= nn.Sequential(
            nn.Linear(32, 10),
        ) 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x= self.den1(x)
        x= self.den2(x)
        return x 
