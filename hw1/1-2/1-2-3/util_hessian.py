
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import numpy as np
from torch.autograd import grad
from numpy.linalg import svd, eigh
assert F


def HessianJacobian(loss, parameters):
    params = [p for p in parameters]
    J, H = [], []
    for p in params:
        g, = torch.autograd.grad(loss, p , create_graph=True)
        sz = list(g.size())
        if len(sz) == 2:
            for i in range(sz[0]):
                for j in range(sz[1]):
                    J.append(g[i, j])
        else:
            for i in range(sz[0]):
                J.append(g[i])
    
    for i, g in enumerate(J):
        H.append([])
        for p in params:
            g2, = torch.autograd.grad(g, p, create_graph=True)
            sz = list(g2.size())
            if len(sz) == 2:
                for j in range(sz[0]):
                    for k in range(sz[1]):
                        H[i].append(g2[j, k].cpu().data.numpy()[0])
            else:
                for j in range(sz[0]):
                    H[i].append(g2[j].cpu().data.numpy()[0])
    J = [i.cpu().data.numpy()[0] for i in J]
    return np.array(H), np.array(J)	        

class Datamanager():
    def __init__(self):
        self.data={}
    def get_data(self,name,b_size,shuf=True):
        X=np.linspace(-5,5,128).reshape((-1,1))
        Y=np.exp(np.sinc(5*X))
        X,Y=torch.from_numpy(X).double().cuda(),torch.from_numpy(Y).double().cuda()
        train_dataset = Data.TensorDataset(data_tensor=X[:], target_tensor=Y[:]) 
        self.data[name]=Data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=shuf)
    def get_Mnist(self,name,b_size):
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../../../dataset', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                    batch_size=b_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../../../dataset', train=False, transform=transforms.Compose([
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

        total_loss = 0
        norm = 0
        min_ratio = 0 

        if(epoch<10000): print('minimize loss')
        else: print('minimize gradient norm')
        for batch_index, (x, y) in enumerate(trainloader):
            print('index=',batch_index)
            x, y= Variable(x).cuda(), Variable(y).cuda()
            output = model(x)
            if(epoch<10000):
                loss = loss_func(output,y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss = loss_func(output,y)
                loss_grads = grad(loss, model.parameters(),retain_graph=True,create_graph=True)
                gn2 = sum([grd.norm()**2 for grd in loss_grads])
                norm = gn2.cpu().data.sqrt()
                optimizer.zero_grad()
                gn2.backward(retain_graph=True)
                optimizer.step()
                """
                #h , j = HessianJacobian(gn2, model.parameters()) 
                input_grad = loss_grads
                #par_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
                par = model.parameters()
                print('input_grad : ',input_grad)
                hessian = []
                print(len(input_grad))
                for i in range(len(input_grad)):
                    print('i=',i)
                    print(len(input_grad[i]))
                    a = input_grad[i].view(-1,1)
                    for j in range(a.shape[0]):
                        print('j=',j) 
                        print('a=',a[j])
                        h = grad( a[j] , par , retain_graph=True , create_graph=True )
                        print(h)
                        hessian.append(h)
                """
            if batch_index % 4 == 0 :
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)]\t |  Loss: {:.6f}'.format(
                        epoch, batch_index * len(x), len(trainloader.dataset),
                        100. * batch_index / len(trainloader), loss.data[0]),end='')
            if epoch>=10000 : print('\nnorm = {} \n'.format(norm))
            total_loss+= loss.data[0]*len(x) # sum up batch loss
        if(epoch==20000):
            h , j = HessianJacobian(loss_func(output,y) , model.parameters())  
            print('\nhessian matrix: {}\n shape: {}\n'.format(h,h.shape))
            mean  = h.mean(axis = 0)
            arr_train = h - mean
            val_train , U_train = eigh(np.cov(arr_train.T))
            eigen_value = list(val_train)
            print(eigen_value)
            count = 0
            for x in eigen_value:
                if x > 0 : count+=1
            min_ratio = float(count/len(eigen_value))
            print('\nMinimal ratio = {} \n'.format(min_ratio))
 
        elapsed= time.time() - start
        total_loss/= len(trainloader.dataset)

        print('\nTime: {}:{}\t | '.format(int(elapsed/60),int(elapsed%60)),end='')
        print('Total loss: {:.4f}'.format(total_loss))
        return total_loss , norm , min_ratio
        
    def val(self,model,valloader,epoch):
        model.eval()
        test_loss = 0
        correct = 0
        for x, y in valloader:
            x, y = Variable(x, volatile=True).cuda(), Variable(y,volatile=True).cuda()
            output = model(x)
            test_loss += F.mse_loss(output, y, size_average=False).data[0] # sum up batch loss
            #print('cross_entropy:',F.cross_entropy(output, y, size_average=False).data.size())
            pred = output.data.max(1,keepdim=True)[1] # get the index of the max log-probability
            #print('max:',output.data.max(1, keepdim=True).size())
            correct += pred.eq(y.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(valloader.dataset)


        print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), grad={:.4f}'.format(
            test_loss, correct, len(valloader.dataset),
            100. * correct / len(valloader.dataset),gradient))
        return test_loss
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
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        for i in self.den:
            x= i(x)
        return x 

class CNN(nn.Module):
    def __init__(self,mode):
        super(CNN, self).__init__()
        self.mode=mode
        if mode=='mnist_deep':
            self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
                nn.Conv2d(1, 5, 3, 1, 1),              # output shape (3, 28, 28)
                nn.ReLU(),
                #nn.AvgPool2d(kernel_size=2),            # output shape (5, 14, 14)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d( 5,  5, 3, 1, 1),              # output shape (5, 14, 14)
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2),            # output shape ( 5, 7, 7)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d( 5,  5, 3, 1, 1),              # output shape ( 5, 7, 7)
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=7),            # output shape ( 5, 3, 3)
            )
            self.den1= nn.Sequential(
                nn.Linear(5* 2 *2,  32),
                nn.ReLU(),
            )
            self.den2= nn.Sequential(
                nn.Linear( 32,  32),
                nn.ReLU(),
            )
            self.den3= nn.Sequential(
                nn.Linear( 32, 10),
            ) 
        elif mode=='mnist_medium':
            self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
                nn.Conv2d(1, 2, 3, 1, 1),              # output shape (5, 28, 28)
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2),            # output shape (5, 14, 14)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d( 2,  4, 3, 1, 1),              # output shape (5, 14, 14)
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=4),            # output shape ( 5, 3, 3)
            )
            self.den1= nn.Sequential(
                nn.Linear(4* 3 *3,  32),
                nn.ReLU(),
            )
            self.den2= nn.Sequential(
                nn.Linear( 32,  32),
                nn.ReLU(),
            )
            self.den3= nn.Sequential(
                nn.Linear( 32, 10),
            ) 
        elif mode=='mnist_shallow':
            self.conv1 = nn.Sequential(                 # input shape (1, 28, 28)
                nn.Conv2d(1, 4, 3, 1, 1),              # output shape (5, 28, 28)
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=8),            # output shape (5,  7,  7)
            )
            self.den1= nn.Sequential(
                nn.Linear(4* 3 *3,  32),
                nn.ReLU(),
            )
            self.den2= nn.Sequential(
                nn.Linear( 32,  32),
                nn.ReLU(),
            )
            self.den3= nn.Sequential(
                nn.Linear( 32, 10),
            ) 
        elif mode=='cifar_deep':
            self.conv1 = nn.Sequential(                 # input shape (3, 32, 32)
                nn.Conv2d(3, 8, 5, 1, 2),              # output shape (3, 32, 32)
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2),            # output shape (5, 16, 16)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d( 8,  8, 5, 1, 2),              # output shape (5, 16, 16)
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2),            # output shape ( 5, 8, 8)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d( 8,  8, 5, 1, 2),              # output shape ( 5, 8, 8)
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2),            # output shape ( 5, 4, 4)
            )
            self.den1= nn.Sequential(
                nn.Linear(8* 4 *4,  128),
                nn.ReLU(),
            )
            self.den2= nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            self.den3= nn.Sequential(
                nn.Linear(128, 10),
            )
            '''
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
            '''
    def forward(self, x):
        if self.mode=='mnist_deep':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.size(0), -1)
            x= self.den1(x)
            x= self.den2(x)
            x= self.den3(x)
        elif self.mode=='mnist_medium':
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            x= self.den1(x)
            x= self.den2(x)
            x= self.den3(x)
        elif self.mode=='mnist_shallow':
            x = self.conv1(x)
            x = x.view(x.size(0), -1)
            x= self.den1(x)
            x= self.den2(x)
            x= self.den3(x)
        elif self.mode=='cifar_deep':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(x.size(0), -1)
            x= self.den1(x)
            x= self.den2(x)
            x= self.den3(x)
        return x 
