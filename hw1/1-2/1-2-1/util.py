
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
from sklearn.decomposition import PCA
import numpy as np
assert F and np and Data 

class Datamanager():
    def __init__(self):
        self.data={}
        self.pca=0
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
    def train(self,model,trainloader,epoch,loss,args):
        start= time.time()

        model.train()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)   # optimize all cnn parameters
        if loss=='mse':
            loss_func = nn.MSELoss()
        elif loss=='cross_entropy':
            loss_func = nn.CrossEntropyLoss()

        total_loss=0
        correct=0
        
        for batch_index, (x, y) in enumerate(trainloader):
            x, y= Variable(x).cuda(), Variable(y).cuda() 
            output = model(x)
            loss = loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = output.data.max(1,keepdim=True)[1] # get the index of the max log-probability
            #print('max:',output.data.max(1, keepdim=True).size())
            total_loss+= loss.data[0]*len(x) # sum up batch loss
            correct += pred.eq(y.data.view_as(pred)).long().cpu().sum()
            if batch_index % 4 == 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)]\t |  Loss: {:.6f}'.format(
                        epoch, batch_index * len(x), len(trainloader.dataset),
                        100. * batch_index / len(trainloader), loss.data[0]),end='')

        elapsed= time.time() - start
        total_loss/= len(trainloader.dataset)
        accu = 100. * correct / len(trainloader.dataset)
        print('\nTime: {}:{}\t | '.format(int(elapsed/60),int(elapsed%60)),end='')
        print('Total loss: {:.4f} | Accu: {:.2f}'.format(total_loss,accu))

        para=[accu]
        if args.mode=="all_layer":
            for p in model.parameters():
                para.extend(list(p.data.cpu().numpy().reshape((-1,))))
        elif args.mode=="first_layer":
            p=model.parameters()
            para.extend(list(p[0].data.cpu().numpy().reshape((-1,))))
            para.extend(list(p[1].data.cpu().numpy().reshape((-1,))))
        else: raise ValueError('Wrong mode.')

        return np.array(para)
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def load_np(self,name,path):
        self.data[name]=np.load(path)
    def pca_construct(self,data,n):
        pca = PCA(n_components=n)
        pca.fit(data)
        self.pca=pca
    def pca_transform(self,data):
        return self.pca.transform(data)
    def pca_reconstruct(self,data):
        return self.pca.inverse_transform(data)

class DNN(nn.Module):
    def __init__(self,args):
        super(DNN, self).__init__()
        self.den=nn.ModuleList()
        self.den.append( nn.Sequential(
            nn.Linear(28*28, args.unit[0]),
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
        x = x.view(x.size(0), -1)
        for i in self.den:
            x= i(x)
        return x 

