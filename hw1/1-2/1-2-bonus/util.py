
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable, grad
import torch.nn.functional as F
import time
import numpy as np
assert F

class Datamanager():
    def __init__(self):
        self.data={}
    def get_data(self,name,b_size,shuf=True):
        X=np.linspace(-5,5,30000).reshape((-1,1))
        Y=np.sign(np.sin(5*X))
        X,Y=torch.from_numpy(X).float().cuda(),torch.from_numpy(Y).float().cuda()
        train_dataset = Data.TensorDataset(data_tensor=X[:], target_tensor=Y[:]) 
        self.data[name]=Data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=shuf)
    def load_np(self,name,path):
        self.data[name]=np.load(path)
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
            x, y= Variable(x,).cuda(), Variable(y).cuda() 
            output = model(x)
            loss = loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_index % 10 == 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)]\t |  Loss: {:.6f}'.format(
                        epoch, batch_index * len(x), len(trainloader.dataset),
                        100. * batch_index / len(trainloader), loss.data[0]),end='')

            # loss
            total_loss+= loss.data[0]*len(x) # sum up batch loss
        total_loss/= len(trainloader.dataset)
        elapsed= time.time() - start
        print('\nTime: {}:{:0>2d}\t | Total loss: {:.4f}'.format(
            int(elapsed/60),int(elapsed%60),total_loss))
    def val(self,model,valloader,verbose=True):
        model.eval()
        loss_func = nn.MSELoss(size_average=False)
        test_loss = 0
        for x, y in valloader:
            x, y = Variable(x).cuda(), Variable(y).cuda()
            output = model(x)
            # loss
            loss = loss_func(output, y)
            test_loss += float(loss)


        test_loss /= len(valloader.dataset)
        if verbose:
            print('Val set: Average loss: {:.4f}'.format( test_loss))
        return test_loss
    def test(self,model,trainloader):
        model.eval()
        pred_x=[]
        pred_y=[]
        for x,y in trainloader:
            pred_x.extend(list(x.cpu().numpy()))
            pred_y.extend(list(model(Variable(x).cuda()).cpu().data.numpy()))
        return np.array([pred_x,pred_y])
    def Hessian(self,l,var):
        g=[i.view(-1) for i in grad(l,var,retain_graph=True,create_graph= True)]
        g=torch.cat(g)
        h=[]
        for i in range(len(g)):
            e=torch.zeros(len(g)).cuda()
            e[i]=1
            e=Variable(e,requires_grad=True)
            t=[i.contiguous().view(-1) for i in grad(g.dot(e),var,retain_graph=True,create_graph= True)]
            t=torch.cat(t).view(1,-1)
            h.append(t)
        h=torch.cat(h,0)
        return (h)
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def interpolation(self,model1,model2,model_o,beta):
        weight={}
        dict1=model1.state_dict()
        dict2=model2.state_dict()
        for i in dict1:
            weight[i]=(1-beta)*dict1[i]+beta*dict2[i]
        model_o.load_state_dict(weight)
        return model_o
    def small_range_loss(self,model,epsilon):
        state=model.state_dict()
        output=[]
        dnn=DNN().cuda()
        c=1
        for i in state:
            for j in range(state[i].size()[0]):
                if len(state[i].size()) ==1:
                    loss=[]
                    print('\rprocessing...parameters {}'.format(c),end='')
                    for e in epsilon:
                        tmp=state.copy()
                        tmp[i][j]=state[i][j]+e
                        dnn.load_state_dict(tmp)
                        loss.append(self.val(dnn,self.data['train'],verbose=False))
                    output.append(torch.FloatTensor(loss).view(1,-1))
                    c+=1
                else:
                    for k in range(state[i].size()[1]):
                        loss=[]
                        print('\rprocessing...parameters {}'.format(c),end='')
                        for e in epsilon:
                            tmp=state.copy()
                            tmp[i][j,k]=state[i][j,k]+e
                            dnn.load_state_dict(tmp)
                            loss.append(self.val(dnn,self.data['train'],verbose=False))
                        output.append(torch.FloatTensor(loss).view(1,-1))
                        c+=1
        print()
        return torch.cat(output,0)


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.den1=( nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
        ))
        self.den2=( nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
        ))
        self.den3=( nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
        ))
        self.den4=( nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
        ))
        self.den5=( nn.Sequential(
            nn.Linear(10, 1),
        ))
    def forward(self, x):
        x = self.den1(x)
        x = self.den2(x)
        x = self.den3(x)
        x = self.den4(x)
        x = self.den5(x)
        return x 
    def save(self,path):
        torch.save(self.state_dict(), path)
    def load(self,path):
        self.load_state_dict(torch.load(path))

