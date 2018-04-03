
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from util import Datamanager, CNN
import numpy as np
import argparse
assert torch and nn and Variable and np and argparse


EPOCH = 30
BATCH_SIZE = 1024
############################################################
#               reading data                               #
############################################################
dm=Datamanager()
print('reading data...',end='')
sys.stdout.flush()
dm.get_Mnist('mnist',BATCH_SIZE)
print('\rreading data...finish')
############################################################
#               training                                   #
############################################################

cnn1 = CNN().cuda()
cnn2 = CNN().cuda()

'''
#print(cnn)
cnn1.load_state_dict(torch.load('weights/64'))
params1 = cnn1.named_parameters()
cnn2.load_state_dict(torch.load('weights/1024'))
#params2 = cnn2.named_parameters()
dict_params2 = dict(params2)
beta_list = [0.3,0.5,0.7,0.9,-1,2]
for beta in beta_list:
    cnn = CNN().cuda()
    print(beta)
    dict_params2 = dict(params2)
    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data = (beta*param1.data + (1-beta)*dict_params2[name1].data)
    cnn.load_state_dict(dict_params2)
    dm.val(cnn,'Val',dm.data['mnist'][1])
    #dict_params2.clear()
    '''

cnn1=torch.load('weights/64')
cnn2=torch.load('weights/1024')
out={}
beta_list = np.linspace(-1,2,100)
print(beta_list)
cnn_list=[]
loss_list=[]
accu_list=[]
for beta in beta_list:
    for i in cnn1:
        out[i]=cnn1[i]*beta + cnn2[i]*(1-beta)
    cnn_tmp=CNN().cuda()
    cnn_tmp.load_state_dict(out)
    loss,accu=dm.val(cnn_tmp,'Val',dm.data['mnist'][1])
    loss_list.append(loss)
    accu_list.append(accu)

