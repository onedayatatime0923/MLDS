import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from util import Datamanager, CNN
import numpy as np
import argparse
assert torch and nn and Variable

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

EPOCH = 500
BATCH_SIZE =512

parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-m','--mode', dest='mode',type=str,required=True)
parser.add_argument('-lo','--loss_output', dest='loss_output',type=str,required=True)
parser.add_argument('-ao','--accu_output', dest='accu_output',type=str,required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager()
print('reading data...',end='')
sys.stdout.flush()
dm.get_CIFAR10('train',BATCH_SIZE)
print('\rreading data...finish')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

cnn = CNN(args.mode).cuda()
print(cnn)
print('total parameters: {}'.format(dm.count_parameters(cnn)))
# training and testing
loss_list=[]
accu_list=[]
for epoch in range(EPOCH):
    dm.train(cnn,dm.data['train'][0],epoch,'cross_entropy')
    loss,accu=dm.val(cnn,dm.data['train'][1],epoch)
    print('-'*50)
    loss_list.append(loss)
    accu_list.append(accu)
print(len(loss_list))
print(len(accu_list))
np.save(args.loss_output,np.array(loss_list))
np.save(args.accu_output,np.array(accu_list))
