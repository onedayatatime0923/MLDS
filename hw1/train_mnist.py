
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from util import Datamanager, CNN
import numpy as np
assert torch and nn and Variable

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

EPOCH = 80
BATCH_SIZE =512

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager()
print('reading data...',end='')
sys.stdout.flush()
dm.get_Mnist('train',BATCH_SIZE)
print('\rreading data...finish')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

cnn = CNN().cuda()
print(cnn)
print('total parameters: {}'.format(dm.count_parameters(cnn)))
# training and testing
loss_list=[]
accu_list=[]
for epoch in range(EPOCH):
    dm.train(cnn,dm.data['train'][0],epoch,'cross_entropy')
    loss,accu=dm.val(cnn,dm.data['train'][1],epoch)
    loss_list.append(loss)
    accu_list.append(accu)
print(len(loss_list))
print(len(accu_list))
np.save('mnist_loss_d.npy',np.array(loss_list))
np.save('mnist_accu_d.npy',np.array(accu_list))

