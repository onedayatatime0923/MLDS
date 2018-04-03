
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from util import Datamanager, CNN
import numpy as np
import argparse
assert torch and nn and Variable and np

############################################################
#               setting option                             #
############################################################

EPOCH = 300
BATCH_SIZE =512*4

parser = argparse.ArgumentParser(description='setting module parameter.')
#parser.add_argument('-m','--mode', dest='mode',type=str,required=True)
#parser.add_argument('-lo','--loss_output', dest='loss_output',type=str,required=True)
#parser.add_argument('-ao','--accu_output', dest='accu_output',type=str,required=True)
args = parser.parse_args()
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

cnn = CNN().cuda()
print(cnn)
print('total parameters: {}'.format(dm.count_parameters(cnn)))
# training and testing
train_record=[]
test_record=[]

for epoch in range(EPOCH):
    dm.train(cnn,dm.data['mnist'][0],epoch,'cross_entropy')
    train_record.append(dm.val(cnn,'Train',dm.data['mnist'][0]))
    test_record.append(dm.val(cnn,'Val',dm.data['mnist'][1]))
    print('-'*50)

np.save('record/train.npy',np.array(train_record))
np.save('record/test.npy',np.array(test_record))
