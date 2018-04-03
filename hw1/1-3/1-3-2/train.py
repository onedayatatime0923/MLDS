
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

EPOCH = 30
BATCH_SIZE =512*4

parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-u','--unit', dest='unit',type=int,required=True)
parser.add_argument('-cr','--clear_record', dest='clear_record',action='store_true')
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

cnn = CNN(args).cuda()
parameters =dm.count_parameters(cnn)
print(cnn)
print('total parameters: {}'.format(parameters))
# training and testing
for epoch in range(EPOCH):
    dm.train(cnn,dm.data['mnist'][0],epoch,'cross_entropy')
    dm.val(cnn,'Train',dm.data['mnist'][0])
    dm.val(cnn,'Val',dm.data['mnist'][1])
    print('-'*50)
train_record=np.array([parameters]+dm.val(cnn,'Train',dm.data['mnist'][0])).reshape((1,3))
test_record=np.array([parameters]+dm.val(cnn,'Val',dm.data['mnist'][1])).reshape((1,3))


if not args.clear_record:
    train_record=np.concatenate((np.load('record/train.npy'),train_record),0)
    test_record=np.concatenate((np.load('record/test.npy'),test_record),0)
print('train record shape: {}'.format(train_record.shape))
print('test record shape: {}'.format(test_record.shape))
np.save('./record/train.npy',train_record)
np.save('./record/test.npy',test_record)
