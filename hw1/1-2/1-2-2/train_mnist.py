
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from util import Datamanager, DNN
import numpy as np
import argparse
assert np and torch and nn and Variable

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

EPOCH = 1000
BATCH_SIZE =4096

parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-u','--unit', dest='unit',nargs='+',type=int,required=True)
parser.add_argument('-lo','--loss_output', dest='loss_output',type=str,required=True)
parser.add_argument('-go','--grad_output', dest='grad_output',type=str,required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager()
print('reading data...',end='')
sys.stdout.flush()
dm.get_Mnist('train',BATCH_SIZE)
print('\rreading data...finish')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

dnn = DNN(args).cuda()
print(dnn)
print('total parameters: {}'.format(dm.count_parameters(dnn)))
# training and testing
loss_list=[]
grad_list=[]
for epoch in range(EPOCH):
    dm.train(dnn,dm.data['train'][0],epoch,'cross_entropy')
    loss,grad=dm.val(dnn,dm.data['train'][0],epoch)
    print('-'*50)
    loss_list.append(loss)
    grad_list.append(grad)
print(len(loss_list))
print(len(grad_list))
np.save(args.loss_output,np.array(loss_list))
np.save(args.grad_output,np.array(grad_list))

