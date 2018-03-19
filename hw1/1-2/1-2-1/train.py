
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

EPOCH = 60
BATCH_SIZE = 1024

parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-u','--unit', dest='unit',nargs='+',type=int,required=True)
parser.add_argument('-po','--p_output', dest='p_output',type=str,required=True)
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
parameter_list=[]
for epoch in range(EPOCH):
    p=dm.train(dnn,dm.data['train'][0],epoch,'cross_entropy')
    print('-'*50)
    if epoch%2==0: parameter_list.append(p)
print(np.array(parameter_list).shape)
np.save(args.p_output,np.array(parameter_list))

