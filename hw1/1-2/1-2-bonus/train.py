
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

EPOCH = 2000
BATCH_SIZE = 128

parser = argparse.ArgumentParser(description='1-2-bonus.')
parser.add_argument('-io','--initial_output', dest='io',type=str,required=True)
parser.add_argument('-eo','--end_output', dest='eo',type=str,required=True)
#parser.add_argument('-go','--grad_output', dest='grad_output',type=str,required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager()
print('reading data...',end='')
sys.stdout.flush()
dm.get_data('train',BATCH_SIZE)
print('\rreading data...finish')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

dnn = DNN().cuda()
dnn.save(args.io)
print(dnn)
print('total parameters: {}'.format(dm.count_parameters(dnn)))
# training and testing
for epoch in range(EPOCH):
    dm.train(dnn,dm.data['train'],epoch,'mse')
    print('-'*50)

dnn.save(args.eo)
