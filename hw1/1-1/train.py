
from util import Datamanager,DNN
import sys
import numpy as np
import argparse
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
EPOCH = 1000
BATCH_SIZE = 2048 
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-f','--func', dest='func',type=int,required=True)
parser.add_argument('-u','--unit', dest='unit',type=int,nargs='+',required=True)
parser.add_argument('-lo','--loss_output', dest='loss_output',type=str,required=True)
parser.add_argument('-po','--pred_output', dest='pred_output',type=str,required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager()
print('getting data...',end='')
sys.stdout.flush()
dm.get_data('train',args.func,BATCH_SIZE)
dm.get_data('test',args.func,BATCH_SIZE,False)
print('\rgetting data...finish')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

dnn = DNN(args).double().cuda()
print(dnn)
print('total parameters: {}'.format(dm.count_parameters(dnn)))
lost=[]
# training and testing
for epoch in range(EPOCH):
    lost.append(dm.train(dnn,dm.data['train'],epoch,'mse'))
    print('-'*50)
np.save(args.loss_output,np.array(lost))
test=dm.test(dnn,dm.data['test'])
np.save(args.pred_output,np.array(test))
