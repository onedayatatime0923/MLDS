
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from util_hessian import Datamanager, CNN, DNN
import numpy as np
import argparse
assert torch and nn and Variable

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

EPOCH = 20001
BATCH_SIZE = 512

parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-lo','--loss_output', dest='loss_output',type=str,required=True)
#parser.add_argument('-no','--norm_output', dest='norm_output',type=str,required=True)
parser.add_argument('-ra','--ratio_output', dest='ratio_output',type=str,required=True)
parser.add_argument('-u','--unit', dest='unit',type=int,nargs='+',required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager()
print('reading data...',end='')
sys.stdout.flush()
dm.get_data('train',BATCH_SIZE)
dm.get_data('test',BATCH_SIZE)
print('\rreading data...finish')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
print(args.loss_output)
print(args.ratio_output)
dnn = DNN(args).cuda().double()
print(dnn)
print('total parameters: {}'.format(dm.count_parameters(dnn)))
#training and testing
loss_list=[]
ratio_list=[]

for epoch in range(EPOCH): 
    loss , norm , min_ratio = dm.train(dnn,dm.data['train'],epoch,'mse')
    if epoch == 20000:
        loss_list.append(loss)
        ratio_list.append(min_ratio)
    print('*'*50)
#print(len(loss_list))
#print(len(norm_list))
np.save('result/'+args.loss_output,np.array(loss_list))
np.save('result/'+args.ratio_output,np.array(ratio_list))



