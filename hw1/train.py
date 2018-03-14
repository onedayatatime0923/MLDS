
from util import Datamanager,DNN
import sys
import numpy as np
import argparse
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
EPOCH = 40
BATCH_SIZE = 2048 
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-u','--unit', dest='unit',type=int,nargs='+',required=True)
parser.add_argument('-o','--output', dest='output',type=str,required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager()
print('getting data...',end='')
sys.stdout.flush()
dm.get_data('train',BATCH_SIZE)
dm.get_data('test',BATCH_SIZE,False)
print('\rgetting data...finish')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

dnn = DNN(args).double().cuda()
print(dnn)
print(dm.count_parameters(dnn))
lost=[]
# training and testing
for epoch in range(EPOCH):
    lost.append(dm.train(dnn,dm.data['train'],epoch,'mse'))
np.save('loss_'+args.output+'.npy',np.array(lost))
test=dm.test(dnn,dm.data['test'])
np.save('prediction_'+args.output+'.npy',np.array(test))
