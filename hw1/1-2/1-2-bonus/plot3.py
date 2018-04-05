
from util import Datamanager,DNN
import matplotlib.pyplot as plt
import torch 
import argparse
assert DNN and plt and torch


dm=Datamanager()
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-ii','--initial_input', dest='ii',type=str,required=True)
parser.add_argument('-ei','--end_input', dest='ei',type=str,required=True)
parser.add_argument('-o','--output', dest='output',type=str,required=True)
args = parser.parse_args()
batch_size=4096*8
dm.get_data('train',batch_size)
########################################################################
#                     bonus 3                                          #
########################################################################
dnn_initial = DNN().cuda()
dnn_initial.load(args.ii)
dnn_end= DNN().cuda()
dnn_end.load(args.ei)
print(dm.count_parameters(dnn_end))

dnn=dm.interpolation(dnn_initial,dnn_end,DNN().cuda(),0.5) 

x=torch.linspace(0,1,10000)
loss=[]
for i in range(len(x)):
    print('\rprocessing...{}'.format(i),end='')
    loss.append(dm.val(dm.interpolation(dnn_initial,dnn_end,DNN().cuda(),x[i]),dm.data['train'],verbose=False))
print('processing...end')
loss=torch.FloatTensor(loss)

plt.figure()
plt.plot(x.numpy(),loss.numpy(),c='b')
plt.title('Loss')
plt.xlabel('beta')
plt.ylabel('loss')

plt.savefig(args.output)
#plt.show()
