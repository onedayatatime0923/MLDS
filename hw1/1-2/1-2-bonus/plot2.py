
from util import Datamanager,DNN
import matplotlib.pyplot as plt
import torch 
import argparse
assert DNN and plt and torch


dm=Datamanager()
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-m','--model', dest='model',type=str,required=True)
#parser.add_argument('-ei','--end_input', dest='ei',type=str,required=True)
parser.add_argument('-o','--output', dest='output',type=str,required=True)
args = parser.parse_args()
batch_size=1024
color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9','b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
########################################################################
#                     bonus 2                                          #
########################################################################
dm.get_data('train',batch_size)
dnn= DNN().cuda()
dnn.load(args.model)
epsilon=torch.linspace(-1,1,21)
y=dm.small_range_loss(dnn,epsilon)
print(y)

plt.figure()
for i in range(len(y)):
    plt.plot(epsilon.numpy(),y[i].numpy(),c=color[i%len(color)])
plt.title('Loss')
plt.xlabel('epsilon')
plt.ylabel('loss')

plt.savefig(args.output)
#plt.show()
