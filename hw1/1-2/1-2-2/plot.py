
from util import Datamanager,DNN
import matplotlib.pyplot as plt
import numpy as np
import argparse
assert DNN and plt and np

dm=Datamanager()
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-o','--output', dest='output',type=str,required=True)
#parser.add_argument('-go','--grad_output', dest='grad_output',type=str,required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       plot mnist_loss                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm.load_np('loss_mnist','record/loss_mnist.npy')
dm.load_np('grad_mnist','record/grad_mnist.npy')
x=np.array(range(1,len(dm.data['loss_mnist'])+1))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x,dm.data['loss_mnist'],c='r',label='loss')
ax2.plot(x,dm.data['grad_mnist'],c='b',label='gradient')
plt.title('Loss Sensitivity')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color='r')
ax1.tick_params('y', colors='r')
ax2.set_ylabel('gradient', color='b')
ax2.tick_params('y', colors='b')
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')

plt.savefig(args.output)
