
from util import Datamanager,DNN
import matplotlib.pyplot as plt
import numpy as np
import argparse
assert DNN and plt and np

dm=Datamanager()
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-o','--output', dest='output',type=str,required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       plot mnist_loss                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm.load_np('loss_mnist','record/loss_mnist.npy')
dm.load_np('grad_mnist','record/grad_mnist.npy')

x=np.array(range(1,len(dm.data['loss_mnist'])+1))
plt.figure()
plt.plot(x,dm.data['loss_mnist'],'g',label='loss')
plt.plot(x,dm.data['grad_mnist'],'y',label='grad')
plt.legend()
plt.savefig(args.output)
'''
'''
