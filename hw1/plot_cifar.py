
from util import Datamanager,DNN
import matplotlib.pyplot as plt
import numpy as np
import argparse
assert DNN and plt and np

dm=Datamanager()
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-lo','--loss_output', dest='loss_output',type=str,required=True)
parser.add_argument('-ao','--accu_output', dest='accu_output',type=str,required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       plot cifar_loss                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm.load_np('loss_cifar_d','record/loss_cifar_d.npy')
dm.load_np('loss_cifar_m','record/loss_cifar_m.npy')
dm.load_np('loss_cifar_s','record/loss_cifar_s.npy')

x=np.array(range(1,len(dm.data['loss_cifar_d'])+1))
plt.figure()
plt.plot(x,dm.data['loss_cifar_d'],'r',label='deep')
plt.plot(x,dm.data['loss_cifar_m'],'y',label='medium')
plt.plot(x,dm.data['loss_cifar_s'],'b',label='shallow')
plt.legend()
plt.savefig(args.loss_output)
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       plot accu                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm.load_np('accu_cifar_d','record/accu_cifar_d.npy')
dm.load_np('accu_cifar_m','record/accu_cifar_m.npy')
dm.load_np('accu_cifar_s','record/accu_cifar_s.npy')

x=np.array(range(1,len(dm.data['accu_cifar_d'])+1))
plt.figure()
plt.plot(x,dm.data['accu_cifar_d'],'r',label='deep')
plt.plot(x,dm.data['accu_cifar_m'],'y',label='medium')
plt.plot(x,dm.data['accu_cifar_s'],'b',label='shallow')
plt.legend()
plt.savefig(args.accu_output)
