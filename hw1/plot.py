
from util import Datamanager,DNN
import matplotlib.pyplot as plt
import numpy as np
import argparse
assert DNN and plt and np

dm=Datamanager()
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-f','--func', dest='func',type=int,required=True)
parser.add_argument('-lo','--loss_output', dest='loss_output',type=str,required=True)
parser.add_argument('-po','--pred_output', dest='pred_output',type=str,required=True)
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       plot prediction                          '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm.get_data('truth',args.func,256,False)
x=dm.data['truth'].dataset.data_tensor.cpu().numpy()
y=dm.data['truth'].dataset.target_tensor.cpu().numpy()

dm.load_np('pred_d','record/pred_d.npy')
dm.load_np('pred_m','record/pred_m.npy')
dm.load_np('pred_s','record/pred_s.npy')

plt.figure()
plt.plot(x,y,'g',label='ground truth')
plt.plot(x,dm.data['pred_d'][1],'r',label='deep')
plt.plot(x,dm.data['pred_m'][1],'y',label='medium')
plt.plot(x,dm.data['pred_s'][1],'b',label='shallow')
plt.legend()
plt.savefig(args.pred_output)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       plot loss                                      '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
'''
dm.load_np('loss_d','record/loss_d.npy')
dm.load_np('loss_m','record/loss_m.npy')
dm.load_np('loss_s','record/loss_s.npy')

x=np.array(range(1,len(dm.data['loss_d'])+1))
plt.figure()
plt.plot(x,dm.data['loss_d'],'r',label='deep')
plt.plot(x,dm.data['loss_m'],'y',label='medium')
plt.plot(x,dm.data['loss_s'],'b',label='shallow')
plt.legend()
plt.savefig(args.loss_output)
'''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       plot mnist_loss                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
dm.load_np('mnist_loss_d','mnist_loss_d.npy')
dm.load_np('mnist_loss_m','mnist_loss_m.npy')
dm.load_np('mnist_loss_s','mnist_loss_s.npy')

x=np.array(range(1,len(dm.data['mnist_loss_d'])+1))
plt.figure()
plt.plot(x,dm.data['mnist_loss_d'],'r',label='deep')
plt.plot(x,dm.data['mnist_loss_m'],'y',label='medium')
plt.plot(x,dm.data['mnist_loss_s'],'b',label='shallow')
plt.legend()
plt.savefig('mnist_loss.png')
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       plot accu                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
dm.load_np('mnist_accu_d','mnist_accu_d.npy')
dm.load_np('mnist_accu_m','mnist_accu_m.npy')
dm.load_np('mnist_accu_s','mnist_accu_s.npy')

x=np.array(range(1,len(dm.data['mnist_accu_d'])+1))
plt.figure()
plt.plot(x,dm.data['mnist_accu_d'],'r',label='deep')
plt.plot(x,dm.data['mnist_accu_m'],'y',label='medium')
plt.plot(x,dm.data['mnist_accu_s'],'b',label='shallow')
plt.legend()
plt.savefig('mnist_accu.png')
'''
