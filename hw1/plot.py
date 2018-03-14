
from util import Datamanager,DNN
import matplotlib.pyplot as plt
import numpy as np
assert DNN and plt

dm=Datamanager()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       plot prediction                          '''
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm.load_np('pred_d','prediction_deep.npy')
dm.load_np('pred_m','prediction_medium.npy')
dm.load_np('pred_s','prediction_shallow.npy')

x=dm.data['pred_d'][0]
y=np.sinc(x)
plt.figure()
plt.plot(x,y,'g',label='ground truth')
plt.plot(x,dm.data['pred_d'][1],'r',label='deep')
plt.plot(x,dm.data['pred_m'][1],'y',label='medium')
plt.plot(x,dm.data['pred_s'][1],'b',label='shallow')
plt.legend()
plt.savefig('prediction.png')
'''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       plot loss                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''
dm.load_np('loss_d','loss_deep.npy')
dm.load_np('loss_m','loss_medium.npy')
dm.load_np('loss_s','loss_shallow.npy')

x=np.array(range(1,len(dm.data['loss_d'])+1))
plt.figure()
plt.plot(x,dm.data['loss_d'],'r',label='deep')
plt.plot(x,dm.data['loss_m'],'y',label='medium')
plt.plot(x,dm.data['loss_s'],'b',label='shallow')
plt.legend()
plt.savefig('loss.png')
'''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       plot mnist_loss                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
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
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       plot accu                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
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
