
from util import Datamanager , CNN
import matplotlib.pyplot as plt
import numpy as np 

dm=Datamanager() 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       plot norm                                '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm.load_np('norm','sinc_norm.npy') 
print(dm.data['norm'].shape)
print(dm.data['norm'])

x=np.linspace(1,670,670)
plt.figure()
plt.plot(x,dm.data['norm'],'r',label='norm ') 
plt.legend()
plt.savefig('norm.png')

