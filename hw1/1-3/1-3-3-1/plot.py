
from util import Datamanager
import matplotlib.pyplot as plt
import numpy as np
import argparse
assert plt and np

############################################################
#               setting option                             #
############################################################
dm=Datamanager()
#parser = argparse.ArgumentParser(description='setting module parameter.')
#parser.add_argument('-lo','--loss_output', dest='loss_output',type=str,required=True)
#parser.add_argument('-ao','--accu_output', dest='accu_output',type=str,required=True)
#args = parser.parse_args()
############################################################
#               loading data                               #
############################################################
dm.load_np('train','record/train.npy')
dm.load_np('test','record/val.npy')
#x=np.array(range(1,len(dm.data['train'])+1))
############################################################
#               plot loss                                  #
############################################################
fig, ax1 = plt.subplots()
x = np.linspace(-1,2,300)
ax1.plot(x, np.log(dm.data['train'][:,0]), 'b-', label='train')
ax1.plot(x, np.log(dm.data['test'][:,0]), 'b-', linewidth=1.0, linestyle='--', label='test')
ax1.set_xlabel('beta')
#plt.plot(x,dm.data['lost'],'b',label='lost')
#plt.plot(x,dm.data['accu'],'g',label='accu')
ax1.set_ylabel('lost', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(x,dm.data['train'][:,1], 'r', label='train')
ax2.plot(x,dm.data['test'][:,1], 'r', linewidth=1.0, linestyle='--', label='test')

ax2.set_ylabel('accu', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.legend(loc='upper right')
#plt.show()
plt.savefig('beta.png')
############################################################
#               plot loss                                  #
############################################################

