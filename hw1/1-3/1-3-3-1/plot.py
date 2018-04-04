
from util import Datamanager
import matplotlib.pyplot as plt
import numpy as np
import argparse
assert plt and np and argparse

############################################################
#               setting option                             #
############################################################
dm=Datamanager()

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
ax1.plot(x, np.log(dm.data['train'][:,0]), 'b-', label='train_loss')
ax1.plot(x, np.log(dm.data['test'][:,0]), 'b-', linewidth=1.0, linestyle='--', label='test_loss')
ax1.set_xlabel('interpolation ratio')
#plt.plot(x,dm.data['lost'],'b',label='lost')
#plt.plot(x,dm.data['accu'],'g',label='accu')
ax1.set_ylabel('cross_entropy(log scale)', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(x,dm.data['train'][:,1], 'r', label='train_acc')
ax2.plot(x,dm.data['test'][:,1], 'r', linewidth=1.0, linestyle='--', label='test_acc')

ax2.set_ylabel('accuracy', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
#plt.legend(loc='upper right')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
#plt.show()
plt.savefig('beta.png')


