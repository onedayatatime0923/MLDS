
from util import Datamanager
import matplotlib.pyplot as plt
import numpy as np
import argparse
assert plt and np

############################################################
#               setting option                             #
############################################################
dm=Datamanager()
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-lo','--loss_output', dest='loss_output',type=str,required=True)
parser.add_argument('-ao','--accu_output', dest='accu_output',type=str,required=True)
args = parser.parse_args()
############################################################
#               loading data                               #
############################################################
dm.load_np('train','record/train.npy')
dm.load_np('test','record/test.npy')
x=np.array(range(1,len(dm.data['train'])+1))
############################################################
#               plot loss                                  #
############################################################
plt.figure()
plt.scatter(dm.data['train'][:,0],dm.data['train'][:,1],c='b',label='train')
plt.scatter(dm.data['test'][:,0],dm.data['test'][:,1],c='g',label='test')
plt.title('Loss')
plt.legend()
plt.savefig(args.loss_output)
############################################################
#               plot loss                                  #
############################################################
plt.figure()
plt.scatter(dm.data['train'][:,0],dm.data['train'][:,2],c='b',label='train')
plt.scatter(dm.data['test'][:,0],dm.data['test'][:,2],c='g',label='test')
plt.title('Accu')
plt.legend()
plt.savefig(args.accu_output)
