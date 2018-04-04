
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
parser.add_argument('-selo','--sensitive_loss_output', dest='selo',type=str,required=True)
parser.add_argument('-seao','--sensitive_accu_output', dest='seao',type=str,required=True)
parser.add_argument('-shlo','--sharpness_loss_output', dest='shlo',type=str,required=True)
parser.add_argument('-shao','--sharpness_accu_output', dest='shao',type=str,required=True)
args = parser.parse_args()
############################################################
#               loading data                               #
############################################################
dm.load_np('train','record/train.npy')
dm.load_np('test','record/test.npy')
############################################################
#               plot loss                                  #
############################################################
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(dm.data['train'][:,0],dm.data['train'][:,1],c='b',label='train')
ax1.plot(dm.data['test'][:,0],dm.data['test'][:,1],c='b',linewidth=1.0,linestyle='--',label='test')
ax2.plot(dm.data['train'][:,0],dm.data['train'][:,3],c='r',label='sensitivity')
plt.title('Loss Sensitivity')
ax1.set_xlabel('batch size')
ax1.set_ylabel('loss', color='b')
ax1.tick_params('y', colors='b')
ax2.set_ylabel('sensitivity', color='r')
ax2.tick_params('y', colors='r')
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
plt.savefig(args.selo)
############################################################
#               plot accu                                  #
############################################################
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(dm.data['train'][:,0],dm.data['train'][:,2],c='b',label='train')
ax1.plot(dm.data['test'][:,0],dm.data['test'][:,2],c='b',linewidth=1.0,linestyle='--',label='test')
ax2.plot(dm.data['train'][:,0],dm.data['train'][:,3],c='r',label='sensitivity')
plt.title('Accu Sensitivity')
ax1.set_xlabel('batch size')
ax1.set_ylabel('accuracy', color='b')
ax1.tick_params('y', colors='b')
ax2.set_ylabel('sensitivity', color='r')
ax2.tick_params('y', colors='r')
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
plt.savefig(args.seao)
############################################################
#               plot loss                                  #
############################################################
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(dm.data['train'][:,0],dm.data['train'][:,1],c='b',label='train')
ax1.plot(dm.data['test'][:,0],dm.data['test'][:,1],c='b',linewidth=1.0,linestyle='--',label='test')
ax2.plot(dm.data['train'][:,0],dm.data['train'][:,4],c='r',label='sharpness')
plt.title('Loss Sharpness')
ax1.set_ylabel('loss', color='b')
ax1.tick_params('y', colors='b')
ax2.set_ylabel('sharpness', color='r')
ax2.tick_params('y', colors='r')
ax1.legend(loc='upper right')
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
plt.savefig(args.shlo)
############################################################
#               plot accu                                  #
############################################################
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(dm.data['train'][:,0],dm.data['train'][:,2],c='b',label='train')
ax1.plot(dm.data['test'][:,0],dm.data['test'][:,2],c='b',linewidth=1.0,linestyle='--',label='test')
ax2.plot(dm.data['train'][:,0],dm.data['train'][:,4],c='r',label='sharpness')
plt.title('Accu Sharpness')
ax1.set_ylabel('accuracy', color='b')
ax1.tick_params('y', colors='b')
ax2.set_ylabel('sharpness', color='r')
ax2.tick_params('y', colors='r')
ax1.legend(loc='upper right')
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
plt.savefig(args.shao)
