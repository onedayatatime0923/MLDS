
from util import Datamanager,DNN
import matplotlib.pyplot as plt
import numpy as np
import argparse
assert DNN and plt and np

dm=Datamanager()
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-i','--input', dest='input',nargs='+',type=str,required=True)
parser.add_argument('-o','--output', dest='output',type=str,required=True)
args = parser.parse_args()
color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       plot parameter                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
para_total=[]
for i in range(len(args.input)):
    dm.load_np('para{}'.format(i),args.input[i])
    para_total.extend(list(dm.data['para{}'.format(i)][:,1:]))

dm.pca_construct(para_total,2)
for i in range(len(args.input)):
    data=dm.pca_transform(dm.data['para{}'.format(i)][:,1:])
    for j in range(len(data)):
        text='{:.2f}'.format(dm.data['para'+str(i)][j,0])
        plt.text(data[j][0], data[j][1], text,fontdict={'size': 10, 'color': color[i]}) 

plt.xlim(-6, 10)
plt.ylim(-10, 10)

#plt.show()
plt.savefig(args.output)
