
from util import Datamanager
import matplotlib.pyplot as plt
import numpy as np
import argparse
assert plt and np

dm=Datamanager()
parser = argparse.ArgumentParser(description='setting module parameter.')
parser.add_argument('-m','--mode', dest='mode',type=str,required=True)
parser.add_argument('-i','--input', dest='input',nargs='+',type=str,required=True)
parser.add_argument('-o','--output', dest='output',type=str,required=True)
args = parser.parse_args()
color=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
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

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title(args.mode)

#plt.show()
plt.savefig(args.output)
