
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
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''       plot parameter                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
para_total=[]
for i in range(len(args.input)):
    dm.load_np('para{}'.format(i),args.input[i])
    para_total.extend(list(dm.data['para{}'.format(i)]))

dm.pca_construct(para_total,2)
for i in range(len(args.input)):
    data=dm.pca_transform(dm.data['para{}'.format(i)])
    plt.scatter(data[:][0], data[:][1], s=1) 

plt.xticks(())  # ignore xticks
plt.yticks(())  # ignore yticks

plt.show()
plt.savefig(args.output)
'''
'''
