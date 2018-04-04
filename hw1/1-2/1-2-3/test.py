import numpy as np
import os 
for i in range(100):
    os.system('python train.py  -lo loss_'+str(i)+' -ra ratio_'+str(i)+' -u 24 16 4 1 ')
