import numpy as np
import matplotlib.pyplot as plt
import os

loss_list = []
ratio_list = []

for i in range(100):
    loss = np.load('result/loss_'+str(i)+'.npy')
    loss_list.append(loss[0])
    ratio = np.load('result/ratio_'+str(i)+'.npy')
    ratio_list.append(ratio[0])

loss_array = np.array(loss_list)
ratio_array = np.array(ratio_list)
title_font = {'fontname':'Arial', 'size':'32', 'color':'black', 'weight':'normal',
 'verticalalignment':'bottom'}
plt.scatter( ratio_array , loss_array , alpha = 0.6 )
plt.xlabel('minimun_ratio')
plt.ylabel(u'loss')
plt.savefig('1-2-3_result.png')
plt.show()

