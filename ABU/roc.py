# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
#import HyperProTool as hyper
from scipy import io as sio
import os

target = sio.loadmat(os.path.join(os.getcwd(), './daxiu/airport3_score2.mat'))['data']
labels = sio.loadmat(os.path.join(os.getcwd(), './data/abu-airport-3.mat'))['map']

target = target.reshape((1,-1))
#target = target.astype(np.float64)
print('target:',target.shape)

rows, cols = labels.shape
label = labels.reshape(1,rows*cols)
result = np.zeros((1,rows*cols))
for i in range(rows*cols):
    result[0,i] = np.linalg.norm(target[:,i])
fpr,tpr,threshold = metrics.roc_curve(label.transpose(),result.transpose(),pos_label=1)
auc = metrics.auc(fpr,tpr)
plt.figure(1,facecolor='white',dpi=100)
plt.plot(fpr,tpr, color='red' ,label = 'Urban AUC{:.6f}'.format(auc))
plt.grid(color='grey', linestyle='--', linewidth=0.5,alpha=0.8)
plt.xlabel('false positive rate')
plt.ylabel('true posive rate')
plt.xlim([0,0.1])
plt.legend(loc='lower right')
plt.show()
print(auc)



