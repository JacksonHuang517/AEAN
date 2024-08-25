import numpy as np
from PIL import Image
from scipy import io as sio
from matplotlib import pyplot as plt

scoremat = sio.loadmat('./daxiu/airport3_score2.mat')['data']
scoremat = sio.loadmat('./mat_save/beach-2score.mat')['data']
re = sio.loadmat('./mat_save/airport-4re.mat')['my_data']
dis = sio.loadmat('./daxiu/beach2_distance.mat')['data']
#score灰度图

dis_map = 255*(dis-np.min(dis))/(np.max(dis)-np.min(dis))
dis_map = dis_map.astype(np.int32)
dis_fig = Image.fromarray(dis_map)
plt.figure(1)
plt.imshow(dis_fig,cmap=plt.get_cmap('hot'))
plt.axis('off')
plt.show()

scoremat = 255*(scoremat-np.min(scoremat))/(np.max(scoremat)-np.min(scoremat))
scoremat = scoremat.astype(np.int32)
dis_map = Image.fromarray(scoremat)
dis_map.show()
plt.figure(1)
plt.imshow(scoremat,cmap=plt.get_cmap('hot'))
plt.show()

