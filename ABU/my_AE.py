import numpy as np
import os
import scipy.io as sio
import cv2 as cv
from sklearn.decomposition import PCA
from keras.models import load_model

encoding_dim = 7
#model = load_model('./model_save/m1.h5')
data_path = os.path.join(os.getcwd(), 'data')
data = sio.loadmat(os.path.join(data_path, 'abu-urban-1.mat'))['data']
labels = sio.loadmat(os.path.join(data_path, 'abu-urban-1.mat'))['map']
encoded_imgs = sio.loadmat('./mat_save/urban-4encode.mat')['data']
re = sio.loadmat('./mat_save/urban-4re.mat')['my_data']
data = data.astype(float)
data -= np.min(data)
data /= np.max(data)-np.min(data)

def applyPCA(X,numComponents=75):
    newX = np.reshape(X,(-1,X.shape[2]))
    pca = PCA(n_components=numComponents,whiten=False)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX,(X.shape[0],X.shape[1],numComponents))
    return newX,pca.explained_variance_ratio_
PCAdata,pca =applyPCA(data,numComponents=4)

#属性滤波
PCAdata_mean = np.zeros((PCAdata.shape[0],PCAdata.shape[1]))
for r in range(PCAdata.shape[0]):
    for c in range(PCAdata.shape[1]):
        PCAdata_mean[r,c] = np.mean(PCAdata[r,c,:])
kernel = np.ones((7,7),np.uint8)
dilation = cv.dilate(PCAdata_mean,kernel,iterations = 1) #膨胀 iterations指的是膨胀的次数
erosion = cv.erode(PCAdata_mean,kernel,iterations = 1) #腐蚀

opening = cv.morphologyEx(PCAdata_mean,cv.MORPH_OPEN,kernel)
closing = cv.morphologyEx(PCAdata_mean,cv.MORPH_CLOSE,kernel)
d_opening = abs(PCAdata_mean-opening)
d_closing = abs(PCAdata_mean-closing)
A = d_opening + d_closing
print('A:',A)

#三维数据边缘填充
windowsize =3
margin = int((windowsize-1)/2)
top_size,bottom_size,left_size,right_size = (margin,margin,margin,margin)
N = (windowsize * 2) + (windowsize - 2) * 2
padded_PCAdata_mean = np.zeros((PCAdata_mean.shape[0]+2*margin,PCAdata_mean.shape[1]+2*margin))
padded_PCAdata_mean = cv.copyMakeBorder(PCAdata_mean,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_REPLICATE)
#sio.savemat('./mat_save/padded_encode_img.mat',mdict={'data':padded_encoded_img})

dis_list = []
window_centerIndex = []
window_centerpadIndex = []
outwindow_index = []

#2计算周围像素距离,也是对的
for r in range(margin,padded_PCAdata_mean.shape[0]-margin):
    for c in range(margin,padded_PCAdata_mean.shape[1]-margin):
        window_center_pad = np.tile(padded_PCAdata_mean[r:r+1,c:c+1], (windowsize - 2, windowsize - 2))
        window_centerIndex.append(padded_PCAdata_mean[r:r+1,c:c+1])
        window_centerpadIndex.append(window_center_pad)
window_centerIndex = np.array(window_centerIndex)
window_centerpadIndex = np.array(window_centerpadIndex)
print(window_centerIndex.shape)

for r in range(margin,padded_PCAdata_mean.shape[0]-margin):
    for c in range(margin,padded_PCAdata_mean.shape[1]-margin):
        outwindow_patch = padded_PCAdata_mean[r-margin:r+margin+1,c-margin:c+margin+1]
        outwindow_index.append(outwindow_patch)
outwindow_index = np.array(outwindow_index)
print(outwindow_index.shape)
#for i in range(padded_PCAdata_mean.shape[0]*padded_PCAdata_mean.shape[1]):
for i in range(len(outwindow_index)):
        outwindow_index[i,1:windowsize-1,1:windowsize-1] = window_centerpadIndex[i,:,:]

#for i in range(padded_PCAdata_mean.shape[0] * padded_PCAdata_mean.shape[1]):
for i in range(len(outwindow_index)):
    dis = abs(outwindow_index[i,:,:] - window_centerIndex[i,:,:])
    dis = np.sum(dis)
    dis = dis/(N)
    dis_list.append(dis)
dis_list = (np.array(dis_list)).reshape(data.shape[0],data.shape[1])
#sio.savemat('./mat_save/distance.mat',mdict={'data':dis_list})
print('临近像素距离差2：',dis_list)

#对score进行平均化
score = re*A*dis_list
print('score:',score)
sio.savemat('./mat_save/urban-4score.mat',mdict={'data':score})
