from keras.layers import Input, Dense #Dense层就是所谓的全连接神经网络层
from keras.models import Model #通过继承 Model 类并在 call 方法中实现你自己的前向传播
from keras import regularizers
import h5py
import numpy as np
import os
import scipy.io as sio

use_regularizer = True
my_regularizer = None
my_epochs = 50

if use_regularizer:
    # add a sparsity constraint on the encoded representations在编码表示上添加稀疏约束
    # note use of 10e-5 leads to blurred results注：使用10e-5会导致结果模糊注
    my_regularizer = regularizers.l1(0.001)
    # and a larger number of epochs as the added regularization the model并以较大数量的时代作为补充正则化模型 不太可能过度适应，可以训练更长时间
    # is less likely to overfit and can be trained longer
    my_epochs = 13


# prepare input data
data_path = os.path.join(os.getcwd(), 'data')
data = sio.loadmat(os.path.join(data_path, 'abu-airport-3.mat'))['data']
labels = sio.loadmat(os.path.join(data_path, 'abu-airport-3.mat'))['map']

# normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
data = data.astype(float)
data -= np.min(data)
data /= np.max(data) -np.min(data)
print('orgin_data.shape:',data.shape)

encoding_dim = 7  # 32 floats -> compression factor 24.5, assuming the input is 784 floats
input_img = Input(shape=(data.shape[2], ))

# "encoded" is the encoded representation of the inputs
encoded = Dense(128, activation='relu',activity_regularizer=my_regularizer)(input_img)
encoded = Dense(encoding_dim * 6, activation='relu')(encoded)
encoded = Dense(encoding_dim *3, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(encoding_dim * 3, activation='relu')(encoded)
decoded = Dense(encoding_dim * 6, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(data.shape[2], activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# Separate Encoder model
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# Separate Decoder model
# create a placeholder for an encoded  input
encoded_input = Input(shape=(encoding_dim, ))

# retrieve the layers of the autoencoder model
decoder_layer1 = autoencoder.layers[-4]
decoder_layer2 = autoencoder.layers[-3]
decoder_layer3 = autoencoder.layers[-2]
decoder_layer4 = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer4(decoder_layer3(decoder_layer2(decoder_layer1(encoded_input)))))

# configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adam', loss='mse')

#输入一个像素的所有波段
patchData = []
for r in range(data.shape[0]):
    for c in range(data.shape[1]):
        patch = data[r:r+1,c:c+1,:]
        patchData.append(patch)
patchData = np.array(patchData)
print('patchData.shape1 after append:',patchData.shape)
patchData = patchData.reshape(-1,patchData.shape[3])
print('patchData.shape2 after append_reshape:',patchData.shape)

#开始训练
autoencoder.fit(patchData, patchData, epochs=my_epochs, batch_size=48, shuffle=True,validation_data=(patchData, patchData),
                verbose=2)#verbose：日志显示verbose = 0 为不在标准输出流输出日志信息verbose = 1 为输出进度条记录verbose = 2 为每个epoch输出一行记录

#保存模型
#autoencoder.save(filepath='./model_save/abu-airport-3.h5')

# encode and decode some digits
encoded_imgs = encoder.predict(patchData)
decoded_imgs = decoder.predict(encoded_imgs)
#print encoded image
print('encoded_img.shape:',encoded_imgs.shape)
encoded_imgs = encoded_imgs.reshape(data.shape[0],data.shape[1],-1)
print('encoded_img.reshape:',encoded_imgs.shape)
sio.savemat('./mat_save2/airport-3encode.mat',mdict={'data':encoded_imgs})

# Visualize the reconstructed encoded representations

re = np.zeros(shape=(data.shape[0]*data.shape[1]))
for i in range(data.shape[0]*data.shape[1]):
    reconstruction_error = (np.sqrt(pow((patchData[i,:] - decoded_imgs[i,:]),2)))
    re[i] = round((np.mean(reconstruction_error)),4)
print('reconstruction_error:',re)
re = re.reshape(data.shape[0],data.shape[1])
list = list(re)
sio.savemat('./mat_save2/airport-3re.mat', mdict={'my_data': list})

