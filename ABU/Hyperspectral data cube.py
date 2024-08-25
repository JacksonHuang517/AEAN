import os
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

import scipy.io as sio
from spectral import *
import wx

# 设置数据集路径
dataset_path = os.path.join('C:/Users/huang/Desktop/dissertation/keras-autoencoders-master/实验/ABU/data')

# 加载高光谱数据
data = sio.loadmat(os.path.join(dataset_path, 'abu-urban-2.mat'))['data']

# 初始化 wx 应用程序
app = wx.App(False)

# 使用 spectral 库可视化高光谱立方体
view_cube(data, bands=[29, 19, 9])

app.MainLoop()
