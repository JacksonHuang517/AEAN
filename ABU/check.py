import scipy.io as sio

# 导入.mat文件
data_path = './data/abu-urban-2.mat'  # 替换为您的.mat文件路径
mat_data = sio.loadmat(data_path)

# 列出.mat文件中的所有变量名称
print("变量名称:")
for key in mat_data.keys():
    print(key)

# 假设数据包含在一个变量中，我们可以进一步检查它的形状
# 请将 'data' 替换为实际变量名
if 'data' in mat_data:
    data = mat_data['data']
    print(f"\n'数据形状: {data.shape}")
else:
    print("\n请确认数据的变量名称，并将 'data' 替换为实际的变量名称。")
