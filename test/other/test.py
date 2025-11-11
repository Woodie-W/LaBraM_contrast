import pickle

# 1. 打开并加载 .pkl 文件
with open('15947_info.pkl', 'rb') as f:  # 注意必须是二进制模式 'rb'
    data = pickle.load(f)

# 2. 查看内容（根据实际数据类型操作）
print(type(data))  # 检查对象类型
print(data)        # 如果是字典、列表等可直接打印
try:
    print(data.shape)
    print(data.dtype)
finally:
    pass
