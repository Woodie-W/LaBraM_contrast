import pickle
import h5py
import numpy as np

def pkl_to_hdf5(pkl_path, hdf5_path, compression=None):
    """
    将 .pkl 文件转换为 .hdf5 文件
    :param pkl_path: 输入的 .pkl 文件路径
    :param hdf5_path: 输出的 .hdf5 文件路径
    :param compression: 压缩方式（如 'gzip'）
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    with h5py.File(hdf5_path, 'w') as hf:
        def _save(hf, name, obj):
            if isinstance(obj, (np.ndarray, list, int, float, str, bytes)):
                hf.create_dataset(name, data=np.array(obj), compression=compression)
            elif isinstance(obj, dict):
                group = hf.create_group(name)
                for key, val in obj.items():
                    _save(group, str(key), val)
            else:
                raise ValueError(f"Unsupported type: {type(obj)}")

        _save(hf, 'data', data)

# 使用示例
if __name__ == "__main__":
    # 替换为实际的 .pkl 文件路径和输出的 .hdf5 文件路径
    pkl_to_hdf5('channel_name.pkl', 'channel_name.hdf5', compression='gzip')