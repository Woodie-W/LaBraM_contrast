import pickle
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import numpy as np

list_path = List[Path]

class SingleShockDataset(Dataset):
    def __init__(self, folder_path: Path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1):
        self.__folder_path = folder_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__length = None
        self.__feature_size = None

        self.__global_idxes = []
        self.__local_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        # 使用with语句打开文件
        with open(self.__folder_path / 'channel_name.pkl', 'rb') as f:
            self.channel_names = pickle.load(f)
        
        # 过滤掉Excel文件
        self.__file_paths = [file for file in self.__folder_path.iterdir() if file.is_file() and file.name != 'channel_name.pkl' and not file.name.endswith('.xlsx')]
        
        # 加载pickle文件并检查内容有效性
        self.__subjects = []
        for path in self.__file_paths:
            with open(path, 'rb') as f:
                subject = pickle.load(f)
                if isinstance(subject, np.ndarray) and subject.ndim == 2:
                    self.__subjects.append(subject)
                else:
                    raise ValueError(f"Invalid subject data in file: {path}")

        global_idx = 0
        for subject in self.__subjects:
            self.__global_idxes.append(global_idx)
            subject_len = subject.shape[1]
            total_sample_num = (subject_len - self.__window_size) // self.__stride_size + 1
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        
        self.__length = global_idx
        self.__feature_size = list(self.__subjects[0].shape)
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
        return self.__subjects[subject_idx][:, item_start_idx:item_start_idx + self.__window_size]

    def get_ch_names(self):
        return self.channel_names


class ShockDataset(Dataset):
    def __init__(self, folder_paths: list_path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0, end_percentage: float = 1):
        self.__folder_paths = folder_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(folder_path=folder_path, window_size=self.__window_size,
                                              stride_size=self.__stride_size, start_percentage=self.__start_percentage,
                                              end_percentage=self.__end_percentage) for folder_path in self.__folder_paths]
        
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        
        self.__length = dataset_idx
        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]

    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()


if __name__ == '__main__':
    dataset = ShockDataset([Path('./pretrain_set/EEG/SEED/60chs')], 200, 200, start_percentage=0, end_percentage=1)
    print(dataset.feature_size)
    print(dataset.get_ch_names())
    print(len(dataset))
    print(dataset[0].shape)
