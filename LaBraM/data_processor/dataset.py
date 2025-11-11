import pickle
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import numpy as np
import os
import torch

list_path = List[Path]


class SingleShockDataset(Dataset):

    def __init__(
        self,
        folder_path: Path,
        window_size: int = 200,
        stride_size: int = 1,
        start_percentage: float = 0,
        end_percentage: float = 1,
    ):
        self.__folder_path = folder_path
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.channel_names = pickle.load(
            open(self.__folder_path / "channel_name.pkl", "rb")
        )
        self.__file_paths = [
            file
            for file in self.__folder_path.iterdir()
            if file.is_file()
            and not file.name.startswith("channel_name")
            and not file.name.endswith(".xlsx")
        ]
        # self.__subjects = [pickle.load(open(path, 'rb')) for path in self.__file_paths]

    def __getitem__(self, idx: int):
        # return self.__subjects[idx]
        data = pickle.load(open(self.__file_paths[idx], "rb"))
        return data, str(self.__file_paths[idx])

    def get_ch_names(self):
        return self.channel_names

    def __len__(self):
        return len(self.__file_paths)


class ShockDataset(Dataset):
    def __init__(
        self,
        folder_paths: list_path,
        window_size: int = 200,
        stride_size: int = 1,
        start_percentage: float = 0,
        end_percentage: float = 1,
    ):
        self.__folder_paths = folder_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None

        self.__dataset_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [
            SingleShockDataset(
                folder_path=folder_path,
                window_size=self.__window_size,
                stride_size=self.__stride_size,
                start_percentage=self.__start_percentage,
                end_percentage=self.__end_percentage,
            )
            for folder_path in self.__folder_paths
        ]
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = idx - self.__dataset_idxes[dataset_idx]
        return self.__datasets[dataset_idx][item_idx]

    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()


# def get_multi_label(label):
#     items = []
#     if label is None:
#         items = [None]
#     elif isinstance(label, list):
#         for item in label:
#             items.append(item)
#     else:
#         assert isinstance(label, str)
#         items = [label]
#     return items

# class SFTSet(Dataset):
#     def __init__(self, data_path=None, dataset_path_without_cls_list=None, clip=False, label2int=None, kept_ints=None):
#         """
#         :param data_path: a dict, each contains data from a cls,
#             e,g.: {'HC': [00001, 00002,], 'MDD': [00100, 00101]}
#         :param data_path_without_cls: a list, the cls of which has to be read from f'{total_id}_info.pkl'
#             e.g. [00200, 00201, ...]
#         """
#         # 添加data_path_without_cls的读取
#         self.dataset_path_without_cls_list = dataset_path_without_cls_list
#         data_path_without_cls = []
#         for dataset in dataset_path_without_cls_list:
#             # print('dataset', dataset)

#             data_path_without_cls.extend([os.path.join(dataset, folder) for folder in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, folder))])
#         # print('data_path_without_cls', data_path_without_cls)

#         self.data_path = data_path
#         self.data_path_without_cls = data_path_without_cls
#         self.file_paths = []  # paths of each chunk
#         self.class_labels = []  # for subj in data_path, len(class_labels) < len(file_paths)

#         self.clip = clip

#         channel_paths = []  # paths of channel_locs of each subject
#         label_paths = []  # 1 for eeg, 0 for fnirs
#         info_paths = []

#         # 这里file_paths包括了所有数据，class_labels只包括有标签的数据，channel_paths、label_paths、info_paths每个被试占一个元素
#         # 遍历文件夹,获取所有 pkl 文件路径
#         if data_path is not None:
#             for class_name in data_path:
#                 for subj_path in data_path[class_name]:
#                     if os.path.isdir(subj_path):
#                         for file in os.listdir(subj_path):
#                             if file.endswith('.pkl') and 'data' in file:
#                                 self.file_paths.append(os.path.join(subj_path, file))
#                                 self.class_labels.append(class_name)
#                             elif file.endswith('.pkl') and 'info' in file:
#                                 info_paths.append(os.path.join(subj_path, file))
#                             # elif file.endswith('.pkl') and 'channel' in file:
#                             #     channel_paths.append(os.path.join(subj_path, file))
#                             # elif file.endswith('.pkl') and 'label' in file:
#                             #     label_paths.append(os.path.join(subj_path, file))

#         # 遍历文件夹,获取所有 pkl 文件路径  for data_path_without_cls
#         if data_path_without_cls is not None:
#             for subj_path in data_path_without_cls:
#                 if os.path.isdir(subj_path):
#                     for file in os.listdir(subj_path):
#                         if file.endswith('.pkl') and 'data' in file:
#                             self.file_paths.append(os.path.join(subj_path, file))
#                         elif file.endswith('.pkl') and 'info' in file:
#                             info_paths.append(os.path.join(subj_path, file))
#                         # elif file.endswith('.pkl') and 'channel' in file:
#                         #     channel_paths.append(os.path.join(subj_path, file))
#                         # elif file.endswith('.pkl') and 'label' in file:
#                         #     label_paths.append(os.path.join(subj_path, file))


#         # 存放被试对应的label、channel、info文件路径
#         # self.id2label_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in label_paths])
#         # self.id2channel_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in channel_paths])
#         self.id2info_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in info_paths])

#         # sub2label被试和对应的标签（单标签和多标签），label2counts标签和对应的数量
#         self.sub2label, label2counts = self.analyse_label()
#         print('Distribution of cls labels', label2counts, 'num samples', len(self.file_paths))

#         # todo: 需要检查，后续可能增加
#         if label2int is None:
#             self.label2int = {
#                 'HC': 0, 'HEALTHY': 0, 'LowOCD': 0,
#                 'AD': 1, 'FTD': 2,
#                 'PD': 3, 'PARKINSON': 3,
#                 'past-MDD': 4, 'MDD': 5, 'Dp': 6,
#                 'ADHD': 7, 'ADHD ': 7,
#                 'OCD': 8, 'HighOCD': 8,
#                 'SMC': 9, 'CHRONIC PAIN': 10, 'MSA-C': 11, 'DYSLEXIA': 12, 'TINNITUS': 13,
#                 'INSOMNIA': 14, 'BURNOUT': 15, 'DEPERSONALIZATION': 16, 'ANXIETY': 17, 'BIPOLAR': 18,
#                 'PDD NOS ': 19, 'PDD NOS': 19,
#                 'ASD': 20, 'ASPERGER': 21, 'MIGRAINE': 22, 'PANIC': 23, 'TUMOR': 24,
#                 'WHIPLASH': 25, 'PAIN': 26, 'CONVERSION DX': 27,
#                 'STROKE ': 28, 'STROKE': 28,
#                 'LYME': 29, 'PTSD': 30,
#                 'EPILEPSY': 31, 'abnormal': 31,
#                 'TRAUMA': 32, 'TBI': 33, 'DPS': 34, 'ANOREXIA': 35, 'DYSPRAXIA': 36,
#                 'DYSCALCULIA': 37, 'GTS': 38,
#                 'mTBI': 39,
#                 'SZ': 40,
#                 'A&A': 41,
#                 'Delirium': 42,
#                 'PD-FOG-': 43, 'PD-FOG+': 44,
#                 'Chronic TBI': 45,
#                 'Recrudesce': 46, 'Somatic': 47,  # todo
#             }
#         else:
#             self.label2int = label2int

#         # Delete data with Unknown label 并且删除不需要的标签
#         for idx in reversed(range(len(self.file_paths))):
#             total_id = self.file_paths[idx].split('/')[-1].split('_')[0]
#             if self.sub2label[int(total_id)] in ['unknown', None, ['unknown']]:
#                 del self.file_paths[idx]
#                 if idx < len(self.class_labels):
#                     del self.class_labels[idx]
#             elif kept_ints is not None:
#                 ints = [self.label2int[l] for l in get_multi_label(self.sub2label[int(total_id)])]
#                 print('ints', ints)
#                 used_flag = False
#                 for i in ints:
#                     if i in kept_ints:
#                         used_flag = True
#                 if not used_flag:
#                     del self.file_paths[idx]
#                     if idx < len(self.class_labels):
#                         del self.class_labels[idx]

#         print('num samples, without unknown label', len(self.file_paths))

#         # Viz distribution of labels
#         int2counts = dict()
#         for label in label2counts:  # todo: 检查label2counts， 应该避免重复被试，按照subject_dataset_id计算，如果某个标签对应的被试只有一个就无法对比
#             if label not in self.label2int:
#                 continue
#             if self.label2int[label] in int2counts:
#                 int2counts[self.label2int[label]] += label2counts[label]
#             else:
#                 int2counts[self.label2int[label]] = label2counts[label]
#         int_counts = sorted([(k, v) for k, v in int2counts.items()], key=lambda x: x[1], reverse=True)
#         print('Distribution of cls labels (without unknown):')
#         for i in int_counts:
#             print('label', i[0], [k for k in self.label2int if self.label2int[k] == i[0]], 'counts', i[1],)

#         self.int2counts = int2counts

#     def analyse_label(self):
#         # 获取疾病标签的分布
#         # collect all possible cls labels
#         # sub2label: {total_id: class_label} class_label can be a str list(mutli-label) or a string(single label)
#         sub2label = dict()
#         for idx in range(len(self.class_labels)):
#             total_id = self.file_paths[idx].split('/')[-1].split('_')[0]
#             sub2label[int(total_id)] = self.class_labels[idx]
#         for total_id in self.id2info_path:
#             if total_id in sub2label:
#                 continue
#             with open(self.id2info_path[total_id], 'rb') as f:
#                 sub2label[total_id] = pickle.load(f)['subject_label']
#         # label2counts: {class_label(single): counts}
#         label2counts = dict()
#         for sub, label in sub2label.items():
#             # new_items: a list of labels
#             new_items = get_multi_label(label)
#             for item in new_items:
#                 if item in label2counts:
#                     label2counts[item] += 1
#                 else:
#                     label2counts[item] = 1

#         return sub2label, label2counts

#     def get_ch_names(self):
#         channel_names = []
#         for dataset in self.dataset_path_without_cls_list:
#             channel_names.append(pickle.load(open(os.path.join(dataset, 'channel_name.pkl'), 'rb')))
#         return channel_names[0]

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):

#         # 加载对应索引的 pkl 文件   total id != idx?
#         with open(self.file_paths[idx], 'rb') as f:
#             de_features = pickle.load(f)
#         total_id = self.file_paths[idx].split('/')[-1].split('_')[0]

#         # # 加载对应的通道名称文件,比较被试ID
#         # with open(self.id2channel_path[int(total_id)], 'rb') as f:
#         #     channels = pickle.load(f)

#         # # 加载对应的标签文件，比较被试ID
#         # with open(self.id2label_path[int(total_id)], 'rb') as f:
#         #     labels = pickle.load(f)

#         if idx < len(self.class_labels):  # if class labels is provided,
#             class_label = self.class_labels[idx]
#         else:
#             assert int(total_id) in self.id2info_path
#             with open(self.id2info_path[int(total_id)], 'rb') as f:
#                 info = pickle.load(f)
#                 class_label = info['subject_label']

#         # 给定的是list,Multi-label or single label
#         if isinstance(class_label, list):
#             print('class_label', class_label)
#             class_label = [self.label2int[cl] for cl in class_label]
#         else:
#             assert isinstance(class_label, str)
#             # class_label = [self.label2int[class_label]]
#             class_label = self.label2int[class_label]

#         if self.clip:
#             # todo: should be the same as the snippet in Subject_data_new
#             mean = de_features.mean(axis=1, keepdims=True)
#             std = de_features.std(axis=1, keepdims=True)
#             de_features = np.clip(de_features, mean - std * 3, mean + std * 3)
#             threshold = 30
#             de_features = np.clip(de_features, -threshold, threshold)
#         # length = min(len(de_features), len(channels))
#         # 返回的内容包括：Data(数据):[n_chans,times,feature_dim]，channels(通道位置:三维):[n_chans,3]，labels(1为脑电数据，0为fnirs数据):[n_chans]，total_id(被试ID):int，class_label(标签):int
#         # return de_features, channels, labels, int(total_id), class_label
#         return de_features, class_label

# def get_multi_label(label):
#     items = []
#     if label is None:
#         items = [None]
#     elif isinstance(label, list):
#         for item in label:
#             items.append(item)
#     else:
#         assert isinstance(label, str)
#         items = [label]
#     return items

# class SFTSet(Dataset):
#     def __init__(self, data_path=None, dataset_path_without_cls_list=None, clip=False, label2int=None, kept_ints=None):
#         # 初始化数据路径
#         self.dataset_path_without_cls_list = dataset_path_without_cls_list
#         data_path_without_cls = []
#         for dataset in dataset_path_without_cls_list:
#             data_path_without_cls.extend([os.path.join(dataset, folder) for folder in os.listdir(dataset) if os.path.isdir(os.path.join(dataset, folder))])

#         self.data_path = data_path
#         self.data_path_without_cls = data_path_without_cls
#         self.file_paths = []
#         self.class_labels = []
#         self.clip = clip

#         # 收集所需文件路径
#         info_paths = []
#         if data_path is not None:
#             for class_name in data_path:
#                 for subj_path in data_path[class_name]:
#                     if os.path.isdir(subj_path):
#                         for file in os.listdir(subj_path):
#                             if file.endswith('.pkl') and 'data' in file:
#                                 self.file_paths.append(os.path.join(subj_path, file))
#                                 self.class_labels.append(class_name)
#                             elif file.endswith('.pkl') and 'info' in file:
#                                 info_paths.append(os.path.join(subj_path, file))

#         if data_path_without_cls is not None:
#             for subj_path in data_path_without_cls:
#                 if os.path.isdir(subj_path):
#                     for file in os.listdir(subj_path):
#                         if file.endswith('.pkl') and 'data' in file:
#                             self.file_paths.append(os.path.join(subj_path, file))
#                         elif file.endswith('.pkl') and 'info' in file:
#                             info_paths.append(os.path.join(subj_path, file))

#         self.id2info_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in info_paths])

#         # 生成 label2int 映射
#         if label2int is None:
#             self.label2int = {
#                 'HC': 0, 'HEALTHY': 0, 'LowOCD': 0,
#                 'AD': 1, 'FTD': 2,
#                 'PD': 3, 'PARKINSON': 3,
#                 'past-MDD': 4, 'MDD': 5, 'Dp': 6,
#                 'ADHD': 7, 'ADHD ': 7,
#                 'OCD': 8, 'HighOCD': 8,
#                 'SMC': 9, 'CHRONIC PAIN': 10, 'MSA-C': 11, 'DYSLEXIA': 12, 'TINNITUS': 13,
#                 'INSOMNIA': 14, 'BURNOUT': 15, 'DEPERSONALIZATION': 16, 'ANXIETY': 17, 'BIPOLAR': 18,
#                 'PDD NOS ': 19, 'PDD NOS': 19,
#                 'ASD': 20, 'ASPERGER': 21, 'MIGRAINE': 22, 'PANIC': 23, 'TUMOR': 24,
#                 'WHIPLASH': 25, 'PAIN': 26, 'CONVERSION DX': 27,
#                 'STROKE ': 28, 'STROKE': 28,
#                 'LYME': 29, 'PTSD': 30,
#                 'EPILEPSY': 31, 'abnormal': 31,
#                 'TRAUMA': 32, 'TBI': 33, 'DPS': 34, 'ANOREXIA': 35, 'DYSPRAXIA': 36,
#                 'DYSCALCULIA': 37, 'GTS': 38,
#                 'mTBI': 39,
#                 'SZ': 40,
#                 'A&A': 41,
#                 'Delirium': 42,
#                 'PD-FOG-': 43, 'PD-FOG+': 44,
#                 'Chronic TBI': 45,
#                 'Recrudesce': 46, 'Somatic': 47,
#             }
#         else:
#             self.label2int = label2int

#         # 分析标签并筛选数据
#         self.sub2label, label2counts = self.analyse_label()
#         print('Initial distribution of labels:', label2counts)

#         # 删除不满足条件的样本
#         for idx in reversed(range(len(self.file_paths))):
#             total_id = int(self.file_paths[idx].split('/')[-1].split('_')[0])
#             class_label = self.sub2label.get(total_id)

#             # 删除多标签或标签不符合条件的样本
#             if isinstance(class_label, list) or class_label not in self.label2int or \
#                (kept_ints is not None and self.label2int[class_label] not in kept_ints):
#                 del self.file_paths[idx]
#                 if idx < len(self.class_labels):
#                     del self.class_labels[idx]

#         # 更新最终标签分布
#         label2counts = self.calculate_label_distribution()
#         print('Final distribution of labels (after filtering):', label2counts)

#     def analyse_label(self):
#         sub2label = {}
#         label2counts = {}
#         for idx in range(len(self.class_labels)):
#             total_id = int(self.file_paths[idx].split('/')[-1].split('_')[0])
#             sub2label[total_id] = self.class_labels[idx]

#         for total_id, path in self.id2info_path.items():
#             if total_id in sub2label:
#                 continue
#             with open(path, 'rb') as f:
#                 sub2label[total_id] = pickle.load(f)['subject_label']

#         for _, label in sub2label.items():
#             if isinstance(label, str):
#                 label2counts[label] = label2counts.get(label, 0) + 1

#         return sub2label, label2counts

#     def calculate_label_distribution(self):
#         label2counts = {}
#         for path in self.file_paths:
#             total_id = int(path.split('/')[-1].split('_')[0])
#             label = self.sub2label.get(total_id)
#             if label and label in self.label2int:
#                 label2counts[label] = label2counts.get(label, 0) + 1
#         return label2counts

#     def get_ch_names(self):
#         channel_names = []
#         for dataset in self.dataset_path_without_cls_list:
#             channel_names.append(pickle.load(open(os.path.join(dataset, 'channel_name.pkl'), 'rb')))
#         # print('channel_names', channel_names[0])
#         return channel_names[0]

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         with open(self.file_paths[idx], 'rb') as f:
#             de_features = pickle.load(f)
#         total_id = int(self.file_paths[idx].split('/')[-1].split('_')[0])
#         class_label = self.label2int[self.sub2label[total_id]]
#         # print('class_label', class_label)

#         if self.clip:
#             mean = de_features.mean(axis=1, keepdims=True)
#             std = de_features.std(axis=1, keepdims=True)
#             de_features = np.clip(de_features, mean - std * 3, mean + std * 3)
#             de_features = np.clip(de_features, -30, 30)

#         return de_features, class_label


default_label2int = {
    'HC': 0, 'HEALTHY': 0, 'LowOCD': 0,
    'MDD': 1,
    'ADHD': 2, 'ADHD ': 2,
    'OCD': 3, 'HighOCD': 3,
    'ANXIETY': 4,  ' ANXIETY': 4,
    'SZ': 5,
    'PD': 6, 'PARKINSON': 6, 'PD-FOG-': 6, 'PD-FOG+': 6,
    'A&A': 7,
    'SMC': 8,
    'BIPOLAR': 9, 'BP': 9,
    'AD': 10, 'FTD': 11,
    'Dp': 12,
    'past-MDD': 13,
    'MSA-C': 14,
    'TINNITUS': 15, 'CHRONIC PAIN': 16, 'PAIN': 17,
    'INSOMNIA': 18, 'insomnia': 18,
    'BURNOUT': 19,
    'PDD NOS ': 20, 'PDD NOS': 20, 'ASD': 21, 'ASPERGER': 22,
    'TUMOR': 23,
    'WHIPLASH': 24, 'TRAUMA': 25,
    'TBI': 26, 'mTBI': 26, 'Chronic TBI': 26,
    'CONVERSION DX': 27,
    'STROKE ': 28, 'STROKE': 28,
    'MIGRAINE': 29,
    'LYME': 30,
    'EPILEPSY': 31, 'abnormal': 31,
    'DPS': 32, 'ANOREXIA': 33,
    'Delirium': 34,
    'Recrudesce': 35, 'Somatic': 36,
    'DYSLEXIA': 37, 'DYSPRAXIA': 38, 'GTS': 39, 'DYSCALCULIA': 40,
    'PTSD': 41, 'PANIC': 42, 'DEPERSONALIZATION': 43,
    'Severe-MDD': 44,'Mid-MDD': 44,'Light-MDD': 45,'SuperLight-MDD': 46,  # todo
    'Severe-HC': 47,'Mid-HC': 48,'Light-HC': 49,'SuperLight-HC': 49,
    'Severe-Dp': 50,'Mid-Dp': 51,'Light-Dp': 52,'SuperLight-Dp': 53,
    'Severe-Past-MDD': 54,'Mid-Past-MDD': 55,'Light-Past-MDD': 56, 'SuperLight-Past-MDD': 57,
}

def get_multi_label(label):
    items = []
    if label is None:
        items = [None]
    elif isinstance(label, list):
        if len(label) == 1:
            label = label[0]
            if ',' in label:
                for item in label.split(','):
                    items.append(item.strip())
            elif '，' in label:
                for item in label.split('，'):
                    items.append(item.strip())
            else:
                items.append(label)
        else:
            for item in label:
                items.append(item)
    else:
        assert isinstance(label, str)
        items = [label]
    if 'MDD' in items:
        # 找到 'MDD' 的索引
        scs_index = items.index('MDD')
        # 移动 'MDD' 到第一位
        items.insert(0, items.pop(scs_index))
    return items


class SFTSet_embedding(Dataset):
    def __init__(
        self,
        data_path,
        data_path_without_cls,
        clip=False,
        label2int=None,
        kept_ints=None,
    ):
        """
        :param data_path: a dict, each contains data from a cls,
            e,g.: {'HC': [00001, 00002,], 'MDD': [00100, 00101]}
        :param data_path_without_cls: a list, the cls of which has to be read from f'{total_id}_info.pkl'
            e.g. [00200, 00201, ...]
        """
        self.data_path = data_path
        self.data_path_without_cls = data_path_without_cls
        self.file_paths = []  # paths of each chunk
        self.class_labels = (
            []
        )  # for subj in data_path, len(class_labels) < len(file_paths)

        self.clip = clip

        # channel_paths = []  # paths of channel_locs of each subject
        # label_paths = []  # 1 for eeg, 0 for fnirs
        info_paths = []

        # 这里file_paths包括了所有数据，class_labels只包括有标签的数据，channel_paths、label_paths、info_paths每个被试占一个元素
        # 遍历文件夹,获取所有 pkl 文件路径
        if data_path is not None:
            for class_name in data_path:
                for subj_path in data_path[class_name]:
                    if os.path.isdir(subj_path):
                        for file in os.listdir(subj_path):
                            if file.endswith(".pkl") and "data" in file:
                                self.file_paths.append(os.path.join(subj_path, file))
                                self.class_labels.append(class_name)
                            elif file.endswith(".pkl") and "info" in file:
                                info_paths.append(os.path.join(subj_path, file))
                            # elif file.endswith('.pkl') and 'channel' in file:
                            #     channel_paths.append(os.path.join(subj_path, file))
                            # elif file.endswith('.pkl') and 'label' in file:
                            #     label_paths.append(os.path.join(subj_path, file))

        # 遍历文件夹,获取所有 pkl 文件路径  for data_path_without_cls
        if data_path_without_cls is not None:
            for subj_path in data_path_without_cls:
                if os.path.isdir(subj_path):
                    for file in os.listdir(subj_path):
                        if file.endswith(".pkl") and "data" in file:
                            self.file_paths.append(os.path.join(subj_path, file))
                        elif file.endswith(".pkl") and "info" in file:
                            info_paths.append(os.path.join(subj_path, file))
                        # elif file.endswith('.pkl') and 'channel' in file:
                        #     channel_paths.append(os.path.join(subj_path, file))
                        # elif file.endswith('.pkl') and 'label' in file:
                        #     label_paths.append(os.path.join(subj_path, file))

        # 存放被试对应的label、channel、info文件路径
        # self.id2label_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in label_paths])
        # self.id2channel_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in channel_paths])
        self.id2info_path = dict(
            [[int(path.split("/")[-1].split("_")[0]), path] for path in info_paths]
        )

        # sub2label: {total_id: class_label} class_label can be a str list(mutli-label) or a string(single label)
        self.sub2label, label2counts = self.analyse_label()
        print(
            "Distribution of cls labels",
            label2counts,
            "num samples",
            len(self.file_paths),
        )

        # todo: 需要检查，后续可能增加
        if label2int is None:
            self.label2int = default_label2int
        else:
            self.label2int = label2int

        # Delete data with Unknown label 并且只保留拥有kept_ints中的标签的数据
        for idx in reversed(range(len(self.file_paths))):
            total_id = self.file_paths[idx].split("/")[-1].split("_")[0]
            if self.sub2label[int(total_id)] in ["unknown", None, ["unknown"]]:
                del self.file_paths[idx]
                if idx < len(self.class_labels):
                    del self.class_labels[idx]
            elif kept_ints is not None:
                ints = [
                    self.label2int[l]
                    for l in get_multi_label(self.sub2label[int(total_id)])
                ]
                used_flag = False
                for i in ints:
                    if i in kept_ints:
                        used_flag = True
                if not used_flag:
                    del self.file_paths[idx]
                    if idx < len(self.class_labels):
                        del self.class_labels[idx]

        print("num samples, without unknown label", len(self.file_paths))

        # Viz distribution of labels
        int2counts = dict()
        for (
            label
        ) in (
            label2counts
        ):  # todo: 检查label2counts， 应该避免重复被试，按照subject_dataset_id计算，如果某个标签对应的被试只有一个就无法对比
            if label not in self.label2int:
                continue
            if self.label2int[label] in int2counts:
                int2counts[self.label2int[label]] += label2counts[label]
            else:
                int2counts[self.label2int[label]] = label2counts[label]
        int_counts = sorted(
            [(k, v) for k, v in int2counts.items()], key=lambda x: x[1], reverse=True
        )
        print("Distribution of cls labels (without unknown):")
        for i in int_counts:
            print(
                "label",
                i[0],
                [k for k in self.label2int if self.label2int[k] == i[0]],
                "counts",
                i[1],
            )

        self.int2counts = int2counts

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        # 加载对应索引的 pkl 文件   total id != idx?
        with open(self.file_paths[idx], "rb") as f:
            de_features = pickle.load(f)
        total_id = self.file_paths[idx].split("/")[-1].split("_")[0]

        # # 加载对应的通道名称文件,比较被试ID
        # with open(self.id2channel_path[int(total_id)], 'rb') as f:
        #     channels = pickle.load(f)

        # # 加载对应的标签文件，比较被试ID
        # with open(self.id2label_path[int(total_id)], 'rb') as f:
        #     labels = pickle.load(f)

        if idx < len(self.class_labels):  # if class labels is provided,
            # single label
            class_label = self.class_labels[idx]
        else:
            assert int(total_id) in self.id2info_path
            with open(self.id2info_path[int(total_id)], "rb") as f:
                info = pickle.load(f)
                class_label = info['subject_label']
                if isinstance(class_label, np.ndarray):
                    assert len(class_label) == 1
                    class_label = class_label[0].split(',')
        class_label = [self.label2int[l] for l in get_multi_label(class_label)]
        # if isinstance(class_label, list):
        #     class_label = [self.label2int[cl] for cl in class_label]
        # else:
        #     assert isinstance(class_label, str)
        #     class_label = [self.label2int[class_label]]

        if self.clip:
            # todo: should be the same as the snippet in Subject_data_new
            mean = de_features.mean(axis=1, keepdims=True)
            std = de_features.std(axis=1, keepdims=True)
            de_features = np.clip(de_features, mean - std * 3, mean + std * 3)
            threshold = 30
            de_features = np.clip(de_features, -threshold, threshold)
        # length = min(len(de_features), len(channels))
        # 返回的内容包括：Data(数据):[n_chans,times,feature_dim]，channels(通道位置:三维):[n_chans,3]，labels(1为脑电数据，0为fnirs数据):[n_chans]，total_id(被试ID):int，class_label(标签):int
        # return de_features, channels, labels, int(total_id), class_label
        return de_features, class_label

    def get_ch_names(self):
        dataset_path = os.path.dirname(os.path.dirname(self.file_paths[0]))
        channel_name = pickle.load(open(os.path.join(dataset_path, "channel_name.pkl"), "rb"))
        # print('channel_names', channel_names[0])
        return channel_name

    def get_dataset_name(self):
        dataset_name = os.path.dirname(os.path.dirname(self.file_paths[0]))
        return dataset_name

    def analyse_label(self):
        # 获取疾病标签的分布
        # collect all possible cls labels
        # sub2label: {total_id: class_label} class_label can be a str list(mutli-label) or a string(single label)
        sub2label = dict()
        for idx in range(len(self.class_labels)):
            total_id = self.file_paths[idx].split("/")[-1].split("_")[0]
            sub2label[int(total_id)] = self.class_labels[idx]
        for total_id in self.id2info_path:
            if total_id in sub2label:
                continue
            with open(self.id2info_path[total_id], 'rb') as f:
                sub2label[total_id] = pickle.load(f)['subject_label']
                if isinstance(sub2label[total_id], np.ndarray):
                    assert len(sub2label[total_id]) == 1
                    print('[total_id][0]', sub2label[total_id][0])
                    sub2label[total_id] = sub2label[total_id][0].split(',')

        label2counts = dict()
        for sub, label in sub2label.items():
            # new_items: a list of labels
            new_items = get_multi_label(label)
            for item in new_items:
                if item in label2counts:
                    label2counts[item] += 1
                else:
                    label2counts[item] = 1

        return sub2label, label2counts


def custom_collate_fn(batch, multi_label=False):
    """
    自定义 collate_fn 函数
    :param batch: 一个批次的数据,每个元素为一个元组(de_features, class_label)
    :param multi_label: 是否多标签
    :return: 填充后的 de_features(batchsize, channel_num, window_num, embedding_dim),class_label(batch_size,5)
    """

    num_samples = len(batch)
    features = []
    if multi_label:
        all_labels = np.ones((num_samples, 5), dtype=int) * (
            -1
        )  # 假设每个被试最多有5个标签，不足用-1填充
    else:
        all_labels = np.zeros(num_samples, dtype=int)

    # 填充每个样本
    for i, sample in enumerate(batch):
        feature, class_label = sample
        features.append(feature)

        if multi_label:
            assert len(class_label) <= 5
            all_labels[i, : len(class_label)] = np.array(class_label, dtype=int)
        else:
            if isinstance(class_label, list):  # SFTSet返回的class_label总是list
                assert len(class_label) == 1
                all_labels[i] = class_label[0]
            else:
                all_labels[i] = class_label

    # 将数组转换为张量
    features = np.array(features, dtype=np.float32)
    features = torch.tensor(features, dtype=torch.float32)
    all_labels = torch.tensor(all_labels, dtype=torch.long)  # cls label
    return features, all_labels


class SFTSet(Dataset):
    def __init__(
        self,
        data_path=None,
        dataset_path_without_cls_list=None,
        clip=False,
        label2int=None,
        kept_ints=None,
    ):
        self.dataset_path_without_cls_list = dataset_path_without_cls_list
        data_path_without_cls = []
        for dataset in dataset_path_without_cls_list:
            data_path_without_cls.extend(
                [
                    os.path.join(dataset, folder)
                    for folder in os.listdir(dataset)
                    if os.path.isdir(os.path.join(dataset, folder))
                ]
            )

        self.data_path = data_path
        self.data_path_without_cls = data_path_without_cls
        self.file_paths = []
        self.class_labels = []
        self.clip = clip
        info_paths = []

        # 获取数据和信息文件的路径
        if data_path is not None:
            for class_name in data_path:
                for subj_path in data_path[class_name]:
                    if os.path.isdir(subj_path):
                        for file in os.listdir(subj_path):
                            if file.endswith(".pkl") and "data" in file:
                                self.file_paths.append(os.path.join(subj_path, file))
                                self.class_labels.append(class_name)
                            elif file.endswith(".pkl") and "info" in file:
                                info_paths.append(os.path.join(subj_path, file))

        if data_path_without_cls is not None:
            for subj_path in data_path_without_cls:
                if os.path.isdir(subj_path):
                    for file in os.listdir(subj_path):
                        if file.endswith(".pkl") and "data" in file:
                            self.file_paths.append(os.path.join(subj_path, file))
                        elif file.endswith(".pkl") and "info" in file:
                            info_paths.append(os.path.join(subj_path, file))

        self.id2info_path = dict(
            [[int(path.split("/")[-1].split("_")[0]), path] for path in info_paths]
        )

        if label2int is None:
            self.label2int = {
                "HC": 0,
                "HEALTHY": 0,
                "LowOCD": 0,
                "AD": 1,
                "FTD": 2,
                "PD": 3,
                "PARKINSON": 3,
                "past-MDD": 4,
                "MDD": 5,
                "Dp": 6,
                "ADHD": 7,
                "ADHD ": 7,
                "OCD": 8,
                "HighOCD": 8,
                "SMC": 9,
                "CHRONIC PAIN": 10,
                "MSA-C": 11,
                "DYSLEXIA": 12,
                "TINNITUS": 13,
                "INSOMNIA": 14,
                "BURNOUT": 15,
                "DEPERSONALIZATION": 16,
                "ANXIETY": 17,
                "BIPOLAR": 18,
                "PDD NOS ": 19,
                "PDD NOS": 19,
                "ASD": 20,
                "ASPERGER": 21,
                "MIGRAINE": 22,
                "PANIC": 23,
                "TUMOR": 24,
                "WHIPLASH": 25,
                "PAIN": 26,
                "CONVERSION DX": 27,
                "STROKE ": 28,
                "STROKE": 28,
                "LYME": 29,
                "PTSD": 30,
                "EPILEPSY": 31,
                "abnormal": 31,
                "TRAUMA": 32,
                "TBI": 33,
                "DPS": 34,
                "ANOREXIA": 35,
                "DYSPRAXIA": 36,
                "DYSCALCULIA": 37,
                "GTS": 38,
                "mTBI": 39,
                "SZ": 40,
                "A&A": 41,
                "Delirium": 42,
                "PD-FOG-": 43,
                "PD-FOG+": 44,
                "Chronic TBI": 45,
                "Recrudesce": 46,
                "Somatic": 47,
            }
        else:
            self.label2int = label2int

        # 筛选并调整标签
        self.sub2label, label2counts = self.analyse_label()
        print("Initial distribution of labels:", label2counts)

        for idx in reversed(range(len(self.file_paths))):
            total_id = int(self.file_paths[idx].split("/")[-1].split("_")[0])
            class_label = self.sub2label.get(total_id)

            # 如果是多标签，选取第一个在 label2int 中的标签
            if isinstance(class_label, list):
                valid_labels = [lbl for lbl in class_label if lbl in self.label2int]
                if valid_labels:
                    class_label = valid_labels[0]  # 选取第一个符合的标签
                else:
                    # 如果没有符合的标签，删除该样本
                    del self.file_paths[idx]
                    if idx < len(self.class_labels):
                        del self.class_labels[idx]
                    continue

            # 如果不是在 label2int 或 kept_ints 中，则删除
            if class_label not in self.label2int or (
                kept_ints is not None and self.label2int[class_label] not in kept_ints
            ):
                del self.file_paths[idx]
                if idx < len(self.class_labels):
                    del self.class_labels[idx]
            else:
                # 将单标签数据的最终标签更新到 sub2label
                self.sub2label[total_id] = class_label

        label2counts = self.calculate_label_distribution()
        print("Final distribution of labels (after filtering):", label2counts)

    def analyse_label(self):
        sub2label = {}
        label2counts = {}
        for idx in range(len(self.class_labels)):
            total_id = int(self.file_paths[idx].split("/")[-1].split("_")[0])
            sub2label[total_id] = self.class_labels[idx]

        for total_id, path in self.id2info_path.items():
            if total_id in sub2label:
                continue
            with open(path, "rb") as f:
                sub2label[total_id] = pickle.load(f)["subject_label"]

        for _, label in sub2label.items():
            if isinstance(label, str):
                label2counts[label] = label2counts.get(label, 0) + 1

        return sub2label, label2counts

    def calculate_label_distribution(self):
        label2counts = {}
        for path in self.file_paths:
            total_id = int(path.split("/")[-1].split("_")[0])
            label = self.sub2label.get(total_id)
            if label and label in self.label2int:
                label2counts[label] = label2counts.get(label, 0) + 1
        return label2counts

    def __len__(self):
        return len(self.file_paths)

    def get_ch_names(self):
        channel_names = []
        for dataset in self.dataset_path_without_cls_list:
            channel_names.append(
                pickle.load(open(os.path.join(dataset, "channel_name.pkl"), "rb"))
            )
        # print('channel_names', channel_names[0])
        return channel_names[0]

    def __getitem__(self, idx):
        with open(self.file_paths[idx], "rb") as f:
            de_features = pickle.load(f)
        total_id = int(self.file_paths[idx].split("/")[-1].split("_")[0])
        class_label = self.label2int[self.sub2label[total_id]]

        if self.clip:
            mean = de_features.mean(axis=1, keepdims=True)
            std = de_features.std(axis=1, keepdims=True)
            de_features = np.clip(de_features, mean - std * 3, mean + std * 3)
            de_features = np.clip(de_features, -30, 30)

        return de_features, class_label


if __name__ == "__main__":
    # dataset = ShockDataset([Path('./pretrain_set/EEG/SEED/60chs')], 200, 200, start_percentage=0, end_percentage=1)
    # print(dataset.get_ch_names())
    # print(len(dataset))
    # print(dataset[0].shape)

    dataset = SFTSet(
        data_path=None,
        dataset_path_without_cls_list=[
            "/data1/wangkuiyu/model_update_code/labram_resting_fine_pool/resting_eye_open/Parkinson_eyes_open_PROCESSED/61chs"
        ],
        clip=False,
        label2int={
            "HC": 0,
            "HEALTHY": 0,
            "LowOCD": 0,
            "PD": 1,
            "PARKINSON": 1,
        },
        kept_ints=None,
    )
    print(dataset.get_ch_names())
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1])
