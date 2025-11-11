import pickle
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import numpy as np
import os
import torch

drug_name_to_id = {
    'Escitalopram': 0,
    'Sertraline': 1,
    'Venlafaxine XR': 2,
    'rTMS_1': 3,
    'rTMS_2': 4,
    'unknown': -1,
}

class SFTSet_embedding(Dataset):
    def __init__(
        self,
        data_path,
        data_path_without_cls,
        clip=False,
        label2int=None, # 保留以兼容接口，但不再使用
        kept_ints=None,
    ):
        self.data_path = data_path
        self.data_path_without_cls = data_path_without_cls
        self.clip = clip
        
        # 1. 扫描文件并构建 info_path 字典
        all_file_paths = []
        info_paths = []
        if data_path_without_cls is not None:
            for subj_path in data_path_without_cls:
                if os.path.isdir(subj_path):
                    for file in os.listdir(subj_path):
                        if file.endswith(".pkl") and "data" in file:
                            all_file_paths.append(os.path.join(subj_path, file))
                        elif file.endswith(".pkl") and "info" in file:
                            info_paths.append(os.path.join(subj_path, file))

        self.id2info_path = dict(
            [[int(path.split("/")[-1].split("_")[0]), path] for path in info_paths]
        )

        # 2. 构建基于 info 文件的“权威”信息字典, sub2label(即remission)无效
        (   self.sub2label, self.sub2drug, self.sub2hamd0, self.sub2delta_hamd,
            label2counts, drug2counts ) = self.analyse_label_and_drug()
        print("Distribution of remission labels from info files:", label2counts)
        print("Distribution of treatment arms from info files:", drug2counts)
        print(f"Total data files found before cleaning: {len(all_file_paths)}")

        # 3. 根据字典，清洗数据文件列表
        valid_subject_ids = set(self.sub2delta_hamd.keys())
        self.file_paths = []
        for path in all_file_paths:
            try:
                total_id = int(path.split("/")[-1].split("_")[0])
                # 只有当一个被试存在于权威名单中，并且其delta_hamd值有效时，才保留
                if total_id in valid_subject_ids and self.sub2delta_hamd[total_id] is not None:
                    self.file_paths.append(path)
            except (ValueError, IndexError):
                # print(f"Warning: Skipping file with invalid name format: {path}")
                continue
        
        print(f"Total samples after cleaning (ensuring valid regression target exists): {len(self.file_paths)}")

        # 4. （可选）根据指定的标签类别 (0或1) 进一步过滤, 过滤的是remission, 这里暂时可以保留
        if kept_ints is not None:
            final_filtered_paths = []
            for path in self.file_paths:
                total_id = int(path.split("/")[-1].split("_")[0])
                # 这里的 .get() 是安全的，因为上一步已经保证了ID存在
                label = self.sub2label.get(total_id)
                if label in kept_ints:
                    final_filtered_paths.append(path)
            self.file_paths = final_filtered_paths
            print(f"Total samples after filtering by kept_ints: {len(self.file_paths)}")

        # 5. 打印最终的标签分布
        final_label_counts = {}
        for path in self.file_paths:
            total_id = int(path.split("/")[-1].split("_")[0])
            label = self.sub2label.get(total_id)
            if label is not None:
                final_label_counts[label] = final_label_counts.get(label, 0) + 1
        print("Final distribution of remission labels in the dataset:", final_label_counts)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, "rb") as f:
            de_features = pickle.load(f)
        
        total_id = int(file_path.split("/")[-1].split("_")[0])
        
        # 因为__init__中的清洗，这里的直接索引现在是绝对安全的
        drug_id = self.sub2drug[total_id]
        hamd0 = self.sub2hamd0[total_id]
        delta_hamd = self.sub2delta_hamd[total_id] # <-- 这将是回归任务的目标

        if self.clip:
            mean = de_features.mean(axis=1, keepdims=True)
            std = de_features.std(axis=1, keepdims=True)
            de_features = np.clip(de_features, mean - std * 3, mean + std * 3)
            threshold = 30
            de_features = np.clip(de_features, -threshold, threshold)
            
        return de_features, delta_hamd, total_id, drug_id, hamd0

    def get_ch_names(self):
        if not self.file_paths:
            # 如果数据集为空，返回一个默认值或抛出错误
            print("Warning: Dataset is empty, cannot get channel names.")
            return []
        dataset_path = os.path.dirname(os.path.dirname(self.file_paths[0]))
        channel_name_path = os.path.join(dataset_path, "channel_name.pkl")
        if os.path.exists(channel_name_path):
            with open(channel_name_path, "rb") as f:
                channel_name = pickle.load(f)
            return channel_name
        else:
            print(f"Warning: channel_name.pkl not found at {channel_name_path}")
            return []


    def get_dataset_name(self):
        if not self.file_paths:
            return "Empty_Dataset"
        dataset_name = os.path.dirname(os.path.dirname(self.file_paths[0]))
        return dataset_name
    
    def analyse_label_and_drug(self):
        sub2label = {}
        sub2drug = {}
        sub2hamd0 = {}
        sub2delta_hamd = {}
        
        for total_id_str, info_path in self.id2info_path.items():
            total_id = int(total_id_str)
            try:
                with open(info_path, 'rb') as f:
                    info = pickle.load(f)

                    # 处理 hamd 标签
                    hamd0_raw = info.get('subject_hamd')
                    delta_hamd_raw = info.get('subject_delta_hamd')
                    sub2hamd0[total_id] = float(hamd0_raw)
                    sub2delta_hamd[total_id] = float(delta_hamd_raw)

                    # 处理 Remission 标签 (可选保留)
                    sub2label[total_id] = info.get('subject_remission')

                    # 处理药物信息
                    if 'subject_treatment_arms' in info and info['subject_treatment_arms'] in drug_name_to_id:
                        drug_name = info['subject_treatment_arms']
                        sub2drug[total_id] = drug_name_to_id[drug_name]
                    else:
                        sub2drug[total_id] = drug_name_to_id['unknown']

            except Exception as e:
                print(f"Warning: Could not process info file {info_path}: {e}")
                sub2label[total_id] = None
                sub2hamd0[total_id] = None
                sub2delta_hamd[total_id] = None
                sub2drug[total_id] = drug_name_to_id['unknown']
        
        label2counts = {}
        drug2counts = {}
        for sub_id, label in sub2label.items():
            if label is not None:
                label2counts[label] = label2counts.get(label, 0) + 1
            
            # drug_id 肯定存在于 sub2drug 中
            drug = sub2drug[sub_id]
            drug2counts[drug] = drug2counts.get(drug, 0) + 1

        return sub2label, sub2drug, sub2hamd0, sub2delta_hamd, label2counts, drug2counts


def custom_collate_fn_with_ids(batch):
    """
    自定义 collate_fn,用于处理回归任务的批次数据。
    这个版本将EEG特征进行动态填充(padding)，并确保输出的特征张量是三维的 (B, C, L),
    同时将所有元数据和目标值转换为Tensor。
    """
    
    # 1. 解包 __getitem__ 返回的所有数据
    de_features_list, delta_hamds, subject_ids, drug_ids, hamd0s = zip(*batch)

    # 2. 对EEG特征进行动态Padding
    max_channels = 0
    max_len = 0
    # 遍历一次以找到当前批次中最大的维度
    for f in de_features_list:
        if f.ndim >= 2:
            max_channels = max(max_channels, f.shape[0])
            max_len = max(max_len, f.shape[1])
            
    # --- START: 核心修改 ---
    # 创建一个用于填充的三维零数组 (B, C, L)
    padded_features_np = np.zeros((len(batch), max_channels, max_len), dtype=np.float32)
    
    # 将每个样本的特征数据复制到这个零数组中
    for i, f in enumerate(de_features_list):
        # 验证输入数据的形状是 (C, L, 1)
        if f.ndim == 3 and f.shape[2] == 1:
            # 使用 .squeeze(-1) 移除最后一个维度
            f_squeezed = f.squeeze(-1)
            c, l = f_squeezed.shape
            padded_features_np[i, :c, :l] = f_squeezed
        elif f.ndim == 2:
            # 如果数据本身就是2维的，直接使用
            c, l = f.shape
            padded_features_np[i, :c, :l] = f
        else:
            # 对于其他不符合预期的形状，打印警告
            print(f"Warning: Skipping feature with unexpected shape {f.shape} in collate_fn.")
    # --- END: 核心修改 ---
            
    # 3. 将所有数据转换为PyTorch张量
    # 现在 padded_features_np 是三维的，转换后 features 也是三维的
    features = torch.from_numpy(padded_features_np)
    delta_hamd_labels = torch.tensor(np.array(delta_hamds), dtype=torch.float32)
    subject_ids_tensor = torch.tensor(np.array(subject_ids), dtype=torch.long)
    drug_ids_tensor = torch.tensor(np.array(drug_ids), dtype=torch.long)
    hamd0_tensor = torch.tensor(np.array(hamd0s), dtype=torch.float32)
    
    # 4. 按照 engine 代码期望的顺序返回打包好的批次数据
    return features, delta_hamd_labels, subject_ids_tensor, drug_ids_tensor, hamd0_tensor


def custom_collate_fn_with_ids(batch):
    """
    自定义 collate_fn,用于处理回归任务的批次数据。
    这个版本将EEG特征进行动态填充(padding)，并确保输出的特征张量是三维的 (B, C, L),
    同时将所有元数据和目标值转换为Tensor。
    """
    
    # 1. 解包 __getitem__ 返回的所有数据
    de_features_list, delta_hamds, subject_ids, drug_ids, hamd0s = zip(*batch)

    # 2. 对EEG特征进行动态Padding
    max_channels = 0; max_len = 0
    
    # 遍历一次以找到当前批次中最大的维度
    for f in de_features_list:
        if f.ndim >= 2:
            max_channels = max(max_channels, f.shape[0])
            max_len = max(max_len, f.shape[1])
            
    # 创建一个用于填充的三维零数组 (B, C, L)
    padded_features_np = np.zeros((len(batch), max_channels, max_len), dtype=np.float32)
    
    # 将每个样本的特征数据复制到这个零数组中
    for i, f in enumerate(de_features_list):
        if f.ndim == 3 and f.shape[2] == 1: # 验证输入数据的形状是 (C, L, 1)
            # 使用 .squeeze(-1) 移除最后一个维度
            f_squeezed = f.squeeze(-1)
            c, l = f_squeezed.shape
            padded_features_np[i, :c, :l] = f_squeezed
        elif f.ndim == 2:
            # 如果数据本身就是2维的，直接使用
            c, l = f.shape
            padded_features_np[i, :c, :l] = f
        else:
            # 对于其他不符合预期的形状，打印警告
            print(f"Warning: Skipping feature with unexpected shape {f.shape} in collate_fn.")
    # --- END: 核心修改 ---
            
    # 3. 将所有数据转换为PyTorch张量
    # padded_features_np 是三维的，转换后 features 也是三维的
    features = torch.from_numpy(padded_features_np)
    delta_hamd_labels = torch.tensor(np.array(delta_hamds), dtype=torch.float32)
    subject_ids_tensor = torch.tensor(np.array(subject_ids), dtype=torch.long)
    drug_ids_tensor = torch.tensor(np.array(drug_ids), dtype=torch.long)
    hamd0_tensor = torch.tensor(np.array(hamd0s), dtype=torch.float32)
    
    # 4. 按照 engine 代码期望的顺序返回打包好的批次数据
    return features, delta_hamd_labels, subject_ids_tensor, drug_ids_tensor, hamd0_tensor



list_path = List[Path]


default_label2int = {
    'HC': 0, 'HEALTHY': 0, 'LowOCD': 0, 'MDD': 1, 'ADHD': 2, 'ADHD ': 2,
    'OCD': 3, 'HighOCD': 3, 'ANXIETY': 4, 'ANXIETY': 4, 'SZ': 5, 'PD': 6, 
    'PARKINSON': 6, 'PD-FOG-': 6, 'PD-FOG+': 6, 'A&A': 7, 'SMC': 8, 'BIPOLAR': 9,
    'BP': 9, 'AD': 10, 'FTD': 11, 'Dp': 12, 'past-MDD': 13, 'MSA-C': 14, 
    'TINNITUS': 15, 'CHRONIC PAIN': 16, 'PAIN': 17, 'INSOMNIA': 18, 'insomnia': 18, 
    'BURNOUT': 19, 'PDD NOS ': 20, 'PDD NOS': 20, 'ASD': 21, 'ASPERGER': 22,
    'TUMOR': 23, 'WHIPLASH': 24, 'TRAUMA': 25, 'TBI': 26, 'mTBI': 26, 'Chronic TBI': 26,
    'CONVERSION DX': 27, 'STROKE ': 28, 'STROKE': 28, 'MIGRAINE': 29, 'LYME': 30,
    'EPILEPSY': 31, 'abnormal': 31, 'DPS': 32, 'ANOREXIA': 33, 'Delirium': 34,
    'Recrudesce': 35, 'Somatic': 36, 'DYSLEXIA': 37, 'DYSPRAXIA': 38, 'GTS': 39, 
    'DYSCALCULIA': 40, 'PTSD': 41, 'PANIC': 42, 'DEPERSONALIZATION': 43, 'Severe-MDD': 44,
    'Mid-MDD': 44,'Light-MDD': 45, 'SuperLight-MDD': 46,  # todo
    'Severe-HC': 47,'Mid-HC': 48,'Light-HC': 49,'SuperLight-HC': 49,
    'Severe-Dp': 50,'Mid-Dp': 51,'Light-Dp': 52,'SuperLight-Dp': 53,
    'Severe-Past-MDD': 54,'Mid-Past-MDD': 55,'Light-Past-MDD': 56, 'SuperLight-Past-MDD': 57,
}


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
                "HC": 0, "HEALTHY": 0, "LowOCD": 0, "AD": 1, "FTD": 2, "PD": 3,
                "PARKINSON": 3, "past-MDD": 4, "MDD": 5, "Dp": 6, "ADHD": 7,
                "ADHD ": 7, "OCD": 8, "HighOCD": 8, "SMC": 9, "CHRONIC PAIN": 10,
                "MSA-C": 11, "DYSLEXIA": 12, "TINNITUS": 13, "INSOMNIA": 14,
                "BURNOUT": 15, "DEPERSONALIZATION": 16, "ANXIETY": 17,
                "BIPOLAR": 18, "PDD NOS ": 19, "PDD NOS": 19, "ASD": 20,
                "ASPERGER": 21, "MIGRAINE" : 22, "PANIC": 23, "TUMOR": 24,
                "WHIPLASH": 25, "PAIN": 26, "CONVERSION DX": 27, "STROKE ": 28,
                "STROKE": 28, "LYME": 29, "PTSD": 30, "EPILEPSY": 31, "abnormal": 31,
                "TRAUMA": 32, "TBI": 33, "DPS": 34, "ANOREXIA" : 35, "DYSPRAXIA": 36,
                "DYSCALCULIA": 37, "GTS" : 38, "mTBI": 39, "SZ": 40, "A&A": 41,
                "Delirium": 42, "PD-FOG" : 43, "PD-FOG+": 44, "Chronic TBI": 45,
                "Recrudesce": 46, "Somatic": 47,
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

