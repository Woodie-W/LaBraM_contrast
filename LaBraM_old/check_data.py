# import pickle
# import numpy as np
# from pathlib import Path

# def check_data(folder_path: Path):
#     flag = {"invalid type": 0, "NaN": 0, "Infinite": 0, "valid": 0, "non-numeric": 0}
#     file_paths = [file for file in folder_path.iterdir() if file.is_file() and not file.name.startswith('channel_name') and not file.name.endswith('.xlsx')]
#     for path in file_paths:
#         with open(path, 'rb') as f:
#             data = pickle.load(f)
#             print(f"Checking file: {path}, shape: {data.shape}")
#             if not isinstance(data, np.ndarray):
#                 print(f"Invalid data type in file: {path}")
#                 flag["invalid type"] += 1
#             elif np.isnan(data).any():
#                 print(f"NaN values found in file: {path}")
#                 nan_indices = np.argwhere(np.isnan(data))
#                 print(f"NaN indices: {nan_indices}")
#                 flag["NaN"] += 1
#             elif np.isinf(data).any():
#                 print(f"Infinite values found in file: {path}")
#                 inf_indices = np.argwhere(np.isinf(data))
#                 print(f"Infinite indices: {inf_indices}")
#                 flag["Infinite"] += 1
#             elif not np.issubdtype(data.dtype, np.number):
#                 print(f"Non-numeric values found in file: {path}")
#                 flag["non-numeric"] += 1
#             else:
#                 # print(f"Data in file {path} is valid.")
#                 flag["valid"] += 1
#     return flag

# datasets_train = [
#     ['./pretrain_set/EEG/SEED/60chs'],
#     ['./pretrain_set/EEG/LEMON/61chs'],
#     ['./pretrain_set/EEG/SXMU_1_PROCESSED/HC/61chs'],
#     ['./pretrain_set/EEG/HBN_1/129chs'],
#     ['./pretrain_set/EEG/HUAWEI_EEG/30chs'],
#     ['./pretrain_set/EEG/HUAWEI_EEG/59chs'],
#     ['./pretrain_set/EEG/SXMU_1_PROCESSED/MDD/72chs'],
#     ['./pretrain_set/EEG/SXMU_1_PROCESSED/HC/72chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/train/abnormal/19chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/train/normal/19chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_events/eval/19chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_epilepsy/00_epilepsy/19chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_epilepsy/01_no_epilepsy/19chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_slowing/19chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_abnormal/train/abnormal/20chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_events/eval/20chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_events/train/19chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_epilepsy/00_epilepsy/20chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_epilepsy/01_no_epilepsy/20chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_slowing/20chs'],
#     ['./pretrain_set/EEG/TUHEEG_PROCESSED/tuh_eeg_epilepsy/00_epilepsy/17chs'],
# ]

# results = {}

# for dataset in datasets_train:
#     for path in dataset:
#         folder_path = Path(path)
#         if folder_path.exists():
#             flag = check_data(folder_path)
#             results[path] = flag
#         else:
#             print(f"Path {folder_path} does not exist.")

# for k, v in results.items():
#     print(f"Dataset: {k}, Results: {v}")

import pickle

chs = pickle.load(open('fine_pool/EEG/SXMU_2_PROCESSED/HC/59chs/channel_name.pkl', "rb"))
print(chs)