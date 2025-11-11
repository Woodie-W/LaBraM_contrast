"""
Consider that there are n datasets (e.g. SEED, FACED, ..., (EEG+fNRIS), ...)
Each dataset has a partial of all possible channels
(e.g. SEED has only EEG chanels while some data has only fNRIS channels)
Each dataset is organized as [sub1.pkl, sub2.pkl, ...],
where each pkl is numpy array of shape[c, n_second * sample_rate]
(n_second and sample_rate can also be different across datasets),
this code would
    1. preprocess each dataset:
        1.1 slice data into windows (in unit of seconds, e.g. 10s / 5s)
        1.2 extract features (e.g. DE),
        1.3 and then save features in another pkl file
    2. build a pytorch Dataset that reads features randomly, and pad each datum to
        [max_n_channels, max_window_length, n_features],
        then return padded sample, and the pad mask (chan_pad) in __getitem__
"""
import time
import logging
import psutil
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import mne
import os

import ast
from multiprocessing import Pool
from functools import partial
from scipy import signal
from mne.io import RawArray
import pandas as pd
from eeg_channels import eeg_locs

# 预处理先切片保存，10s左右一个，每次前进五s有重叠。
# 提取de特征时按照1s提取。
# get_item不能用属性维护，初始的时候要指定好idex对应的样本。
# getitem返回的时候，dataloder的时候使用collate_fn
# 返回position(返回通道名)
# 每个batch batchsize*chan*window*feature


# 近红外频率太低。或者简单做下采样插值，把每秒17个值降为5个值。或者就用17个点。


# 一个预训练数据应该有的类型

class Subject_data():

    def __init__(self):
        self.subject_total_id = None  # 被试在所有数据集里的编号
        self.subject_gender = None  # 0 for female, 1 for male
        self.subject_age = None  # the age of the subject
        self.subject_dataset = None  # 数据集名称
        self.subject_id_dataset = None  # 这个被试在这个数据集里的编号
        self.subject_file_name = None  # 这个被处理的数据文件的原始名称
        self.subject_dataset_note = 'This is a dataset for LEM'  # 用于概括数据集的作用
        self.subject_label = None  # 一些关于被试的标记
        self.file_format = None
        self.hh = None

        self.eeg_info = 'no'  # 有EEG数据
        self.fnirs_info = 'no'  # 有近红外数据
        self.fmri_info = 'no'  # 用于未来数据集
        self.mri_info = 'no'  # 用于未来数据集

        self.eeg = None  # 具体的数据信息
        self.fnirs = None
        self.fmri = None
        self.mri = None
        self.preprocess = None
        self.add_preprocess_history()  # 初始化参数

    def add_eeg_data(self, device, channels, data, note):
        self.eeg = self.EEG_data(device, channels, data, note)
        self.eeg_info = 'yes'

    def add_fnirs_data(self, device, channels, data, note):
        self.fnirs = self.FNIRS_data(device, channels, data, note)
        self.fnirs_info = 'yes'

    def add_preprocess_history(self):  # 这一条全靠变量赋予
        self.preprocess = self.Preprocess_info()

    class EEG_data:
        def __init__(self, device, channels, data, note, eeg_locs):
            self.eeg_device = device
            self.eeg_channels = channels  # 这个被试真实有效的通道，不要参考通道
            self.eeg_chn_locs = eeg_locs  # 128个通道的位置信息  {'Fz':(1,1),'Oz':(100,12)}
            self.eeg_data = data  # 128个通道的信息  {'Fz':array, 'Oz':array, 'Cz':None, ..., 'Trigger': array}
            self.trigger_note = note  # trigger含义 '0 for positive video, 1 for neutral video , 2 for negative'

            # 后添加的
            self.fs = 250  # 采样率(250Hz)
            self.freqs = [[1, 4], [4, 8], [8, 14], [14, 30], [30, 47]]  # 五个频率段
            self.window_size = 10  # 单个切片的窗口大小(10秒)
            self.de_window_size = 1  # 提取DE特征的窗口大小(1秒)
            self.step = 5  # 切片窗口移动的步长(5秒)

        def save_data(self, save_folder, total_id):
            # 获取每个通道的值,通道名称在real内并且其值不为None,并且通道名称也不为'Trigger'
            eeg_data_array = np.array([self.eeg_data[chn] for chn in self.eeg_channels if
                                       chn in self.eeg_data and self.eeg_data[chn] is not None and chn != 'Trigger'])

            # 归一化整个数据序列
            eeg_data_array = (eeg_data_array - np.mean(eeg_data_array, axis=1, keepdims=True)) / np.std(eeg_data_array,
                                                                                                        axis=1,
                                                                                                        keepdims=True)

            # 对归一化后的数据进行裁剪,将数值限制在-10到10的范围内
            eeg_data_array = np.clip(eeg_data_array, -10, 10)

            # 获取通道数和每个通道的长度
            n_chns, n_samples = eeg_data_array.shape

            # 获取通道名称列表,排除'Trigger'通道
            channels = [chn for chn in self.eeg_channels if
                        chn in self.eeg_data and self.eeg_data[chn] is not None and chn != 'Trigger']

            # 根据channel数量来创建文件夹
            save_path = os.path.join(save_folder, str(n_chns) + 'chs')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 保存channel名称文件
            channel_filename = f'channel_name.pkl'
            if not os.path.exists(os.path.join(save_path, channel_filename)):
                channel_filepath = os.path.join(save_path, channel_filename)
                with open(channel_filepath, 'wb') as f:
                    pickle.dump(channels, f)  # 将通道名称保存为pkl文件
            else:
                flag = 0
                for file in os.listdir(save_path):
                    if file.startswith('channel_name'):
                        if channels == pickle.load(open(os.path.join(save_path, file), 'rb')):
                            flag = 1
                            # print("chs:",channels)
                            # print("load chs:", pickle.load(open(os.path.join(save_path, file), 'rb')))
                            break
                if flag == 0:
                    channel_filename = f'channel_name_{total_id}.pkl'
                    channel_filepath = os.path.join(save_path, channel_filename)
                    with open(channel_filepath, 'wb') as f:
                        pickle.dump(channels, f)

            # 保存数据切片文件
            # 对应每份数据取多少的时间长度
            second_dict = {60: 4, 61: 4, 129: 2, 30: 8, 59: 4, 72: 4, 19: 13, 20: 13, 17: 15}
            len_windows = second_dict[n_chns] * 200

            # 步长 len_windows * 0.8
            slices = []
            for start in range(0, n_samples - len_windows + 1, int(len_windows * 0.8)):
                end = start + len_windows
                slices.append(eeg_data_array[:, start:end])
            # 保存切片数据，分了被试
            subject_path = os.path.join(save_path, str(total_id))
            if not os.path.exists(subject_path):
                os.makedirs(subject_path)
            for i, slice_data in enumerate(slices):
                filepath = os.path.join(subject_path, f'{total_id}_eeg_data_{i}.pkl')
                with open(filepath, 'wb') as f:
                    pickle.dump(slice_data, f)

            # # 步长 200
            # slices = []
            # for start in range(0, n_samples - len_windows + 1, 200):
            #     end = start + len_windows
            #     slices.append(eeg_data_array[:, start:end])
            # # 保存切片数据 , 没有分被试
            # for i, slice_data in enumerate(slices):
            #     filepath = os.path.join(save_path, f'{total_id}_eeg_data_{i}.pkl')  # 构建文件路径
            #     with open(filepath, 'wb') as f:
            #         pickle.dump(slice_data, f)  # 将切片后的EEG数据保存为pkl文件

            # # 将被试的通道数追加到Excel文件中
            # excelpath = os.path.join(save_path, 'sub_chan_num.xlsx')
            # if os.path.exists(excelpath):
            #     df = pd.read_excel(excelpath)
            # else:
            #     df = pd.DataFrame(columns=["Subject ID", "Number of Channels", "EEG Channels"])

            # new_row = pd.DataFrame(
            #         {"Subject ID": [total_id], "Number of Channels": [n_chns], "EEG Channels": [channels]})
            # df = pd.concat([df, new_row], ignore_index=True)

            # df.to_excel(excelpath, index=False)
            return str(n_chns) + 'chs'

        def filter(self, data):
            n_freqs = len(self.freqs)  # 获取频率段数量

            filtered_data = np.zeros((n_freqs, *data.shape))

            # 遍历每个频率段
            for freq_idx, freq_range in enumerate(self.freqs):
                low_freq, high_freq = freq_range  # 获取当前频率段的上下限
                # 对当前窗口数据进行滤波
                data_filt = mne.filter.filter_data(data, self.fs, l_freq=low_freq, h_freq=high_freq, verbose=False)
                filtered_data[freq_idx] = data_filt
            # Drop start and end
            assert filtered_data.shape[-1] > 2 * self.fs, (f'eeg too short! {filtered_data.shape}')
            filtered_data = filtered_data[..., int(self.fs):-int(self.fs)]
            return filtered_data

        def extract_de_features(self, data_window):
            # data_window[n_freqs, *], data of each band
            # 提取单个窗口的DE特征
            n_freqs = len(self.freqs)  # 获取频率段数量

            de_features = np.zeros((data_window.shape[1], n_freqs))  # 初始化DE特征数组

            # 遍历每个频率段
            for freq_idx, freq_range in enumerate(self.freqs):
                # 计算当前频率段的DE特征
                data_window_filt = data_window[freq_idx]
                de_one = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(data_window_filt, axis=1, keepdims=True)))
                de_features[:, freq_idx] = de_one.squeeze()  # 将DE特征存储在对应的频率段列中

            return de_features  # 返回DE特征数组

        def sliding_window_extract_de(self, save_folder, total_id):

            # 滑动窗口提取DE特征并保存为pkl文件
            # 获取每个通道的值,通道名称在real内并且其值不为None,并且通道名称也不为'Trigger'
            eeg_data_array = np.array([self.eeg_data[chn] for chn in self.eeg_channels if
                                       chn in self.eeg_data and self.eeg_data[chn] is not None and chn != 'Trigger'])

            # 归一化整个数据序列
            eeg_data_array = (eeg_data_array - np.mean(eeg_data_array, axis=1, keepdims=True)) / np.std(eeg_data_array,
                                                                                                        axis=1,
                                                                                                        keepdims=True)
            # 对归一化后的数据进行裁剪,将数值限制在-10到10的范围内
            # if (np.abs(eeg_data_array) > 10).astype(np.float32).mean() > 1e-3:
            #     raise ValueError(f'too many outliers! {eeg_data_array.shape}, '
            #                      f'outlier ratio: {(np.abs(eeg_data_array) > 10).astype(np.float32).mean().item()}')
            eeg_data_array = np.clip(eeg_data_array, -10, 10)

            # filter the whole data sequence first
            t0 = time.time()
            eeg_data_array = self.filter(eeg_data_array)  # [n_freqs, nchns, n_samples]
            t1 = time.time()
            # 获取通道数和每个通道的长度
            _, n_chns, n_samples = eeg_data_array.shape

            # 脑电标签，脑电设定为1
            labels = np.ones(n_chns, dtype=int)
            # print('label:', labels.shape)
            # print(labels)
            # 获取通道名称列表,排除'Trigger'通道
            channels = [chn for chn in self.eeg_channels if
                        chn in self.eeg_data and self.eeg_data[chn] is not None and chn != 'Trigger'
                        ]

            # 获取每个通道的位置信息，channels_local数组的维度是[通道数，3（三个坐标点）]
            channels_local = np.array([self.eeg_chn_locs[chn] for chn in channels])

            # print(channels_local[0])
            # 计算单个窗口的采样点数
            window_size_samples = int(self.window_size * self.fs)
            # 计算单个de特征提取窗口的采样点数
            de_window_size_samples = int(self.de_window_size * self.fs)

            # 计算总窗口数量
            n_windows = (n_samples - window_size_samples) // (self.step * int(self.fs)) + 1

            # 初始化DE特征列表
            all_de_features = []

            # 滑动窗口提取DE特征并保存
            # 循环处理每个窗口
            for window in range(n_windows):

                start = int(window * self.step * self.fs)  # 计算当前窗口的起始位置
                end = int(start + window_size_samples)  # 计算当前窗口的结束位置

                # 提取所有通道的单个窗口的数据
                data_window = eeg_data_array[..., start:end]

                # 提取DE特征
                # =====================================================================
                # data_window = data_window.reshape(len(self.freqs), n_chns,
                #                                   window_size_samples // de_window_size_samples, de_window_size_samples)
                # de_window = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(data_window, axis=-1)))  # [len(self.freqs), n_chns, 10s]
                # de_window = np.transpose(de_window, (1, 2, 0))
                # ----------------------------------------------------------------------
                # 初始化单个窗口的de特征
                de_window = np.zeros((n_chns, window_size_samples // de_window_size_samples, len(self.freqs)))
                # 提取DE特征(使用向量化操作同时处理所有通道)
                for start_de in range(0, window_size_samples, de_window_size_samples):
                    end_de = start_de + de_window_size_samples
                    data_de_window = data_window[..., start_de:end_de]
                    de_window[:, start_de // de_window_size_samples] = self.extract_de_features(data_de_window)
                # =====================================================================

                # 保存为pkl文件
                filename = f'{total_id}_eeg_de_features_window_{window}.pkl'  # 构建文件名
                channel_filename = f'{total_id}_eeg_channel.pkl'
                label_filename = f'{total_id}_eeg_label.pkl'

                # 保存路径
                save_path = os.path.join(save_folder, str(total_id))

                # 创建保存路径,每个被试单独创建一个文件夹
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                filepath = os.path.join(save_path, filename)  # 构建文件路径
                with open(filepath, 'wb') as f:
                    pickle.dump(de_window, f)  # 将DE特征保存为pkl文件

                channel_filepath = os.path.join(save_path, channel_filename)
                with open(channel_filepath, 'wb') as f:
                    pickle.dump(channels_local, f)  # 将通道位置保存为pkl文件

                label_filepath = os.path.join(save_path, label_filename)
                with open(label_filepath, 'wb') as f:
                    pickle.dump(labels, f)  # 将通道位置保存为pkl文件

                # all_de_features.append(de_window)  # 将当前窗口的DE特征添加到列表中(这是所有的de特征，不是单个窗口的)
                # return de_window, channels

            t2 = time.time()
            print(f'eeg process takes {t1 - t0}s in filter and {t2 - t1}s in DE extraction, save {n_windows} windows')
            return all_de_features, channels_local  # 返回所有窗口的DE特征和通道名称列表

    class FNIRS_data:
        def __init__(self, device, channels, data, note):
            self.fnirs_device = device
            self.fnirs_channels = channels  # 这个数据真实有效的近红外通道 {'(20.2,102.2,120)','()'}
            self.fnirs_chn_locs = {}  # 所有数据综合的近红外通道信息
            self.fnirs_data = data  # 真实的数据信息
            self.dataset_fnirs_note = note  # trigger含义，可以后期补

            # 后面加的
            self.fs = 17  # 采样率(2Hz)
            self.freqs = [0.01, 0.2]  # 指定的频率提取范围
            self.window_size = 10  # 单个切片的窗口大小(10秒)
            self.de_window_size = 1  # 提取DE特征的窗口大小(1秒)
            self.step = 5  # 切片窗口移动的步长(5秒)

        def sliding_window_extract_fe(self, save_folder, total_id, HBT=False):

            # 单独处理每个通道
            if HBT == False:
                # 滑动窗口提取DE特征并保存为pkl文件
                # 获取每个通道的值,通道名称在real内并且其值不为None,并且通道名称也不为'Trigger'
                fnirs_data_array = np.array([self.fnirs_data[chn] for chn in self.fnirs_channels if
                                             chn in self.fnirs_data and self.fnirs_data[
                                                 chn] is not None and chn != 'Trigger'])
                # 获取通道名称列表,排除'Trigger'通道
                channels = [chn for chn in self.fnirs_channels if
                            chn in self.fnirs_data and self.fnirs_data[chn] is not None and chn != 'Trigger']
                # print(channels[0])
                # 通道位置保存
                channels_local = []
                for chn in channels:
                    # 使用ast.literal_eval将字符串转换为tuple
                    coords = ast.literal_eval(chn)
                    # fnirs位置数据的第四个没有用，只保留前三个
                    channels_local.append(coords[:3])

                # 将列表转换为numpy数组
                channels_local = np.array(channels_local)
            # 计算HBT=HBO-HBR，合并相同位置的通道
            else:
                merged_channels = {}
                incomplete_channels = []
                incomplete_folder = 'incomplete_FNIRS'  # HBO或HBR数据不完整的
                channels_local = []
                for chn in self.fnirs_channels:
                    if chn in self.fnirs_data and self.fnirs_data[chn] is not None and chn != 'Trigger':
                        coords = ast.literal_eval(chn)
                        loc = tuple(coords[:3])
                        label = coords[3]
                        if loc not in merged_channels:
                            merged_channels[loc] = {'HBO': None, 'HBR': None}
                        merged_channels[loc][label] = self.fnirs_data[chn]

                # 获取合并后的通道数据
                fnirs_data_array = []
                for loc, data in merged_channels.items():
                    # 通道位置保存
                    channels_local.append(loc)
                    if data['HBO'] is not None and data['HBR'] is not None:
                        merged_data = data['HBO'] - data['HBR']
                        fnirs_data_array.append(merged_data)
                    else:
                        incomplete_channels.append(loc)
                fnirs_data_array = np.array(fnirs_data_array)
                channels_local = np.array(channels_local)
                # 检查是否有不完整的通道数据,即只有HBO或者HBR
                if len(incomplete_channels) > 0:
                    # 将不完整通道的ID保存到指定文件夹
                    incomplete_path = os.path.join(incomplete_folder, str(total_id))
                    if not os.path.exists(incomplete_path):
                        os.makedirs(incomplete_path)
                    incomplete_filename = f'{total_id}_incomplete_channels.pkl'
                    incomplete_filepath = os.path.join(incomplete_path, incomplete_filename)
                    with open(incomplete_filepath, 'wb') as f:
                        pickle.dump(incomplete_channels, f)

            # 归一化整个数据序列
            fnirs_data_array = (fnirs_data_array - np.mean(fnirs_data_array, axis=1, keepdims=True)) / np.std(
                    fnirs_data_array,
                    axis=1,
                    keepdims=True)
            # 对归一化后的数据进行裁剪,将数值限制在-10到10的范围内
            fnirs_data_array = np.clip(fnirs_data_array, -10, 10)

            # 对整个数据序列进行滤波
            # fnirs_data_array = mne.filter.filter_data(fnirs_data_array, self.fs, l_freq=self.freqs[0],
            #                                           h_freq=self.freqs[1])
            assert fnirs_data_array.shape[-1] > 2 * self.fs, (f'fnirs too short! {fnirs_data_array.shape}')
            fnirs_data_array = fnirs_data_array[..., 2 * int(self.fs):]  # fixme: drop start to offset fnirs_data_array

            # 获取通道数和每个通道的长度
            n_chns, n_samples = fnirs_data_array.shape

            # 脑电标签,fnirs设定为0
            labels = np.zeros(n_chns, dtype=int)
            # print('label:', labels.shape)
            # print(labels)

            # 计算单个窗口的采样点数
            window_size_samples = int(self.window_size * self.fs)

            # 计算总窗口数量
            n_windows = int((n_samples - window_size_samples) // (self.step * int(self.fs))) + 1

            # 初始化特征列表
            all_de_features = []

            # 滑动窗口提取DE特征并保存
            # 循环处理每个窗口
            for window in range(n_windows):
                start = int(window * self.step * self.fs)  # 计算当前窗口的起始位置
                end = start + window_size_samples  # 计算当前窗口的结束位置
                # print(f"start: {start}, type: {type(start)}")
                # print(f"end: {end}, type: {type(end)}")
                # start = int(start)
                # end = int(end)

                # 提取所有通道的单个窗口的数据
                data_window = fnirs_data_array[:, start:end]

                # 对当前窗口进行降采样
                downsampled_window = mne.filter.resample(data_window, down=int(self.fs / 5), npad='auto', verbose=False)

                # 获取降采样后的长度
                downsampled_len = downsampled_window.shape[1]  # 降采样后的长度

                # 对每个通道进行插值
                n_chns = downsampled_window.shape[0]

                # =========================================================================
                # new_len = 50
                # interpolated_data = signal.resample(downsampled_window, new_len, axis=-1)
                # -----------------------------------------------------------------------
                interpolated_data = np.zeros((n_chns, 50))
                # print('interpolate', downsampled_window.shape, interpolated_data.shape)
                for i in range(n_chns):
                    # 对第i个通道的数据进行插值
                    old_len = downsampled_len
                    new_len = 50
                    interpolated_data[i, :] = signal.resample(downsampled_window[i, :], new_len)
                # ==============================================================================
                # print(interpolated_data.shape)
                # 重构downsampled_window为 (n_chns, window_size_samples // de_window_size_samples, 5)
                n_chns, n_samples = interpolated_data.shape
                n_segments = n_samples // 5
                reshaped_window = interpolated_data.reshape(n_chns, n_segments, 5)

                # 保存为pkl文件
                filename = f'{total_id}_fnirs_de_features_window_{window}.pkl'  # 构建文件名
                channel_filename = f'{total_id}_fnirs_channel.pkl'
                label_filename = f'{total_id}_fnirs_label.pkl'

                # 保存路径
                save_path = os.path.join(save_folder, str(total_id))

                # 创建保存路径,每个被试单独创建一个文件夹
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                filepath = os.path.join(save_path, filename)  # 构建文件路径
                with open(filepath, 'wb') as f:
                    pickle.dump(reshaped_window, f)  # 将降采样后的数据保存为pkl文件

                channel_filepath = os.path.join(save_path, channel_filename)
                with open(channel_filepath, 'wb') as f:
                    pickle.dump(channels_local, f)  # 将通道位置保存为pkl文件

                label_filepath = os.path.join(save_path, label_filename)
                with open(label_filepath, 'wb') as f:
                    pickle.dump(labels, f)  # 将通道位置保存为pkl文件

                # all_de_features.append(downsampled_window)  # 将当前窗口的降采样数据添加到列表中

            return all_de_features, channels_local  # 返回所有窗口的降采样数据和通道名称列表
            # # 滑动窗口提取DE特征并保存为pkl文件
            # # 获取每个通道的值,通道名称在real内并且其值不为None,并且通道名称也不为'Trigger'
            # eeg_data_array = np.array([self.fnirs_data[chn] for chn in self.fnirs_channels if
            #                            chn in self.fnirs_data and self.fnirs_data[
            #                                chn] is not None and chn != 'Trigger'])
            # # 获取通道数和每个通道的长度
            # n_chns, n_samples = eeg_data_array.shape
            #
            # # 脑电标签，fnirs设定为0
            # labels = np.zeros(n_chns, dtype=int)
            # print('label:', labels.shape)
            # print(labels)
            #
            # # 获取通道名称列表,排除'Trigger'通道
            # channels = [chn for chn in self.fnirs_channels if
            #             chn in self.fnirs_data and self.fnirs_data[chn] is not None and chn != 'Trigger']
            # print(channels[0])
            #
            # channels_local = []
            # for chn in channels:
            #     # 使用ast.literal_eval将字符串转换为tuple
            #     coords = ast.literal_eval(chn)
            #     channels_local.append(coords)
            #
            # # 将列表转换为numpy数组
            # channels_local = np.array(channels_local)
            #
            # print(channels_local[1])
            #
            # # 计算单个窗口的采样点数
            # window_size_samples = self.window_size * self.fs
            #
            # # 计算单个de特征提取窗口的采样点数
            # de_window_size_samples = self.de_window_size * self.fs
            #
            # # 计算总窗口数量
            # n_windows = (n_samples - window_size_samples) // (self.step * self.fs) + 1
            #
            # # 初始化DE特征列表
            # all_de_features = []
            #
            # # 滑动窗口提取DE特征并保存
            # # 循环处理每个窗口
            # for window in range(n_windows):
            #
            #     start = window * self.step * self.fs  # 计算当前窗口的起始位置
            #     end = start + window_size_samples  # 计算当前窗口的结束位置
            #
            #     # 提取所有通道的单个窗口的数据
            #     data_window = eeg_data_array[:, start:end]
            #
            #     # 提取DE特征
            #     # 初始化单个窗口的de特征
            #     de_window = np.zeros((n_chns, window_size_samples // de_window_size_samples, len(self.freqs)))
            #
            #     # 提取DE特征(使用向量化操作同时处理所有通道)
            #     for start_de in range(0, window_size_samples, de_window_size_samples):
            #         end_de = start_de + de_window_size_samples
            #         data_de_window = data_window[:, start_de:end_de]
            #         de_window[:, start_de // de_window_size_samples] = self.extract_de_features(data_de_window)
            #
            #     # 保存为pkl文件
            #     filename = f'{total_id}_fnirs_de_features_window_{window}.pkl'  # 构建文件名
            #     channel_filename = f'{total_id}_fnirs_channel.pkl'
            #     label_filename = f'{total_id}_fnirs_label.pkl'
            #     # 保存路径
            #     save_path = os.path.join(save_folder, total_id)
            #
            #     # 创建保存路径,每个被试单独创建一个文件夹
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #
            #     filepath = os.path.join(save_path, filename)  # 构建文件路径
            #     with open(filepath, 'wb') as f:
            #         pickle.dump(de_window, f)  # 将DE特征保存为pkl文件
            #
            #     channel_filepath = os.path.join(save_path, channel_filename)
            #     with open(channel_filepath, 'wb') as f:
            #         pickle.dump(channels_local, f)  # 将通道位置保存为pkl文件
            #
            #     label_filepath = os.path.join(save_path, label_filename)
            #     with open(label_filepath, 'wb') as f:
            #         pickle.dump(labels, f)  # 将通道位置保存为pkl文件
            #
            #     # all_de_features.append(de_window)  # 将当前窗口的DE特征添加到列表中(这是所有的de特征，不是单个窗口的)
            #     # return de_window,channels
            # return all_de_features, channels_local  # 返回所有窗口的DE特征和通道名称列表

    class Preprocess_info:
        def __init__(self):
            # EEG预处理留档
            self.eeg_preprocess_pipeline = []
            self.eeg_freq = None
            self.eeg_delete_chns = []
            self.eeg_bad_channels = []
            self.eeg_freq_bands1 = []
            self.eeg_ica_or_no = 'no'
            self.eeg_rereference = 'common average'
            self.eeg_freq_bands2 = []
            self.eeg_note = None

            # fNIRS预处理留档
            self.fnirs_preprocess_pipeline = []
            self.fnirs_freq = None
            self.fnirs_delete_chns = []  # 实际上是插值的通道
            self.fnirs_freq_bands1 = []
            self.fnirs_note = None

    def conbine_eeg_fnirs(self, save_folder, total_id):

        save_path = os.path.join(save_folder, str(total_id))

        # 获取eeg和fnirs数据的文件路径
        eeg_files = [f for f in os.listdir(save_path) if f.startswith(f'{total_id}_eeg_de_features')]
        fnirs_files = [f for f in os.listdir(save_path) if f.startswith(f'{total_id}_fnirs_de_features')]

        # 确保eeg和fnirs文件数量相同
        # assert len(eeg_files) == len(fnirs_files), "EEG and fNIRS file counts do not match"

        # 确定数量少的是eeg还是fnirs
        if len(eeg_files) < len(fnirs_files):
            min_files = eeg_files
            max_files = fnirs_files
            print(
                    f"EEG文件数量少于fNIRS文件数量,分别为{len(eeg_files)}和{len(fnirs_files)},少了{len(fnirs_files) - len(eeg_files)}个文件")
        elif len(eeg_files) > len(fnirs_files):
            min_files = fnirs_files
            max_files = eeg_files
            print(
                    f"fNIRS文件数量少于EEG文件数量,分别为{len(fnirs_files)}和{len(eeg_files)},少了{len(eeg_files) - len(fnirs_files)}个文件")
        else:
            min_files = eeg_files

        # 合并eeg和fnirs数据
        for min_file in min_files:

            # 从文件名中提取窗口索引
            window_index = int(min_file.split('_')[-1].split('.')[0])

            # 根据窗口索引找到对应的eeg和fnirs文件
            eeg_file = f'{total_id}_eeg_de_features_window_{window_index}.pkl'
            fnirs_file = f'{total_id}_fnirs_de_features_window_{window_index}.pkl'
            eeg_path = os.path.join(save_path, eeg_file)
            fnirs_path = os.path.join(save_path, fnirs_file)

            with open(eeg_path, 'rb') as f:
                eeg_data = pickle.load(f)

            with open(fnirs_path, 'rb') as f:
                fnirs_data = pickle.load(f)

            combined_data = np.concatenate((eeg_data, fnirs_data), axis=0)

            combined_filename = f'{total_id}_eeg_fnirs_de_features_window_{window_index}.pkl'
            combined_path = os.path.join(save_path, combined_filename)
            with open(combined_path, 'wb') as f:
                pickle.dump(combined_data, f)

        # 删除原始的EEG和fNIRS文件
        for file in eeg_files + fnirs_files:
            file_path = os.path.join(save_path, file)
            os.remove(file_path)
        # # 合并eeg和fnirs数据
        # for i, (eeg_file, fnirs_file) in enumerate(zip(eeg_files, fnirs_files)):
        #     eeg_path = os.path.join(save_path, eeg_file)
        #     fnirs_path = os.path.join(save_path, fnirs_file)
        #
        #     with open(eeg_path, 'rb') as f:
        #         eeg_data = pickle.load(f)
        #
        #     with open(fnirs_path, 'rb') as f:
        #         fnirs_data = pickle.load(f)
        #
        #     combined_data = np.concatenate((eeg_data, fnirs_data), axis=0)
        #
        #     combined_filename = f'{total_id}_eeg_fnirs_de_features_window_{i}.pkl'
        #     combined_path = os.path.join(save_path, combined_filename)
        #     with open(combined_path, 'wb') as f:
        #         pickle.dump(combined_data, f)
        #
        #     # 删除原始eeg和fnirs文件
        #     os.remove(eeg_path)
        #     os.remove(fnirs_path)

        # 合并通道文件
        eeg_channel_path = os.path.join(save_path, f'{total_id}_eeg_channel.pkl')
        fnirs_channel_path = os.path.join(save_path, f'{total_id}_fnirs_channel.pkl')

        with open(eeg_channel_path, 'rb') as f:
            eeg_channels = pickle.load(f)

        with open(fnirs_channel_path, 'rb') as f:
            fnirs_channels = pickle.load(f)

        combined_channels = np.concatenate((eeg_channels, fnirs_channels), axis=0)

        combined_channel_path = os.path.join(save_path, f'{total_id}_eeg_fnirs_channel.pkl')
        with open(combined_channel_path, 'wb') as f:
            pickle.dump(combined_channels, f)

        # 删除原始通道文件
        os.remove(eeg_channel_path)
        os.remove(fnirs_channel_path)

        # 合并标签文件
        eeg_label_path = os.path.join(save_path, f'{total_id}_eeg_label.pkl')
        fnirs_label_path = os.path.join(save_path, f'{total_id}_fnirs_label.pkl')

        with open(eeg_label_path, 'rb') as f:
            eeg_labels = pickle.load(f)

        with open(fnirs_label_path, 'rb') as f:
            fnirs_labels = pickle.load(f)

        combined_labels = np.concatenate((eeg_labels, fnirs_labels), axis=0)

        combined_label_path = os.path.join(save_path, f'{total_id}_eeg_fnirs_label.pkl')
        with open(combined_label_path, 'wb') as f:
            pickle.dump(combined_labels, f)

        # 删除原始标签文件
        os.remove(eeg_label_path)
        os.remove(fnirs_label_path)


def set_affinity(process_id, num_cores):
    p = psutil.Process(os.getpid())  # Get the current process
    # Assign the CPU core, wrap around if process_id exceeds available cores
    core_id = process_id % num_cores
    p.cpu_affinity([core_id])  # Set the process to only run on the assigned core


def process_subject(file_path, save_folder):
    total_id = int(file_path.split('/')[-1].replace('sub_', ''))
    set_affinity(total_id, 64)

    try:
        with open(file_path, 'rb') as f:
            subject = pickle.load(f)
        # 如果已经存在一个被试文件夹就不再处理这个被试数据
        if os.path.exists(os.path.join(save_folder, str(total_id))):  # fixme
            return
        if int(subject.subject_total_id) != total_id:
            print('wrong total id', subject.subject_total_id, total_id)
        pid = os.getpid()
        cpu_list = os.sched_getaffinity(pid)
        print('process', pid, 'total_id', total_id, 'run in cpu:', cpu_list)
        from types import MethodType
        # subject.conbine_eeg_fnirs = MethodType(Subject_data.conbine_eeg_fnirs, subject)
        if subject.eeg_info == 'yes':
            # print(list(vars(subject.eeg).keys()))
            subject.eeg.save_data = MethodType(Subject_data.EEG_data.save_data, subject.eeg)
            subject.eeg.sliding_window_extract_de = MethodType(Subject_data.EEG_data.sliding_window_extract_de,
                                                               subject.eeg)
            subject.eeg.extract_de_features = MethodType(Subject_data.EEG_data.extract_de_features, subject.eeg)
            subject.eeg.filter = MethodType(Subject_data.EEG_data.filter, subject.eeg)
            subject.eeg.window_size = 10
            subject.eeg.fs = subject.preprocess.eeg_freq
            subject.eeg.de_window_size = 1  # fixme: tunable
            if '/SLEEP/' in file_path:
                subject.eeg.step = 30  # todo
            else:
                subject.eeg.step = 5
            subject.eeg.freqs = [[1, 4], [4, 8], [8, 14], [14, 30], [30, 47]]

        if subject.fnirs_info == 'yes':
            # print(list(vars(subject.fnirs).keys()))
            # from types import MethodType
            subject.fnirs.sliding_window_extract_fe = MethodType(Subject_data.FNIRS_data.sliding_window_extract_fe,
                                                                 subject.fnirs)
            subject.fnirs.window_size = 10
            subject.fnirs.fs = subject.preprocess.fnirs_freq
            subject.fnirs.de_window_size = 1  # todo: tunable
            subject.fnirs.step = 5
            subject.fnirs.freqs = [0.01, 0.2]

        print('to process', 'label', subject.subject_label,
              'dataset', subject.subject_dataset,
              'total id', subject.subject_total_id,
              'subject id', subject.subject_id_dataset)
        if subject.eeg is not None:
            # subject.eeg.sliding_window_extract_de(save_folder, total_id)
            str_chs = subject.eeg.save_data(save_folder, total_id)

        # # #     raise NotImplementedError
        # # print(subject.fnirs.fnirs_channels)
        # if subject.fnirs is not None:
        #     # 只针对EEG，先直接跳过
        #     pass
        #     #HBT=True时合并相同位置通道，计算HBT=HBO-HBR作为通道的值，HBT=False时，则单独处理每个通道。
        #     subject.fnirs.sliding_window_extract_fe(save_folder, total_id, HBT=False)
        # #     # raise NotImplementedError
        # if (subject.eeg is not None) and (subject.fnirs is not None):
        #     # 只针对EEG，先直接跳过
        #     pass
        #     subject.conbine_eeg_fnirs(save_folder, total_id)
        #     # raise NotImplementedError

        # Save subject info
        info_filename = f'{total_id}_info.pkl'
        info = {'subject_label': subject.subject_label, 'subject_id_dateset' : subject.subject_id_dataset}
        if os.path.exists(os.path.join(save_folder, str_chs, str(total_id))):
            with open(os.path.join(save_folder, str_chs, str(total_id), info_filename), 'wb') as f:
                print(os.path.join(save_folder, str_chs, str(total_id), info_filename))
                pickle.dump(info, f)

    except (AssertionError, ValueError) as e:
        logging.error("An error occurred during multiprocessing: %s. Subject ID: %s, Total ID: %s, subject.eeg_info: %s", e,
                      subject.subject_total_id, total_id,subject.eeg_info)
    except Exception as e:
        logging.error("An error occurred during multiprocessing: %s. Subject ID: %s, Total ID: %s, subject.eeg_info: %s", e,
                      subject.subject_total_id, total_id,subject.eeg_info)


if __name__ == '__main__':
    # Pretrain Data ===================================================

    """
    先直接finetune 而且不用fnirs的数据集
    """

    # # Data processing for the directories to be preprocessed
    data_folders = [
            'Tdbrain_challenge_mdd_hc_predict/66_Td_mdd_predict_eyeopen',
            'Tdbrain_challenge_mdd_hc_predict/65_Td_mdd_predict_eyeclose'
            # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/LEMON',
            # '/home/wangkuiyu/data1/LEM/new_data_pool/EEG/HBN_1',
    ]
    save_folders = [
            'pretrain_set/EEG/eyeopen',
            'pretrain_set/EEG/eyeopen',
            # './pretrain_set/EEG/LEMON',
            # './pretrain_set/EEG/HBN_1',
    ]

    # for data_folder, save_folder in zip(data_folders, save_folders):
    #     file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith('sub_')]
    #     # Use a partial function to include save_folder during processing
    #     process_subject_partial = partial(process_subject, save_folder=save_folder)
    #     with Pool(processes=64) as pool:
    #         pool.map(process_subject_partial, file_paths)

    # Downstream for test.py dataset processing (directories marked as "leave for downstream")
    downstream_data_folders = [
            'Tdbrain_challenge_mdd_hc_predict/66_Td_mdd_predict_eyeopen/',
            'Tdbrain_challenge_mdd_hc_predict/65_Td_mdd_predict_eyeclose/'
            # '/data1/wangkuiyu/LEM/clinical/HUILONGGUAN_1028/',
    ]

    downstream_save_folders = [
            'pretrain_set/EEG/eyeopen',
            'pretrain_set/EEG/eyeopen',
            # '/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/clinical/HUILONGGUAN_1028/',
    ]

    for data_folder, save_folder in zip(downstream_data_folders, downstream_save_folders):
        file_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith('sub_')]
        process_subject_partial = partial(process_subject, save_folder=save_folder)
        with Pool(processes=32) as pool:
            pool.map(process_subject_partial, file_paths)
