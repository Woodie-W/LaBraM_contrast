import argparse
from run_class_finetune_for_embedding import prepare_data
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from functools import partial
import pickle
from timm.models import create_model
import utils

def get_multi_label(label):
    items = []
    if label is None:
        items = [None]
    elif isinstance(label, list):
        for item in label:
            items.append(item)
    else:
        assert isinstance(label, str)
        items = [label]
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
            self.label2int = {
                # "HC": 0,
                # "HEALTHY": 0,
                # "LowOCD": 0,
                # "AD": 1,
                # "FTD": 2,
                # "PD": 3,
                # "PARKINSON": 3,
                # "past-MDD": 4,
                # "MDD": 5,
                # "Dp": 6,
                # "ADHD": 7,
                # "ADHD ": 7,
                # "OCD": 8,
                # "HighOCD": 8,
                # "SMC": 9,
                # "CHRONIC PAIN": 10,
                # "MSA-C": 11,
                # "DYSLEXIA": 12,
                # "TINNITUS": 13,
                # "INSOMNIA": 14,
                # "BURNOUT": 15,
                # "DEPERSONALIZATION": 16,
                # "ANXIETY": 17,
                # "BIPOLAR": 18,
                # "PDD NOS ": 19,
                # "PDD NOS": 19,
                # "ASD": 20,
                # "ASPERGER": 21,
                # "MIGRAINE": 22,
                # "PANIC": 23,
                # "TUMOR": 24,
                # "WHIPLASH": 25,
                # "PAIN": 26,
                # "CONVERSION DX": 27,
                # "STROKE ": 28,
                # "STROKE": 28,
                # "LYME": 29,
                # "PTSD": 30,
                # "EPILEPSY": 31,
                # "abnormal": 31,
                # "TRAUMA": 32,
                # "TBI": 33,
                # "DPS": 34,
                # "ANOREXIA": 35,
                # "DYSPRAXIA": 36,
                # "DYSCALCULIA": 37,
                # "GTS": 38,
                # "mTBI": 39,
                # "SZ": 40,
                # "A&A": 41,
                # "Delirium": 42,
                # "PD-FOG-": 43,
                # "PD-FOG+": 44,
                # "Chronic TBI": 45,
                # "Recrudesce": 46,
                # "Somatic": 47,  # todo
                "HC": 0,
                "HEALTHY": 0,
                "LowOCD": 0,
                "MDD": 1,
                "ADHD": 2,
                "ADHD ": 2,
                "DYSLEXIA": 2,
                "DYSPRAXIA": 2,
                "GTS": 2,
                "DYSCALCULIA": 2,
                "OCD": 3,
                "HighOCD": 3,
                "ANXIETY": 4,
                "PTSD": 4,
                "PANIC": 4,
                "SZ": 5,
                "DEPERSONALIZATION": 5,
                "PD": 6,
                "PARKINSON": 6,
                "PD-FOG-": 6,
                "PD-FOG+": 6,
                "A&A": 7,
                "past-MDD": 8,
                "AD": 9,
                "FTD": 10,
                "Dp": 11,
                "SMC": 12,
                "MSA-C": 13,
                "TINNITUS": 14,
                "CHRONIC PAIN": 14,
                "PAIN": 14,
                "INSOMNIA": 15,
                "insomnia": 15,
                "BURNOUT": 16,
                "BIPOLAR": 17,
                "BP": 17,
                "PDD NOS ": 18,
                "PDD NOS": 18,
                "ASD": 18,
                "ASPERGER": 18,
                "TUMOR": 19,
                "WHIPLASH": 20,
                "TRAUMA": 20,
                "TBI": 20,
                "mTBI": 20,
                "Chronic TBI": 20,
                "CONVERSION DX": 21,
                "STROKE ": 22,
                "STROKE": 22,
                "MIGRAINE": 22,
                "LYME": 23,
                "EPILEPSY": 24,
                "abnormal": 24,
                "DPS": 25,
                "ANOREXIA": 26,
                "Delirium": 27,
                "Recrudesce": 28,
                "Somatic": 29,
                "Severe-MDD": 30,
                "Mid-MDD": 30,
                "Light-MDD": 32,
                "SuperLight-MDD": 33,  # todo
                "Severe-HC": 34,
                "Mid-HC": 35,
                "Light-HC": 36,
                "SuperLight-HC": 36,
                "Severe-Dp": 37,
                "Mid-Dp": 38,
                "Light-Dp": 39,
                "SuperLight-Dp": 40,
                "Severe-Past-MDD": 41,
                "Mid-Past-MDD": 42,
                "Light-Past-MDD": 43,
                "SuperLight-Past-MDD": 44,
            }
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
                class_label = info["subject_label"]

        # 给定的是list or str,Multi-label or single label,class_label最终都是int的list列表
        if isinstance(class_label, list):
            class_label = [self.label2int[cl] for cl in class_label]
        else:
            assert isinstance(class_label, str)
            class_label = [self.label2int[class_label]]

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
        return de_features, class_label,total_id

    def get_ch_names(self):
        dataset_path = os.path.dirname(os.path.dirname(self.file_paths[0]))
        channel_name = pickle.load(
            open(os.path.join(dataset_path, "channel_name.pkl"), "rb")
        )
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
            with open(self.id2info_path[total_id], "rb") as f:
                sub2label[total_id] = pickle.load(f)["subject_label"]
        # label2counts: {class_label(single): counts}
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

    id = []
    # 填充每个样本
    for i, sample in enumerate(batch):
        feature, class_label,total_id = sample
        features.append(feature)
        id.append(total_id)
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
    id = torch.tensor(id, dtype=torch.long)
    return features, all_labels,id


def prepare_data(seed, split=True):
    seed = 12345
    np.random.seed(seed)
    # 读取数据路径
    hc = {
        "HC": [
            os.path.join(
                "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/SXMU_2_PROCESSED/HC/59chs",
                file,
            )
            for file in os.listdir(
                "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/SXMU_2_PROCESSED/HC/59chs"
            )
        ]
    }

    mdd = {
        "MDD": [
            os.path.join(
                "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/SXMU_2_PROCESSED/MDD/59chs",
                file,
            )
            for file in os.listdir(
                "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/SXMU_2_PROCESSED/MDD/59chs"
            )
        ]
    }
    data_path_with_cls = {k: v for d in [hc, mdd] for k, v in d.items()}

    sft_paths = [
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/Td_eyeopen/26chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/Td_eyeclose/26chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/Parkinson_eyes_open_PROCESSED/61chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/Parkinson_eyes_open_PROCESSED/63chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/AD_FD_HC_PROCESSED/19chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/PREDICT-Depression_Rest/60chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/PREDICT-PD_LPC_Rest/61chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/PREDICT-PD_LPC_Rest/63chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/PREDICT-PD_LPC_Rest_2/63chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/CSA-PD-W/30chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/Parkinson/32chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/Porn-addiction/19chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/Porn-addiction/19chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/EEG_in_SZ/19chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/First_Episode_Psychosis_Control_1/60chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/First_Episode_Psychosis_Control_2/60chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/QDHSM/20chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/SXMU-ERP/20chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/QDHSM/20chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/SXMU-ERP/20chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/LEMON/61chs",
    ]

    # data_path_wo_cls字典，里面存放了某个路径数据集下所有被试路径
    data_path_wo_cls = {}
    for sftp in sft_paths:
        data_path_wo_cls[sftp] = [os.path.join(sftp, file) for file in os.listdir(sftp)]

    if not split:
        return data_path_wo_cls, data_path_with_cls

    # 分十折
    kfold_indices_wo_cls = {}
    for key, paths in data_path_wo_cls.items():
        np.random.shuffle(paths)
        folds = np.array_split(paths, 10)
        kfold_indices_wo_cls[key] = folds

    kfold_indices_with_cls = {}
    for key, paths in data_path_with_cls.items():
        np.random.shuffle(paths)
        folds = np.array_split(paths, 10)
        kfold_indices_with_cls[key] = folds

    return kfold_indices_wo_cls, kfold_indices_with_cls


def get_dataset(args):
    seed = 12345
    np.random.seed(seed)
    kfold_indices_wo_cls, kfold_indices_with_cls = prepare_data(seed)

    train_indices_wo_cls = []
    eval_indices_wo_cls = []
    train_indices_with_cls = dict()
    eval_indices_with_cls = dict()
    args.fold = 0

    for dataset_key, folds in kfold_indices_wo_cls.items():
        train_fold = []
        eval_fold = []
        for i in range(10):
            if i == args.fold:
                eval_fold = folds[i].tolist()
            else:
                train_fold.extend(folds[i].tolist())
        train_indices_wo_cls.append(train_fold)
        eval_indices_wo_cls.append(eval_fold)

    for key, folds in kfold_indices_with_cls.items():
        train_fold = []
        eval_fold = []
        for i in range(10):
            if i == args.fold:
                eval_fold = folds[i].tolist()
            else:
                train_fold.extend(folds[i].tolist())
        train_indices_with_cls[key] = train_fold
        eval_indices_with_cls[key] = eval_fold

    used_ints = [i for i in range(45)]
    dataset_train_list = []
    train_ch_names_list = []
    train_dataset_names = []
    dataset = SFTSet_embedding(
        data_path=train_indices_with_cls,
        data_path_without_cls=None,
        clip=False,
        kept_ints=used_ints,
    )
    dataset_train_list.append(dataset)
    train_ch_names_list.append(dataset.get_ch_names())
    train_dataset_names.append(dataset.get_dataset_name())

    for dataset_subject_paths in train_indices_wo_cls:
        dataset = SFTSet_embedding(
            data_path=None,
            data_path_without_cls=dataset_subject_paths,
            clip=False,
            kept_ints=used_ints,
        )
        dataset_train_list.append(dataset)
        train_ch_names_list.append(dataset.get_ch_names())
        train_dataset_names.append(dataset.get_dataset_name())

    dataset_eval_list = []
    eval_ch_names_list = []
    eval_dataset_names = []

    dataset = SFTSet_embedding(
        data_path=eval_indices_with_cls,
        data_path_without_cls=None,
        clip=False,
        kept_ints=used_ints,
    )
    dataset_eval_list.append(dataset)
    eval_ch_names_list.append(dataset.get_ch_names())
    eval_dataset_names.append(dataset.get_dataset_name())

    for dataset_subject_paths in eval_indices_wo_cls:
        dataset = SFTSet_embedding(
            data_path=None,
            data_path_without_cls=dataset_subject_paths,
            clip=False,
            kept_ints=used_ints,
        )
        dataset_eval_list.append(dataset)
        eval_ch_names_list.append(dataset.get_ch_names())
        eval_dataset_names.append(dataset.get_dataset_name())

    if args.phase == "train":
        return dataset_train_list, train_ch_names_list
    if args.phase == "eval":
        return dataset_eval_list, eval_ch_names_list


def get_models(args):
    model = create_model(
        "labram_base_patch200_200",
        pretrained=False,
        num_classes=45,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=True,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=True,
    )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        type=str,
    )
    parser.add_argument("--phase", type=str)
    parser.add_argument("--save_name", type=str)
    args = parser.parse_args()

    dataset_list, ch_names_list = get_dataset(args)

    model = get_models(args)
    checkpoint = torch.load(
        "/data1/wangkuiyu/model_update_code/LEM_forlabram/LaBraM/checkpoints/finetune_base_embedding_1112/fold0/checkpoint-29.pth",
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"]).cuda()

    save_path = f"./embs_{args.save_name}"
    os.mkdir(save_path)
    n_sample_per_sub = 20

    data_loader_list = []
    for dataset in dataset_list:
        data_loader_train = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            collate_fn=partial(custom_collate_fn, multi_label=True),
        )
        data_loader_list.append(data_loader_train)
        
    for i,(data_loader, ch_names) in enumerate(zip(data_loader_list, ch_names_list)):
        print("data_loader: ", data_loader.dataset.get_dataset_name())
        input_chans = utils.get_input_chans(ch_names)
        for step,batch in enumerate(data_loader):
            features, labels, id = batch
            features = features.cuda()
            with torch.no_grad():
                x = model.forward_features(x,input_chans=input_chans,return_patch_tokens=True,return_all_tokens=False)
                x = x.sum(1)
            
            for b in range(features.shape[0]):
                sub_path =  os.path.join(save_path, str(id[b].item()))
                if not os.path.exists(sub_path):
                    os.mkdir(sub_path)
                    with open(os.path.join(sub_path, "label.pkl"), "wb") as f:
                        pickle.dump(labels[b][labels[b]>=0].detach().cpu().numpy().tolist(), f)
                
                if len(os.listdir(sub_path)) >= n_sample_per_sub:
                    continue
                
                np.save(os.path.join(sub_path, f"batch{step}_{b}.npy"), x[b].detach().cpu().numpy())