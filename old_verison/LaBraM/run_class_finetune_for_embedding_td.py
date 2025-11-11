# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import datetime
from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import (
    create_optimizer,
    get_parameter_groups,
    LayerDecayValueAssigner,
)

from engine_for_finetuning_td import train_one_epoch_for_embedding, evaluate_for_embedding
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from scipy import interpolate
import modeling_finetune
from data_processor.dataset import SFTSet_embedding,custom_collate_fn
import pickle

def get_args():
    parser = argparse.ArgumentParser(
        "LaBraM fine-tuning and evaluation script for EEG classification",
        add_help=False,
    )
    parser.add_argument("--fold", default=7, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=5, type=int)

    # robust evaluation
    parser.add_argument(
        "--robust_test", default=None, type=str, help="robust evaluation dataset"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="labram_base_patch200_200",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--qkv_bias", action="store_true")
    parser.add_argument("--disable_qkv_bias", action="store_false", dest="qkv_bias")
    parser.set_defaults(qkv_bias=True)
    parser.add_argument("--rel_pos_bias", action="store_true")
    parser.add_argument(
        "--disable_rel_pos_bias", action="store_false", dest="rel_pos_bias"
    )
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument("--abs_pos_emb", action="store_true")
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument(
        "--layer_scale_init_value",
        default=0.1,
        type=float,
        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale",
    )

    parser.add_argument("--input_size", default=200, type=int, help="EEG input size")

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--attn_drop_rate",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Attention dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--disable_eval_during_finetuning", action="store_true", default=False
    )

    parser.add_argument("--model_ema", action="store_true", default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999, help="")
    parser.add_argument(
        "--model_ema_force_cpu", action="store_true", default=False, help=""
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument("--layer_decay", type=float, default=0.9)

    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )

    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)
    parser.add_argument("--model_filter_name", default="gzp", type=str)
    parser.add_argument("--init_scale", default=0.001, type=float)
    parser.add_argument("--use_mean_pooling", action="store_true")
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")
    parser.add_argument(
        "--disable_weight_decay_on_rel_pos_bias", action="store_true", default=False
    )

    # Dataset parameters
    parser.add_argument(
        "--nb_classes", default=0, type=int, help="number of the classification types"
    )

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--enable_deepspeed", action="store_true", default=False)
    parser.add_argument(
        "--dataset", default="TUAB", type=str, help="dataset: TUAB | TUEV"
    )
    parser.add_argument(
        "--gpu_id", default="8", type=str, help=""
    )
    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig

            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_models(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value,
        qkv_bias=args.qkv_bias,
    )

    return model


def prepare_data(seed, split=True,json_folder=None):
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

    # todo

    # data_path_with_cls = {k: v for d in [hc, mdd] for k, v in d.items()}
    data_path_with_cls = {}

    sft_paths = [
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/Td_eyeopen/26chs",
        "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/Td_eyeclose/26chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/Parkinson_eyes_open_PROCESSED/61chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/Parkinson_eyes_open_PROCESSED/63chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/AD_FD_HC_PROCESSED/19chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/PREDICT-PD_LPC_Rest/61chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/PREDICT-PD_LPC_Rest/63chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/PREDICT-PD_LPC_Rest_2/63chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/CSA-PD-W/30chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/Parkinson/32chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/Porn-addiction/19chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/Porn-addiction/19chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/EEG_in_SZ/19chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/First_Episode_Psychosis_Control_1/60chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/First_Episode_Psychosis_Control_2/60chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/QDHSM/20chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/SXMU-ERP/20chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/QDHSM/20chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/SXMU-ERP/20chs",
        # "/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/LEMON/61chs",
        # '/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/HUAWEI_eye_open/30chs',
        # '/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/HUAWEI_eye_open/59chs',
        # '/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/HUAWEI_eye_close/30chs',
        # '/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/HUAWEI_eye_close/59chs',
        # '/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_unknown/57_Predict-Depression-Rest-New/60chs',
        # '/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_close/60_ANDing/20chs',
        # '/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/resting_eye_open/60_ANDing/20chs',
        # '/data1/wangkuiyu/model_update_code/labram2_resting_fine_pool/clinical/HUILONGGUAN_1028/30chs',
    ]

    # 创建按数据集和被试分组的嵌套字典
    dataset_subject_dict = {}

    # 遍历每个数据集路径，获取被试字典
    for sftp in sft_paths:
        dataset_subject_dict[sftp] = {}
        for file in os.listdir(sftp):
            file_path = os.path.join(sftp, file)
            info_file = os.path.join(file_path, f"{file}_info.pkl")

            try:
                with open(info_file, 'rb') as f:
                    info_dict = pickle.load(f)
                # subject_id = info_dict['subject_id_dateset']
                subject_id = str(file)
                if subject_id not in dataset_subject_dict[sftp]:
                    dataset_subject_dict[sftp][subject_id] = []
                dataset_subject_dict[sftp][subject_id].append(file_path)
            except:
                print(f"获取字典警告: 无法处理文件 {file} 的信息")
                continue

    # data_path_wo_cls字典
    data_path_wo_cls = {}
    for sftp in sft_paths:
        data_path_wo_cls[sftp] = [os.path.join(sftp, file) for file in os.listdir(sftp)]

    if not split:
        return data_path_wo_cls, data_path_with_cls

    # 读取json文件中的折划分信息
    fold_subjects = get_fold_subjects(json_folder)
    # 打印每一折中的验证集被试数量
    print("\nJson文件中每折的验证集被试数量：")
    for fold_idx, subjects in fold_subjects.items():
        print(f"Fold {fold_idx}: {len(subjects)} subjects")

    # 获取所有json文件中的被试
    all_json_subjects = set()
    for subjects in fold_subjects.values():
        all_json_subjects.update(subjects)
    print(f"\nJson文件中总共的唯一被试数量: {len(all_json_subjects)}")

    # 根据json文件的划分信息进行十折划分
    kfold_indices_wo_cls = {}
    for dataset_path, subject_dict in dataset_subject_dict.items():
        dataset_folds = []
        # 打印数据集中的被试数量
        # print(f"\n数据集 {dataset_path} 中的被试数量: {len(subject_dict)}")

        for fold_idx in range(10):
            fold_files = []
            for subject_id, files in subject_dict.items():
                # 如果被试在当前折的验证集中，将其文件添加到该折
                if subject_id in fold_subjects[fold_idx]:
                    fold_files.extend(files)
            dataset_folds.append(np.array(fold_files))
        kfold_indices_wo_cls[dataset_path] = dataset_folds

    # 对有标签数据也按相同方式进行十折划分
    kfold_indices_with_cls = {}
    for key, paths in data_path_with_cls.items():
        subject_dict = {}
        for path in paths:
            file = os.path.basename(path)
            info_file = os.path.join(path, f"{file}_info.pkl")
            try:
                with open(info_file, 'rb') as f:
                    info_dict = pickle.load(f)
                # subject_id = info_dict['subject_id_dateset']
                subject_id = str(file)
                if subject_id not in subject_dict:
                    subject_dict[subject_id] = []
                subject_dict[subject_id].append(path)
            except:
                print(f"按被试划分警告: 无法处理文件 {file} 的信息")
                continue

        class_folds = []
        for fold_idx in range(10):
            fold_files = []
            for subject_id, files in subject_dict.items():
                if subject_id in fold_subjects[fold_idx]:
                    fold_files.extend(files)
            class_folds.append(np.array(fold_files))

        kfold_indices_with_cls[key] = class_folds


    # 1111111
    # # 检查现有的数据集中的所有被试数和给定的十折划分的被试是否相同
    distributed_subjects = set()
    # # 从dataset_subject_dict中收集
    for dataset_path, subject_dict in dataset_subject_dict.items():
        distributed_subjects.update(subject_dict.keys())

    # # 从data_path_with_cls中收集
    for key, paths in data_path_with_cls.items():
        for path in paths:
            file = os.path.basename(path)
            subject_id = str(file)
            distributed_subjects.add(subject_id)

    # 比较与fold_subjects中的被试
    all_fold_subjects = set()
    for subjects in fold_subjects.values():
        all_fold_subjects.update(subjects)
    #
    # # 检查差异
    missing_subjects = all_fold_subjects - distributed_subjects
    extra_subjects = distributed_subjects - all_fold_subjects
    # extra_subjects = sorted(extra_subjects, key=int)
    print("\n被试分布检查结果:")
    print(f"Fold_subjects中的总被试数: {len(all_fold_subjects)}")
    print(f"实际待分配的总被试数: {len(distributed_subjects)}")
    if missing_subjects:
        print(f"在fold_subjects中存在但未被分配的被试: {missing_subjects}")
    if extra_subjects:
        print(f"数据集中有但不在fold_subjects中的被试: {extra_subjects}")
    # 1111111



    # 比较最终获取的被试数和给定的十折划分中的被试是否相同。
    # 按折收集实际分配的被试
    distributed_subjects_by_fold_wo_cls = {i: set() for i in range(10)}
    distributed_subjects_by_fold_with_cls = {i: set() for i in range(10)}
    # 从kfold_indices_wo_cls中按折收集
    for dataset_path, folds in kfold_indices_wo_cls.items():
        for fold_idx, fold_files in enumerate(folds):
            for file_path in fold_files:
                subject_id = str(os.path.basename(file_path))
                distributed_subjects_by_fold_wo_cls[fold_idx].add(subject_id)

    # 从kfold_indices_with_cls中按折收集
    for class_key, folds in kfold_indices_with_cls.items():
        for fold_idx, fold_files in enumerate(folds):
            for file_path in fold_files:
                subject_id = str(os.path.basename(file_path))
                distributed_subjects_by_fold_with_cls[fold_idx].add(subject_id)

    # 获取所有实际分配的被试
    distributed_subjects = set()
    for fold_subjects_set in distributed_subjects_by_fold_wo_cls.values():
        distributed_subjects.update(fold_subjects_set)
    for fold_subjects_set in distributed_subjects_by_fold_with_cls.values():
        distributed_subjects.update(fold_subjects_set)


    print("\n按折比较被试分布:")
    for fold_idx in range(10):
        expected_subjects = set(fold_subjects[fold_idx])
        actual_subjects = distributed_subjects_by_fold_wo_cls[fold_idx] | distributed_subjects_by_fold_with_cls[fold_idx] #取并集

        print(f"\nFold {fold_idx}:")
        print(f"  预期被试数: {len(expected_subjects)}")
        print(f"  实际分配被试数_wo_cls: {len(distributed_subjects_by_fold_wo_cls[fold_idx])}")
        print(f"  实际分配被试数_with_cls: {len(distributed_subjects_by_fold_with_cls[fold_idx])}")
        print(f"  实际分配总被试数: {len(actual_subjects)}")

        missing_in_fold = expected_subjects - actual_subjects
        extra_in_fold = actual_subjects - expected_subjects

        if missing_in_fold:
            print(f"  该折中缺失的被试: {missing_in_fold}")
        if extra_in_fold:
            print(f"  该折中多出的被试: {extra_in_fold}")

    print("\n总体统计:")
    print(f"Fold_subjects中的总被试数: {len(all_fold_subjects)}")
    print(f"实际分配的总被试数: {len(distributed_subjects)}")

    # 检查总体差异
    missing_subjects = all_fold_subjects - distributed_subjects
    extra_subjects = distributed_subjects - all_fold_subjects

    if missing_subjects:
        print(f"在fold_subjects中存在但未被分配的被试: {missing_subjects}")
    if extra_subjects:
        print(f"被分配但不在fold_subjects中的被试: {extra_subjects}")


    return kfold_indices_wo_cls, kfold_indices_with_cls

def get_fold_subjects(json_folder):
    """读取json文件获取每一折的验证集被试"""
    fold_subjects = {}
    for i in range(10):
        #
        # todo 0
        #
        # json_file = os.path.join(json_folder, f'subs.json_cls10_rerun_ckpt001_{i}.json')  #all
        json_file = os.path.join(json_folder, f'subs_Tdsplit_{i}.json')  #tdbrain
        with open(json_file, 'r') as f:
            data = json.load(f)
            fold_subjects[i] = set(data['eval'])  # 使用集合便于查找
    return fold_subjects


def get_dataset(args):
    seed = 12345
    np.random.seed(seed)
    #
    # todo 1
    #
    # kfold_indices_wo_cls, kfold_indices_with_cls = prepare_data(seed, json_folder='10fold_split_dict/all')
    kfold_indices_wo_cls, kfold_indices_with_cls = prepare_data(seed,json_folder = '10fold_split_dict/tdbrain')

    print(f"Fold {args.fold + 1}/10")

    train_indices_wo_cls = []
    # train_indices_wo_cls_linshi=[]
    eval_indices_wo_cls = []
    train_indices_with_cls = dict()
    eval_indices_with_cls = dict()

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

    # used_ints = [i for i in range(45)]

    # todo 2

    # used_ints = [0,1,2,3,4,5,6,7,8,9]
    used_ints = [0,1]
    dataset_train_list = []
    train_ch_names_list = []
    train_dataset_names = []

    # todo 3

    # hc mdd

    # dataset = SFTSet_embedding(
    #     data_path=train_indices_with_cls, data_path_without_cls=None, clip=False, kept_ints=used_ints
    # )
    # dataset_train_list.append(dataset)
    # train_ch_names_list.append(dataset.get_ch_names())
    # train_dataset_names.append(dataset.get_dataset_name())


    #
    for dataset_subject_paths in train_indices_wo_cls:
        dataset = SFTSet_embedding(
            data_path=None, data_path_without_cls=dataset_subject_paths, clip=False, kept_ints=used_ints
        )
        dataset_train_list.append(dataset)
        train_ch_names_list.append(dataset.get_ch_names())
        train_dataset_names.append(dataset.get_dataset_name())

    dataset_eval_list = []
    eval_ch_names_list = []
    eval_dataset_names = []

    # todo 4

    # # # hc mdd

    # dataset = SFTSet_embedding(
    #     data_path=eval_indices_with_cls, data_path_without_cls=None, clip=False, kept_ints=used_ints
    # )
    # dataset_eval_list.append(dataset)
    # eval_ch_names_list.append(dataset.get_ch_names())
    # eval_dataset_names.append(dataset.get_dataset_name())


    for dataset_subject_paths in eval_indices_wo_cls:
        if dataset_subject_paths == []:
            continue
        dataset = SFTSet_embedding(
            data_path=None, data_path_without_cls=dataset_subject_paths, clip=False, kept_ints=used_ints
        )
        dataset_eval_list.append(dataset)
        eval_ch_names_list.append(dataset.get_ch_names())
        eval_dataset_names.append(dataset.get_dataset_name())

    metrics = ["accuracy", "balanced_accuracy"]

    # todo 5

    # args.nb_classes = 45
    args.nb_classes = 2
    # args.nb_classes = 1 #二分类(不用这行)
    args.dataset_name = train_dataset_names

    # print(train_ch_names_list, eval_ch_names_list)

    total_train = 0
    total_eval = 0
    for dataset in dataset_train_list:
        print("len_train", len(dataset))
        total_train += len(dataset)
    for dataset in dataset_eval_list:
        print("len_val", len(dataset))
        total_eval += len(dataset)
    print('total_train', total_train)
    print('total_eval', total_eval)

    return (
        dataset_train_list,
        train_ch_names_list,
        dataset_eval_list,
        eval_ch_names_list,
        metrics,
    )


def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.

    # dataset_train, dataset_test, dataset_val, ch_names, metrics = get_dataset(args)
    (
        dataset_train_list,
        train_ch_names_list,
        dataset_val_list,
        val_ch_names_list,
        metrics,
    ) = get_dataset(args)

    print(dataset_train_list)
    print(train_ch_names_list)
    print(dataset_val_list)
    print(val_ch_names_list)
    print(metrics)
    if args.disable_eval_during_finetuning:
        dataset_val = None
        # dataset_test = None

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)
            # print("Sampler_train = %s" % str(sampler_train))

        # sampler_train = torch.utils.data.DistributedSampler(
        #     dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        # )
        # print("Sampler_train = %s" % str(sampler_train))
        sampler_eval_list = []
        if args.dist_eval:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
                )
                sampler_eval_list.append(sampler_val)
        else:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.SequentialSampler(dataset)
                sampler_eval_list.append(sampler_val)
        # if args.dist_eval:
        #     if len(dataset_val) % num_tasks != 0:
        #         print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
        #               'This will slightly alter validation results as extra duplicate entries are added to achieve '
        #               'equal num of samples per-process.')
        #     sampler_val = torch.utils.data.DistributedSampler(
        #         dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        #     if type(dataset_test) == list:
        #         sampler_test = [torch.utils.data.DistributedSampler(
        #             dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False) for dataset in dataset_test]
        #     else:
        #         sampler_test = torch.utils.data.DistributedSampler(
        #             dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        # else:
        #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        #     sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            # num_workers=args.num_workers,
            num_workers=2,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn = partial(custom_collate_fn, multi_label=True),
        )
        data_loader_train_list.append(data_loader_train)

    for data_loader in data_loader_train_list:
        print("iter: ",len(data_loader))
    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    if dataset_val_list is not None:
        data_loader_val_list = []
        for dataset, sampler in zip(dataset_val_list, sampler_eval_list):
            data_loader_val = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                #   batch_size=int(1.5 * args.batch_size),
                batch_size=args.batch_size,
                # num_workers=args.num_workers,
                num_workers=2,
                pin_memory=args.pin_mem,
                drop_last=False,
                collate_fn = partial(custom_collate_fn, multi_label=True),
            )
            data_loader_val_list.append(data_loader_val)
    else:
        data_loader_val_list = None

    # if dataset_val is not None:
    #     data_loader_val = torch.utils.data.DataLoader(
    #         dataset_val, sampler=sampler_val,
    #         batch_size=int(1.5 * args.batch_size),
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False
    #     )
    #     if type(dataset_test) == list:
    #         data_loader_test = [torch.utils.data.DataLoader(
    #             dataset, sampler=sampler,
    #             batch_size=int(1.5 * args.batch_size),
    #             num_workers=args.num_workers,
    #             pin_memory=args.pin_mem,
    #             drop_last=False
    #         ) for dataset, sampler in zip(dataset_test, sampler_test)]
    #     else:
    #         data_loader_test = torch.utils.data.DataLoader(
    #             dataset_test, sampler=sampler_test,
    #             batch_size=int(1.5 * args.batch_size),
    #             num_workers=args.num_workers,
    #             pin_memory=args.pin_mem,
    #             drop_last=False
    #         )
    # else:
    #     data_loader_val = None
    #     data_loader_test = None

    model = get_models(args)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and (args.model_filter_name != ""):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith("student."):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    # num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    num_training_steps_per_epoch = (
        sum([len(dataset) // total_batch_size for dataset in dataset_train_list])
    )
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    # print("Number of training examples = %d" % len(dataset_train))
    print(
        "Number of training examples = %d"
        % sum([len(dataset) for dataset in dataset_train_list])
    )
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(
                args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
            )
        )
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add(
                "blocks.%d.attn.relative_position_bias_table" % i
            )

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model,
            args.weight_decay,
            skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
        )
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print(
            "model.gradient_accumulation_steps() = %d"
            % model.gradient_accumulation_steps()
        )
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True
            )
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args,
            model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
    )

    # if args.eval:
    #     balanced_accuracy = []
    #     accuracy = []
    #     for data_loader in data_loader_test:
    #         test_stats = evaluate(
    #             data_loader,
    #             model,
    #             device,
    #             header="Test:",
    #             ch_names=ch_names,
    #             metrics=metrics,
    #             is_binary=(args.nb_classes == 1),
    #         )
    #         accuracy.append(test_stats["accuracy"])
    #         balanced_accuracy.append(test_stats["balanced_accuracy"])
    #     print(
    #         f"======Accuracy: {np.mean(accuracy)} {np.std(accuracy)}, balanced accuracy: {np.mean(balanced_accuracy)} {np.std(balanced_accuracy)}"
    #     )
    #     exit(0)

    # for train_loader in data_loader_train_list:
    #     print("len_train_loader",len(train_loader))
    # if data_loader_val_list is not None:
    #     for val_loader in data_loader_val_list:
    #         print("len_val_loader",len(val_loader))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # data_loader_train.sampler.set_epoch(epoch)
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch_for_embedding(
            model,
            criterion,
            data_loader_train_list,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            ch_names_list=train_ch_names_list,
            is_binary=args.nb_classes == 1,
            args=args,
        )

        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                model_ema=model_ema,
                save_ckpt_freq=args.save_ckpt_freq,
            )

        if data_loader_val_list is not None:
            val_stats = evaluate_for_embedding(
                data_loader_val_list,
                model,
                device,
                header="Val:",
                ch_names_list=val_ch_names_list,
                metrics=metrics,
                is_binary=args.nb_classes == 1,
            )
            print(
                f"Accuracy of the network on the {sum([len(dataset) for dataset in dataset_val_list])} val EEG: {val_stats['accuracy']:.2f}%"
            )
            if max_accuracy < val_stats["accuracy"]:
                max_accuracy = val_stats["accuracy"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=model_ema,
                    )
            print(f"Max accuracy val: {max_accuracy:.2f}%")

            # if data_loader_val is not None:
            #     val_stats = evaluate(data_loader_val, model, device, header='Val:', ch_names=ch_names, metrics=metrics,
            #                          is_binary=args.nb_classes == 1)
            #     print(f"Accuracy of the network on the {len(dataset_val)} val EEG: {val_stats['accuracy']:.2f}%")
            #     test_stats = evaluate(data_loader_test, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
            #                           is_binary=args.nb_classes == 1)
            #     print(f"Accuracy of the network on the {len(dataset_test)} test EEG: {test_stats['accuracy']:.2f}%")

            #     if max_accuracy < val_stats["accuracy"]:
            #         max_accuracy = val_stats["accuracy"]
            #         if args.output_dir and args.save_ckpt:
            #             utils.save_model(
            #                 args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #                 loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            #         max_accuracy_test = test_stats["accuracy"]

            #     print(f'Max accuracy val: {max_accuracy:.2f}%, max accuracy test: {max_accuracy_test:.2f}%')
            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == "accuracy":
                        log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == "balanced_accuracy":
                        log_writer.update(
                            balanced_accuracy=value, head="val", step=epoch
                        )
                    elif key == "balanced_accuracy_old":
                        log_writer.update(
                            balanced_accuracy_old=value, head="val", step=epoch
                        )
                    elif key == "accuracy_old":
                        log_writer.update(
                            accuracy_old=value, head="val", step=epoch
                        )
                    elif key == "f1_weighted":
                        log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == "pr_auc":
                        log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == "roc_auc":
                        log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == "cohen_kappa":
                        log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == "loss":
                        log_writer.update(loss=value, head="val", step=epoch)
                # for key, value in test_stats.items():
                #     if key == 'accuracy':
                #         log_writer.update(accuracy=value, head="test", step=epoch)
                #     elif key == 'balanced_accuracy':
                #         log_writer.update(balanced_accuracy=value, head="test", step=epoch)
                #     elif key == 'f1_weighted':
                #         log_writer.update(f1_weighted=value, head="test", step=epoch)
                #     elif key == 'pr_auc':
                #         log_writer.update(pr_auc=value, head="test", step=epoch)
                #     elif key == 'roc_auc':
                #         log_writer.update(roc_auc=value, head="test", step=epoch)
                #     elif key == 'cohen_kappa':
                #         log_writer.update(cohen_kappa=value, head="test", step=epoch)
                #     elif key == 'loss':
                #         log_writer.update(loss=value, head="test", step=epoch)

            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            #              **{f'val_{k}': v for k, v in val_stats.items()},
            #              **{f'test_{k}': v for k, v in test_stats.items()},
            #              'epoch': epoch,
            #              'n_parameters': n_parameters}
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)  # 使用指定的GPU

    main(opts, ds_init)

#--finetune finetune_model/checkpoint-4.pth \
# used_ints
# args.nb_classes
# hc ，mdd
# json_folder = '10fold_split_dict/all'