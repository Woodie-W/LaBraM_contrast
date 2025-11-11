import argparse, json, os
import numpy as np
import datetime, time
import torch, pickle  # 让数据持久化保存
import torch.backends.cudnn as cudnn  # 优化cuDNN性能

from functools import partial  # 部分应用函数
from pathlib import Path  # 用于处理文件路径
from collections import OrderedDict  # 有序字典，用于加载模型权重
from timm.models import create_model  # timm库中用于创建模型的函数
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma  # timm库中的模型指数移动平均（EMA）工具
from optim_factory import (create_optimizer, get_parameter_groups, LayerDecayValueAssigner)
import utils, modeling_finetune
from utils import NativeScalerWithGradNormCount as NativeScaler
from engine_for_finetuning import train_one_epoch_for_embedding, evaluate_for_embedding
from data_processor.dataset import SFTSet_embedding, custom_collate_fn


def get_args():
    parser = argparse.ArgumentParser(
        "LaBraM fine-tuning and evaluation script for EEG classification",
        add_help=False,
    )
    # --- 数据集与训练参数 ---
    parser.add_argument("--fold", default=7, type=int, help="指定交叉验证的折数")  # 一共十折
    parser.add_argument("--batch_size", default=64, type=int, help="每个GPU的批处理大小")
    parser.add_argument("--epochs", default=30, type=int, help="训练的总轮数")
    parser.add_argument("--update_freq", default=1, type=int, help="梯度累积的频率")
    parser.add_argument("--save_ckpt_freq", default=5, type=int, help="保存检查点的频率(每x轮)")

    # robust evaluation 鲁棒性评估数据集
    parser.add_argument("--robust_test", default=None, type=str, help="robust evaluation dataset")

    # 模型建立参数
    parser.add_argument("--model", default="labram_base_patch200_200",
                        type=str, metavar="MODEL", help="Name of model to train", )
    parser.add_argument("--input_size", default=200, type=int, help="EEG input size")

    # 模型架构策略
    parser.add_argument("--qkv_bias", action="store_true")
    parser.add_argument("--disable_qkv_bias", action="store_false", dest="qkv_bias")
    parser.set_defaults(qkv_bias=True)  # 默认开启 QKV 偏置​​

    parser.add_argument("--rel_pos_bias", action="store_true")
    parser.add_argument("--disable_rel_pos_bias", action="store_false", dest="rel_pos_bias")
    parser.set_defaults(rel_pos_bias=True)  # Relative Position Bias 相对位置偏置

    parser.add_argument("--abs_pos_emb", action="store_true")
    parser.set_defaults(abs_pos_emb=True)  # 使用绝对位置编码(Absolute Positional Embedding)

    parser.add_argument("--layer_scale_init_value", default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale",
                        )  # 层缩放(Layer Scale)初始值

    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT",
                        help="Dropout rate (default: 0.)")  # 默认不采用随机失活
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, metavar="PCT",
                        help="Attention dropout rate (default: 0.)")  # 默认注意力不采用随机失活
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT",
                        help="Drop path rate (default: 0.1)")  # 残差路径随机丢弃

    parser.add_argument("--disable_eval_during_finetuning", action="store_true", default=False)
    parser.add_argument("--model_ema", action="store_true", default=False)  # 模型指数移动平均(EMA)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999)  # EMA的​​衰减率
    parser.add_argument("--model_ema_force_cpu", action="store_true", default=False)  # 权重于CPU

    # 优化器参数
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER",
                        help='Optimizer (default: "adamw")')  # 优化算法选择
    parser.add_argument("--opt_eps", default=1e-8, type=float, metavar="EPSILON",
                        help="Optimizer Epsilon (default: 1e-8)")  # 数值稳定性常数,防止除零
    parser.add_argument("--opt_betas", default=None, type=float, nargs="+", metavar="BETA",
                        help="Optimizer Betas (default: None, use opt default)")  # 优化器参数
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM",
                        help="Clip gradient norm (default: None, no clipping)")  # 默认不使用梯度裁剪
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M",
                        help="SGD momentum (default: 0.9)")  # SGD(随机梯度下降)优化器的动量系数
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="weight decay (default: 0.05)")  # 权重衰减
    parser.add_argument("--weight_decay_end", type=float, default=None,
                        help="""Final value of the weight decay. 
        We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")  # 权重衰减最终值
    parser.add_argument("--lr", type=float, default=5e-4, metavar="LR",
                        help="learning rate (default: 5e-4)")  # 学习率
    parser.add_argument("--layer_decay", type=float, default=0.9)  # 学习率衰减
    parser.add_argument("--warmup_lr", type=float, default=1e-6, metavar="LR",
                        help="warmup learning rate (default: 1e-6)")  # 学习率预热
    parser.add_argument("--min_lr", type=float, default=1e-6, metavar="LR",
                        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)")  # 最小学习率
    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N",
                        help="epochs to warmup LR, if scheduler supports")  # 学习率预热轮数
    parser.add_argument("--warmup_steps", type=int, default=-1, metavar="N",
                        help="num of steps to warmup LR, will overload warmup_epochs if set > 0")
    parser.add_argument("--smoothing", type=float, default=0.1,
                        help="Label smoothing (default: 0.1)")  # 标签平滑

    # * Random Erase params 随机擦除参数
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT",
                        help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel",
                        help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1,
                        help="Random erase count (default: 1)")
    parser.add_argument("--resplit", action="store_true", default=False,
                        help="Do not random erase first (clean) augmentation split")

    # * Finetuning params 模型微调策略
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")  # 微调的预训练模型(源)
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)  # 模型前缀过滤, 选择性加载部分参数
    parser.add_argument("--model_filter_name", default="gzp", type=str)  # 权重过滤名
    parser.add_argument("--init_scale", default=0.001, type=float)  # 控制微调时新增层的权重初始化范围。
    parser.add_argument("--use_mean_pooling", action="store_true")  # 均值池化开关
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")
    parser.add_argument("--disable_weight_decay_on_rel_pos_bias", action="store_true", default=False)

    # Dataset parameters 训练策略
    parser.add_argument("--nb_classes", default=0, type=int, help="number of the classification types")  # 分类数
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")  # 输出目录
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")  # tensorboard日志目录
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")  # 从checkpoint恢复(路径)
    parser.add_argument("--auto_resume", action="store_true")  # 自动恢复
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")  # 指定参数的存储名称
    parser.set_defaults(auto_resume=True)
    parser.add_argument("--save_ckpt", action="store_true")  # 保存checkpoint
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")  # 恢复工作epoch
    parser.add_argument("--num_workers", default=10, type=int)  # 数据加载线程数
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")  # 是否锁内存,增效
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")  # 只进行评估
    parser.add_argument("--dist_eval", action="store_true", default=False,
                        help="Enabling distributed evaluation")  # 分布式评估

    # distributed training parameters 分布式训练参数
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")  # 进程数
    parser.add_argument("--local_rank", default=-1, type=int)  # 本地进程本地进程编号
    parser.add_argument("--dist_on_itp", action="store_true")  # 启用特定分布式后端
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--enable_deepspeed", action="store_true", default=False)  # 启用deepspeed优化库
    parser.add_argument("--dataset", default="TUAB", type=str, help="dataset: TUAB | TUEV")  # 数据集
    parser.add_argument("--gpu_id", default="7", type=str, help="")  # GPU编号
    parser.add_argument("--td", action="store_true", default=False, help="only TDBrain dataset")  # 仅使用TDBrain数据集

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize  # DeepSpeed初始化函数
        except:
            print("Please 'pip install deepspeed==0.4.0'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def get_models(args):
    return create_model(args.model,
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

path4 = '/data/sxk/labram2_resting_fine_pool/resting_unknown/57_Predict-Depression-Rest-New/60chs'

def prepare_data(seed, split=True, json_folder=None):
    seed = 12345
    np.random.seed(seed)
    # 读取数据路径
    root = "/data/sxk/labram2_resting_fine_pool"
    path_hc = "resting_unknown/SXMU_2_PROCESSED/HC/59chs"  # HC 健康对照组数据路径
    path_mdd = "resting_unknown/SXMU_2_PROCESSED/MDD/59chs"  # MDD 抑郁症数据路径
    hc = {"HC": [os.path.join(root, path_hc, file)
                 for file in os.listdir(os.path.join(root, path_hc))]}
    mdd = {"MDD": [os.path.join(root, path_mdd, file)
                   for file in os.listdir(os.path.join(root, path_mdd))]}


    data_path_with_cls = {k: v for d in [hc, mdd] for k, v in d.items()}

    ''' {'HC': ['/path/to/hc/subject1', '/path/to/hc/subject2', ...],
        'MDD': ['/path/to/mdd/subject101', '/path/to/mdd/subject102', ...]} '''


    sft_paths =[
        "/data/sxk/labram2_resting_fine_pool/resting_eye_open/Td_eyeopen/26chs",
        "/data/sxk/labram2_resting_fine_pool/resting_eye_close/Td_eyeclose/26chs",

        "/data/sxk/labram2_resting_fine_pool/resting_eye_open/QDHSM/20chs",
        "/data/sxk/labram2_resting_fine_pool/resting_eye_open/SXMU-ERP/20chs",
        "/data/sxk/labram2_resting_fine_pool/resting_eye_close/QDHSM/20chs",
        "/data/sxk/labram2_resting_fine_pool/resting_eye_close/SXMU-ERP/20chs",

        '/data/sxk/labram2_resting_fine_pool/resting_unknown/57_Predict-Depression-Rest-New/60chs',
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

    def get_fold_subjects(json_folder):
        # 读取json文件获取每一折的验证集被试
        fold_subjects = {}
        for i in range(10):  # todo 0
            json_file = os.path.join(json_folder, f'subs.json_cls10_rerun_ckpt001_{i}.json')  # all
            with open(json_file, 'r') as f:
                data = json.load(f)
                fold_subjects[i] = set(data['eval'])  # 使用集合便于查找
        return fold_subjects

    # 读取json文件中的折划分信息
    fold_subjects = get_fold_subjects(json_folder)  # 字典 键是折的编号 值是被试ID的集合(Set)
    # json文件结构：{0: {'sub1', 'sub5', ...}, 1: {'sub2', 'sub8', ...}, ...}

    # 打印每一折中的验证集被试数量
    print("\nJson文件中每折的验证集被试数量:")
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
    # 结构如下:{数据集路径 : [fold_0, fold_1, ..., fold_9]}, 每一个fold_list为一个numpy数组, 存放具体数据文件路径

    # 对有标签数据也按相同方式进行十折划分
    kfold_indices_with_cls = {}
    for key, paths in data_path_with_cls.items():
        subject_dict = {}
        for path in paths:
            file = os.path.basename(path)
            info_file = os.path.join(path, f"{file}_info.pkl")
            try:
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
    
    '''kfold_indices_with_cls = {
    'HC': [np.array(['/path/HC/subj_A/file1.pkl'] # Fold 0 ), ……, # Fold 1, ..., Fold 9 ],
    'MDD': [np.array(['/path/MDD/subj_X/file10.pkl'] # Fold 0 ), ……, # Fold 1, ..., Fold 9 ] }'''

    return kfold_indices_wo_cls, kfold_indices_with_cls



def get_dataset(args):
    seed = 12345
    np.random.seed(seed)
    #
    # todo 1
    #
    
    kfold_indices_wo_cls, kfold_indices_with_cls = prepare_data(seed,json_folder='10fold_split_dict/all')

    print(f"Fold {args.fold + 1}/10")

    train_indices_wo_cls = []  # 具体.pkl文件路径
    eval_indices_wo_cls = []
    train_indices_with_cls = dict()
    
    for dataset_key, folds in kfold_indices_wo_cls.items():  # key = dataset_path, value = [fold_0, ..., fold_9]
        train_fold = []
        eval_fold = []
        if(dataset_key == path4): 
            for i in range(10): eval_fold = folds[i].tolist()
        else: 
            for i in range(10): train_fold.extend(folds[i].tolist())
        train_indices_wo_cls.append(train_fold)
        eval_indices_wo_cls.append(eval_fold)  # list of list, 不同数据集的.pkl, 一个数据集的所有被试全部放一起

    for key, folds in kfold_indices_with_cls.items():  # key = 'HC' or 'MDD', value = [fold_0, fold_1, ..., fold_9]
        train_fold = []
        for i in range(10): train_fold.extend(folds[i].tolist())
        train_indices_with_cls[key] = train_fold

    # todo 2
    if (args.nb_classes == 1): used_ints = [0, 1]
    else: used_ints = [i for i in range(args.nb_classes)]

    dataset_train_list = []
    train_ch_names_list = []
    train_dataset_names = []

    # todo 3

    # hc mdd
    dataset = SFTSet_embedding(
        data_path=train_indices_with_cls, data_path_without_cls=None, clip=False, kept_ints=used_ints)
    dataset_train_list.append(dataset)
    train_ch_names_list.append(dataset.get_ch_names())  # EEG通道名称
    train_dataset_names.append(dataset.get_dataset_name())  # 数据集名称

    #
    for dataset_subject_paths in train_indices_wo_cls:
        dataset = SFTSet_embedding(
            data_path=None, data_path_without_cls=dataset_subject_paths, clip=False, kept_ints=used_ints)
        if (len(dataset) == 0): continue
        print("size of dataset:", len(dataset))
        dataset_train_list.append(dataset)
        train_ch_names_list.append(dataset.get_ch_names())
        train_dataset_names.append(dataset.get_dataset_name())

    dataset_eval_list = []
    eval_ch_names_list = []
    eval_dataset_names = []

    # todo 4

    # # # hc mdd
    for dataset_subject_paths in eval_indices_wo_cls:
        if dataset_subject_paths == []: continue
        dataset = SFTSet_embedding(
            data_path=None, data_path_without_cls=dataset_subject_paths, clip=False, kept_ints=used_ints)
        if (len(dataset) == 0): continue
        dataset_eval_list.append(dataset)
        eval_ch_names_list.append(dataset.get_ch_names())
        eval_dataset_names.append(dataset.get_dataset_name())

    metrics = ["accuracy", "balanced_accuracy"]  # 直接给定需要计算的指标

    # todo 5

    args.dataset_name = train_dataset_names

    # print(train_ch_names_list, eval_ch_names_list)

    total_train = 0
    total_eval = 0
    for dataset in dataset_train_list: print("len_train", len(dataset)); total_train += len(dataset)
    for dataset in dataset_eval_list: print("len_val", len(dataset)); total_eval += len(dataset)
    print('total_train', total_train)
    print('total_eval', total_eval)

    return ( dataset_train_list, train_ch_names_list,
             dataset_eval_list, eval_ch_names_list, metrics, used_ints)


def main(args, ds_init):
    utils.init_distributed_mode(args)  # 初始化 PyTorch 的分布式训练环境，支持多种启动方式(直接启动、SLURM、MPI等)

    # DeepSpeed初始化函数
    if ds_init is not None: utils.create_ds_config(args)

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
    ( dataset_train_list, train_ch_names_list, dataset_val_list, val_ch_names_list,
        metrics, used_ints) = get_dataset(args)

    print(dataset_train_list); print(train_ch_names_list)
    print(dataset_val_list); print(val_ch_names_list); print(metrics)

    if True:  # args.distributed:           分布式训练准备数据采样器(Sampler)
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
            sampler_train_list.append(sampler_train)
            # print("Sampler_train = %s" % str(sampler_train))
        sampler_eval_list = []
        if args.dist_eval:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
                sampler_eval_list.append(sampler_val)
        else:
            for dataset in dataset_val_list:
                sampler_val = torch.utils.data.SequentialSampler(dataset)
                sampler_eval_list.append(sampler_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else: log_writer = None

    data_loader_train_list = []  # 训练数据加载器(DataLoader)
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler, batch_size=args.batch_size, num_workers=2, pin_memory=args.pin_mem,
            drop_last=True, collate_fn=partial(custom_collate_fn, multi_label=True),
        )
        data_loader_train_list.append(data_loader_train)

    for data_loader in data_loader_train_list: print("iter: ", len(data_loader))

    if dataset_val_list is not None:
        data_loader_val_list = []  # 验证数据加载
        for dataset, sampler in zip(dataset_val_list, sampler_eval_list):
            data_loader_val = torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                # batch_size  = int(1.5 * args.batch_size),
                batch_size=args.batch_size, 
                num_workers=2, pin_memory=args.pin_mem, drop_last=False,
                collate_fn=partial(custom_collate_fn, multi_label=True),
            )
            data_loader_val_list.append(data_loader_val)
    else:
        data_loader_val_list = None

    model = get_models(args)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )  # checkpoint 为存放训练信息的字典, key如 model, optimizer, epoch, best_acc, best_epoch
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
                    new_dict[key[8:]] = checkpoint_model[key]  # 将key开头的student.去掉
                else: pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()  # 适配分类头, 移除与新任务不兼容的旧分类头, Transformer层的权重可以被加载
        for k in ["head.weight", "head.bias"]:
            if (k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None  # 指数移动平均(Exponential Moving Average, EMA)的模型
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model, decay=args.model_ema_decay, resume="",
            device="cpu" if args.model_ema_force_cpu else "",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数总数

    print("Model = %s" % str(model_without_ddp))  # 模型的完整架构字符串, 可以看到模型的每一层定义
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()  # 有效的总批次大小
    # num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    num_training_steps_per_epoch = (
        sum([len(dataset) // total_batch_size for dataset in dataset_train_list])
    )  # 每个训练周期 (epoch) 中, 模型权重更新次数
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    # print("Number of training examples = %d" % len(dataset_train))
    print("Number of training examples = %d" % sum([len(dataset) for dataset in dataset_train_list]))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None: print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()  # 模型中不使用权重衰减的参数名
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
        )
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )
        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_steps=args.warmup_steps, warmup_epochs=args.warmup_epochs,
    )  # 带预热的余弦退火学习率(LR)调度器, 生成整个训练过程中每一步对应学习率的列表
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs,
        num_training_steps_per_epoch,
    )  # 无预热权重衰减(WD)调度器, 生成整个训练过程中每一步对应权重衰减的列表
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数 Binary Cross Entropy
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)  # 标签平滑交叉熵损失函数
    else: criterion = torch.nn.CrossEntropyLoss()  # 多分类交叉熵损失函数 Cross Entropy Loss

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema,
    )  # 自动地从上一次中断的训练中恢复状态

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_balanced_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # data_loader_train.sampler.set_epoch(epoch)
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch_for_embedding(
            model, criterion, data_loader_train_list,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, 
            log_writer = log_writer,
            start_steps = epoch * num_training_steps_per_epoch,
            lr_schedule_values = lr_schedule_values,
            wd_schedule_values = wd_schedule_values,
            num_training_steps_per_epoch = num_training_steps_per_epoch,
            update_freq = args.update_freq,
            ch_names_list = train_ch_names_list,
            is_binary = args.nb_classes==1,
            args = args,
            used_ints = used_ints,
        )  # 进行一轮完整训练(遍历数据、前向传播、计算损失、反向传播、更新权重等)

        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args = args, model = model, model_without_ddp = model_without_ddp,
                optimizer = optimizer, loss_scaler = loss_scaler, epoch = epoch,
                model_ema = model_ema, save_ckpt_freq = args.save_ckpt_freq,
            )  # 保存模型

        if data_loader_val_list is not None:
            val_stats = evaluate_for_embedding(
                data_loader_val_list, model, device,
                header = "Val:", ch_names_list = val_ch_names_list,
                metrics = metrics,  # 传入需要计算的指标列表，如 ["accuracy", "balanced_accuracy"]
                is_binary = args.nb_classes==1,
                used_ints = used_ints, args=args,
            )
            print(
                "Accuracy of the network on the %d val EEG: %.2f"
                % (sum(len(dataset) for dataset in dataset_val_list), val_stats['accuracy'])
            )

            if max_balanced_accuracy < val_stats["balanced_accuracy_old"]:
                max_balanced_accuracy = val_stats["balanced_accuracy_old"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=model_ema,
                    )  # 保存最佳模型
            print(f"Max max_balanced_accuracy val: {max_balanced_accuracy:.2f}")

            if log_writer is not None:
                for key, value in val_stats.items():
                    if key == "accuracy": log_writer.update(accuracy=value, head="val", step=epoch)
                    elif key == "balanced_accuracy": log_writer.update(balanced_accuracy=value, head="val", step=epoch)
                    elif key == "balanced_accuracy_old": log_writer.update(balanced_accuracy_old=value, head="val", step=epoch)
                    elif key == "accuracy_old": log_writer.update(accuracy_old=value, head="val", step=epoch)
                    elif key == "f1_weighted": log_writer.update(f1_weighted=value, head="val", step=epoch)
                    elif key == "pr_auc": log_writer.update(pr_auc=value, head="val", step=epoch)
                    elif key == "roc_auc": log_writer.update(roc_auc=value, head="val", step=epoch)
                    elif key == "cohen_kappa": log_writer.update(cohen_kappa=value, head="val", step=epoch)
                    elif key == "loss": log_writer.update(loss=value, head="val", step=epoch)
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch, "n_parameters": n_parameters, }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch, "n_parameters": n_parameters,
            }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir: Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)  # 使用指定的GPU

    main(opts, ds_init)

# --finetune finetune_model/checkpoint-4.pth \
# used_ints
# args.nb_classes
# hc ，mdd
# json_folder = '10fold_split_dict/all'