import argparse, json, os
import numpy as np
import datetime, time
import torch, pickle  # 让数据持久化保存
import torch.backends.cudnn as cudnn  # 优化cuDNN性能
from collections import defaultdict
from pathlib import Path  # 用于处理文件路径
from collections import OrderedDict  # 有序字典，用于加载模型权重
from timm.models import create_model  # timm库中用于创建模型的函数
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEma  # timm库中的模型指数移动平均（EMA）工具
from optim_factory import (create_optimizer, get_parameter_groups, LayerDecayValueAssigner)
import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from engine_for_finetuning import train_one_epoch_for_embedding, evaluate_for_embedding
from data_processor.dataset import SFTSet_embedding, custom_collate_fn_with_ids
import modeling_finetune # 用于注册模型
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from swanlab.integration.pytorch_lightning import SwanLabLogger
import datetime


# 计算标准化统计量
def calculate_normalization_stats(dataset_list):
    """从训练数据集中计算HAMD₀的均值和标准差"""
    print("Calculating normalization stats from training data...")
    # 将多个数据集合并进行计算
    full_train_dataset = torch.utils.data.ConcatDataset(dataset_list)
    # 使用一个临时的DataLoader来遍历数据
    temp_loader = torch.utils.data.DataLoader(
        full_train_dataset, batch_size=256, # 使用较大的batch size加速计算
        shuffle=False, num_workers=4, collate_fn=custom_collate_fn_with_ids
    )
    
    # 解包以获取 hamd0 tensor
    hamd0_values = []
    for batch in temp_loader:
        _, _, _, _, hamd0 = batch
        hamd0_values.append(hamd0)
        
    all_hamd0s = torch.cat(hamd0_values)
    mean = all_hamd0s.mean().item()
    std = all_hamd0s.std().item()
    
    if std < 1e-6: std = 1.0 # 防止除以零
    print(f"Normalization stats calculated: HAMD0_mean={mean:.4f}, HAMD0_std={std:.4f}")
    return {"mean": mean, "std": std}


# 简易的 Normalizer 类，用于传递统计数据
class SimpleNormalizer:
    def __init__(self, stats):
        self.hamd0_mean = stats['mean']
        self.hamd0_std = stats['std']
    def normalize_hamd0(self, hamd0):
        return (hamd0 - self.hamd0_mean) / self.hamd0_std


def get_args():
    parser = argparse.ArgumentParser("LaBraM fine-tuning for Regression", add_help=False)
    # --- 数据集与训练参数 ---
    parser.add_argument("--fold", default=7, type=int, help="指定交叉验证的折数")  # 一共十折
    parser.add_argument("--batch_size", default=64, type=int, help="每个GPU的批处理大小")
    parser.add_argument("--epochs", default=30, type=int, help="训练的总轮数")
    parser.add_argument("--update_freq", default=1, type=int, help="梯度累积的频率")
    parser.add_argument("--save_ckpt_freq", default=5, type=int, help="保存检查点的频率(每x轮)")

    # robust evaluation 鲁棒性评估数据集
    parser.add_argument("--robust_test", default=None, type=str, help="robust evaluation dataset")

    # 模型建立以及保存日志
    parser.add_argument("--model", default="labram_base_patch200_200",
                        type=str, metavar="MODEL", help="Name of model to train", )
    parser.add_argument("--input_size", default=200, type=int, help="EEG input size")
    parser.add_argument("--back_up", action="store_true", default = False,
                        help="Whether to back up and resume training")
    parser.add_argument("--enable_tensorboard",action="store_true", default=False)
    parser.add_argument("--swanlab_project", default="w_test", help ="projiect name")
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
    parser.add_argument("--nb_classes", default=1, type=int, help="only support for 1")  
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
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader\
                         for more efficient (sometimes) transfer to GPU.")  # 是否锁内存,增效
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

    parser.add_argument("--save_name", default="regression_experiment", type=str)    # 保存名称
    parser.add_argument("--enable_deepspeed", action="store_true", default=False)   # 启用deepspeed优化库
    parser.add_argument("--dataset", default="TUAB", type=str, help="dataset: TUAB | TUEV")  # 数据集
    parser.add_argument("--gpu_id", default="7", type=str, help="")  # GPU编号
    
    
    ## 药物嵌入
    parser.add_argument("--num_drugs", default=5, type=int, help="Number of drug types for embedding layer.")
    parser.add_argument("--drug_embed_dim", default=8, type=int, help="Dimension of the drug embedding.")
    
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
    return create_model(
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
        num_drugs=args.num_drugs,   ## 药物嵌入数量
        drug_embed_dim=args.drug_embed_dim, ## 药物嵌入维度
        qkv_bias=args.qkv_bias,
    )

def prepare_data(seed):
    seed = 12345
    np.random.seed(seed)

    # 1. 遍历所有记录，建立“真正被试ID”到“文件路径”和“标签”的映射
    sft_paths = ["/data1/wangkuiyu/LabraM/processed_data"]
    
    # key: 真正的被试ID, value: list of recording_folder_paths
    true_subject_to_paths_map = defaultdict(list)
    subject_to_label_map = {}

    print("开始扫描所有记录，使用 'subject_remission' 作为分类标签...")
    
    for base_path in sft_paths:
        if not os.path.isdir(base_path): continue
        
        for dataset_folder in os.listdir(base_path): # e.g., '26chs'
            dataset_path = os.path.join(base_path, dataset_folder)
            if not os.path.isdir(dataset_path) or dataset_folder.startswith('.'): continue

            for recording_folder in os.listdir(dataset_path): # e.g., '605776'
                recording_path = os.path.join(dataset_path, recording_folder)
                if not os.path.isdir(recording_path) or recording_folder.startswith('.'): continue
                
                try:
                    info_file = os.path.join(recording_path, f"{recording_folder}_info.pkl")
                    if not os.path.exists(info_file): continue
                    
                    with open(info_file, 'rb') as f:
                        info_dict = pickle.load(f)
                    
                    # 使用 'subject_remission' 作为标签 
                    expected_keys = ['subject_id_dateset', 'subject_remission']
                    if not all(key in info_dict for key in expected_keys):
                        continue    # 跳过缺少必要键的文件
                        
                    true_subject_id = info_dict['subject_id_dateset']
                    remission_label = info_dict['subject_remission']
                    
                    # 确保标签是 0 或 1，而不是 None 或其他值
                    if remission_label not in [0, 1]:
                        # print(f"警告:{recording_path} 的 'subject_remission' 无效，已跳过记录 。")
                        continue
                    
                    true_subject_to_paths_map[true_subject_id].append(recording_path)
                    subject_to_label_map[true_subject_id] = remission_label
                    
                except Exception as e:
                    # print(f"处理文件夹 {recording_path} 时出错: {e}")
                    continue

    # 2. 被试ID级十折划分
    print("\n被试ID级分层十折划分...")
    
    all_true_subjects = list(subject_to_label_map.keys())
    if not all_true_subjects:
        raise ValueError("错误：没有找到任何带有有效 'subject_remission' 标签的被试数据。")

    label_0_subjects = [subj for subj in all_true_subjects if subject_to_label_map[subj] == 0]
    label_1_subjects = [subj for subj in all_true_subjects if subject_to_label_map[subj] == 1]

    print(f"有效被试总数: {len(all_true_subjects)}")
    print(f"  (其中 Remission 0: {len(label_0_subjects)} 人, Remission 1: {len(label_1_subjects)} 人)")
    
    np.random.shuffle(label_0_subjects)
    np.random.shuffle(label_1_subjects)

    folds_by_subject_id = {i: set() for i in range(10)}
    
    for i, subject in enumerate(label_0_subjects):
        folds_by_subject_id[i % 10].add(subject)
    
    for i, subject in enumerate(label_1_subjects):
        folds_by_subject_id[i % 10].add(subject)

    print("\n动态分折统计:")
    for i in range(10):
        l0_count = sum(1 for s in folds_by_subject_id[i] if subject_to_label_map[s] == 0)
        l1_count = sum(1 for s in folds_by_subject_id[i] if subject_to_label_map[s] == 1)
        print(f"  折 {i}: {len(folds_by_subject_id[i])} 被试 ({l0_count}个Remission 0, {l1_count}个Remission 1)")

    # 3. 根据分折结果，组装最终的记录文件夹路径列表
    print("\n正在根据分折结果组装文件路径...")
    kfold_indices_wo_cls = {}
    
    for base_path in sft_paths:
        dataset_folds = []
        for fold_idx in range(10):
            fold_files = []
            subjects_in_this_fold = folds_by_subject_id[fold_idx]
            
            for subject_id in subjects_in_this_fold:
                paths_for_this_subject = true_subject_to_paths_map[subject_id]
                fold_files.extend(paths_for_this_subject)
                
            dataset_folds.append(np.array(fold_files))
        kfold_indices_wo_cls[base_path] = dataset_folds
        
    print("数据准备完成！")
    return kfold_indices_wo_cls

def get_dataset(args):
    seed = 12345
    np.random.seed(seed)
    #
    # todo 1
    #
    kfold_indices_wo_cls = prepare_data(seed)

    print(f"Fold {args.fold + 1}/10")

    train_indices_wo_cls = []  # 具体.pkl文件路径
    eval_indices_wo_cls = []

    for dataset_key, folds in kfold_indices_wo_cls.items():  # key = dataset_path, value = [fold_0, ..., fold_9]
        train_fold = []
        eval_fold = []
        for i in range(10):
            if i == args.fold:
                eval_fold = folds[i].tolist()
            else:
                train_fold.extend(folds[i].tolist())
        train_indices_wo_cls.append(train_fold)
        eval_indices_wo_cls.append(eval_fold)  # list of list, 不同数据集的.pkl, 一个数据集的所有被试全部放一起

    # todo 2
    if (args.nb_classes == 1): used_ints = [0, 1]
    else: used_ints = [i for i in range(args.nb_classes)]

    dataset_train_list = []
    train_ch_names_list = []
    train_dataset_names = []

    # todo 3
    for dataset_subject_paths in train_indices_wo_cls:
        dataset = SFTSet_embedding(
            data_path=None, data_path_without_cls=dataset_subject_paths, clip=False, kept_ints=used_ints
        )
        if (len(dataset) == 0): continue
        print("size of dataset:", len(dataset))
        dataset_train_list.append(dataset)
        train_ch_names_list.append(dataset.get_ch_names())
        train_dataset_names.append(dataset.get_dataset_name())

    dataset_eval_list = []
    eval_ch_names_list = []
    eval_dataset_names = []

    # todo 4
    for dataset_subject_paths in eval_indices_wo_cls:
        if dataset_subject_paths == []: continue
        dataset = SFTSet_embedding(
            data_path=None, data_path_without_cls=dataset_subject_paths, clip=False, kept_ints=used_ints
        )
        if (len(dataset) == 0): continue
        dataset_eval_list.append(dataset)
        eval_ch_names_list.append(dataset.get_ch_names())
        eval_dataset_names.append(dataset.get_dataset_name())

    metrics = ["accuracy", "balanced_accuracy"]  # 直接给定需要计算的指标

    # todo 5

    args.dataset_name = train_dataset_names
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
        used_ints,
    )


def main(args, ds_init):
    utils.init_distributed_mode(args)  # 初始化 PyTorch 的分布式训练环境，支持多种启动方式(直接启动、SLURM、MPI等)
    args.nb_classes = 1
    assert args.nb_classes == 1 # regression
    
    # DeepSpeed初始化函数
    if ds_init is not None: utils.create_ds_config(args)

    date_str = datetime.datetime.now().strftime("%m%d").lstrip("0")
    args.save_name = f"{date_str}_drug_fold_{args.fold}_lr_{args.lr}\
            _wd_{args.weight_decay}_warm_{args.warmup_epochs}_llrd{args.layer_decay}"
    
    # print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_train, dataset_test, dataset_val: follows the standard format of torch.utils.data.Dataset.
    # ch_names: list of strings, channel names of the dataset. It should be in capital letters.
    (   dataset_train_list, train_ch_names_list,
        dataset_val_list, val_ch_names_list,
        # metrics, used_ints, regresion取消这些参数
        _, _,
    ) = get_dataset(args)

    # 微调时禁用评估
    if args.disable_eval_during_finetuning: dataset_val = None

    if True:  # args.distributed:           分布式训练准备数据采样器(Sampler)
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
            sampler_train_list.append(sampler_train)

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

    data_loader_train_list = []  # 训练数据加载器(DataLoader)
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            # num_workers = args.num_workers,
            num_workers=2,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=custom_collate_fn_with_ids,
        )
        data_loader_train_list.append(data_loader_train)

    for data_loader in data_loader_train_list:
        print("iter: ", len(data_loader))

    if dataset_val_list is not None:
        data_loader_val_list = []  # 验证数据加载
        for dataset, sampler in zip(dataset_val_list, sampler_eval_list):
            data_loader_val = torch.utils.data.DataLoader(
                dataset, sampler=sampler, batch_size=args.batch_size,
                num_workers=2, pin_memory=args.pin_mem,
                drop_last=False, collate_fn=custom_collate_fn_with_ids,
            )
            data_loader_val_list.append(data_loader_val)
    else:
        data_loader_val_list = None

    # 2. 新增创建Normalizer
    norm_stats = calculate_normalization_stats(dataset_train_list)
    normalizer = SimpleNormalizer(norm_stats)
    
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
        else: checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None: checkpoint_model = checkpoint
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
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数总数

    # print("Model = %s" % str(model_without_ddp))  # 模型的完整架构字符串, 可以看到模型的每一层定义
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()  # 有效的总批次大小
    # num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    num_training_steps_per_epoch = (
        sum([len(dataset) // total_batch_size for dataset in dataset_train_list])
    )  # 每个训练周期 (epoch) 中, 模型权重更新次数
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % sum([len(dataset) for dataset in dataset_train_list]))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else: assigner = None

    skip_weight_decay_list = model.no_weight_decay()  # 模型中不使用权重衰减的参数名
    if args.disable_weight_decay_on_rel_pos_bias:
        for i in range(num_layers):
            skip_weight_decay_list.add("blocks.%d.attn.relative_position_bias_table" % i)

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
            args=args, model=model, model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )
        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True
            )
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
    if args.weight_decay_end is None: args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch
    )  # 无预热权重衰减(WD)调度器, 生成整个训练过程中每一步对应权重衰减的列表
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    # 3. 修改损失函数为 HuberLoss
    criterion = torch.nn.HuberLoss()
    print("criterion = %s" % str(criterion))

    if(args.back_up):
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema,
        )  # 自动地从上一次中断的训练中恢复状态

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    loggers = []
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.enable_tensorboard:
        tb_logger = pl_loggers.TensorBoardLogger(
            "fine_labramlogs/regression", version=args.save_name + f'_{args.fold + 1}')
        loggers.append(tb_logger)
    swanlab_logger = SwanLabLogger(
        project=args.swanlab_project, workspace='01', reinit=True,
        experiment_name=args.save_name + '_fold' + f'{args.fold + 1}_{current_time}',
    )
    loggers.append(swanlab_logger)
    
    max_performance_metric = -np.inf  # 对于Pearson越大越好
    best_detailed_metrics_to_save = None
    best_subject_level_df = None
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_for_embedding(
            model, criterion, data_loader_train_list, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema,
            loggers=loggers, normalizer=normalizer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq, ch_names_list=train_ch_names_list, args=args
        )

        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema,
                save_ckpt_freq=args.save_ckpt_freq,
            )

        if data_loader_val_list is not None and not args.disable_eval_during_finetuning:
            val_stats, val_metrics_to_log, subject_level_df = evaluate_for_embedding(
                data_loader_val_list, model, device, header="Val:",
                ch_names_list=val_ch_names_list, normalizer=normalizer, args=args 
            )
            print(
                f"\nValidation on {sum(len(d.dataset) for d in data_loader_val_list)} subjects: "
                f"MAE={val_stats['mae']:.4f}, R2={val_stats['r2']:.4f}, Pearson={val_stats['pearson']:.4f}"
            )

            # 以Pearson相关系数作为选择最佳模型的标准
            current_performance = val_stats.get("pearson", -np.inf)
            if max_performance_metric < current_performance:
                max_performance_metric = current_performance
                best_detailed_metrics_to_save = val_metrics_to_log.copy()
                best_detailed_metrics_to_save['best_epoch'] = epoch
                best_subject_level_df = subject_level_df.copy() 
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp,
                        optimizer=optimizer, loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                
                
            print(f"Max Pearson Correlation on val set: {max_performance_metric:.4f}\n")

            if loggers:
                for logger in loggers: logger.log_metrics(metrics=val_metrics_to_log, step=epoch)
            
            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                         **{f"val_{k}": v for k, v in val_stats.items()}, "epoch": epoch}
        else: log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

        log_stats_std = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in log_stats.items()}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_std) + "\n")
    
    if args.output_dir and utils.is_main_process() and best_detailed_metrics_to_save is not None:
        final_metrics_path = os.path.join(args.output_dir, "best_metrics.json")
        try:
            with open(final_metrics_path, 'w', encoding='utf-8') as f:
                json.dump(best_detailed_metrics_to_save, f, ensure_ascii=False, indent=4)
            print(f"\nFinal best metrics for fold {args.fold} saved to {final_metrics_path}")
        except Exception as e:
            print(f"\nError saving final best metrics to JSON: {e}")   
            
        if best_subject_level_df is not None:
            best_preds_path = os.path.join(args.output_dir, "best_epoch_predictions.json")
            try:    # 将drug_id转换回可读的药物名称, 保存为JSON文件
                id_to_drug_name_map = {v: k for k, v in utils.drug_name_to_id.items()}
                best_subject_level_df['drug_name'] = best_subject_level_df['drug_id'].map(id_to_drug_name_map).fillna('unknown')
                best_subject_level_df.to_json(best_preds_path, orient='records', indent=4, force_ascii=False)
                print(f"Final best epoch prediction data for fold {args.fold} saved to {best_preds_path}")
            except Exception as e:
                print(f"\nError saving final best prediction data to JSON: {e}") 
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir: Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)  # 使用指定的GPU
    main(opts, ds_init)

# --finetune finetune_model/checkpoint-4.pth \

