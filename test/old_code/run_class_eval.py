import argparse, json, os
import numpy as np
import datetime, time
import torch
import torch.backends.cudnn as cudnn

from functools import partial
from pathlib import Path
from collections import OrderedDict
from timm.models import create_model

import utils, modeling_finetune
from engine_for_finetuning import evaluate_for_embedding
from data_processor.dataset import SFTSet_embedding, custom_collate_fn


def get_args_eval():
    parser = argparse.ArgumentParser("LaBraM 10-Fold Single-GPU Evaluation Script", add_help=False)
    # --- 关键参数 ---
    parser.add_argument("--checkpoint_base_dir", required=True, type=str)
    parser.add_argument("--scenario", required=True, type=int, default=0)
    parser.add_argument("--log_dir", default="./eval_logs", help="Path to save TensorBoard logs.")
    
    # --- 模型和评估参数 ---
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--nb_classes", default=1, type=int)
    parser.add_argument("--model", default="labram_base_patch200_200", type=str)
    
    # --- 硬件和环境参数 ---
    parser.add_argument("--gpu_id", default="0", type=str, help="GPU ID to use.")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", default=True)

    # --- 模型架构参数 (应与训练时一致) ---
    parser.add_argument("--input_size", default=200, type=int)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--use_mean_pooling", action="store_true", default=True)
    parser.add_argument("--disable_rel_pos_bias", action="store_false", dest="rel_pos_bias", default=True)
    parser.add_argument("--abs_pos_emb", action="store_true", dest="abs_pos_emb", default=True)
    parser.add_argument("--disable_qkv_bias", action="store_false", dest="qkv_bias", default=True)
    parser.add_argument("--layer_scale_init_value", default=0.1, type=float)
    
    return parser.parse_args()


# 这个函数和之前一样，用于加载硬编码的测试数据集
def prepare_test_data(args):
    print("--- Loading Hardcoded Test Datasets ---")
    if args.scenario == 0: sft_paths = [
            "/data/sxk/labram2_resting_fine_pool/resting_eye_open/Td_eyeopen/26chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_close/Td_eyeclose/26chs",
        ]
    elif args.scenario == 1: sft_paths = [
            "/data/sxk/labram2_resting_fine_pool/resting_eye_open/Parkinson_eyes_open_PROCESSED/61chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_open/Parkinson_eyes_open_PROCESSED/63chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_close/AD_FD_HC_PROCESSED/19chs",
            "/data/sxk/labram2_resting_fine_pool/resting_unknown/PREDICT-PD_LPC_Rest/61chs",
            "/data/sxk/labram2_resting_fine_pool/resting_unknown/PREDICT-PD_LPC_Rest/63chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_open/PREDICT-PD_LPC_Rest_2/63chs",
        ]
    elif args.scenario == 2: sft_paths = [
            "/data/sxk/labram2_resting_fine_pool/resting_unknown/CSA-PD-W/30chs",
            "/data/sxk/labram2_resting_fine_pool/resting_unknown/Parkinson/32chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_close/Porn-addiction/19chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_open/Porn-addiction/19chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_close/EEG_in_SZ/19chs",
        ]
    elif args.scenario == 3: sft_paths = [
            "/data/sxk/labram2_resting_fine_pool/resting_unknown/First_Episode_Psychosis_Control_1/60chs",
            "/data/sxk/labram2_resting_fine_pool/resting_unknown/First_Episode_Psychosis_Control_2/60chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_open/QDHSM/20chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_open/SXMU-ERP/20chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_close/QDHSM/20chs",
            "/data/sxk/labram2_resting_fine_pool/resting_eye_close/SXMU-ERP/20chs",
        ]
    elif args.scenario == 4: sft_paths = [
            '/data/sxk/labram2_resting_fine_pool/resting_eye_open/HUAWEI_eye_open/30chs',
            '/data/sxk/labram2_resting_fine_pool/resting_eye_open/HUAWEI_eye_open/59chs',
            '/data/sxk/labram2_resting_fine_pool/resting_eye_close/HUAWEI_eye_close/30chs',
            '/data/sxk/labram2_resting_fine_pool/resting_eye_close/HUAWEI_eye_close/59chs',
        ]
    elif args.scenario == 5: sft_paths = [
            "/data/sxk/labram2_resting_fine_pool/resting_unknown/LEMON/61chs",
            '/data/sxk/labram2_resting_fine_pool/resting_unknown/57_Predict-Depression-Rest-New/60chs',
            '/data/sxk/labram2_resting_fine_pool/resting_eye_close/60_ANDing/20chs',
            '/data/sxk/labram2_resting_fine_pool/resting_eye_open/60_ANDing/20chs',
        ]

    print(f"Selected scenario: '{args.scenario}' with {len(sft_paths)} dataset paths to load.")
    dataset_test_list, test_ch_names_list, test_dataset_names = [], [], []
    used_ints = [i for i in range(args.nb_classes)] if args.nb_classes > 1 else [0, 1]

    for dataset_path in sft_paths:
        all_subject_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
        if not all_subject_paths: continue
        dataset = SFTSet_embedding(data_path=None, data_path_without_cls=all_subject_paths, clip=False, kept_ints=used_ints)
        if len(dataset) > 0:
            dataset_test_list.append(dataset)
            test_ch_names_list.append(dataset.get_ch_names())
            test_dataset_names.append(dataset.get_dataset_name())
            print(f"Successfully loaded {len(dataset)} samples from {dataset.get_dataset_name()}.")

    metrics = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "cohen_kappa"]
    return dataset_test_list, test_ch_names_list, metrics, used_ints


def get_models(args):
    return create_model(args.model, 
            pretrained=False, 
            num_classes=args.nb_classes, 
            drop_rate=0.0,
            drop_path_rate=args.drop_path, 
            attn_drop_rate=0.0, 
            drop_block_rate=None, 
            use_mean_pooling=args.use_mean_pooling,
            init_scale=0.001,
            use_rel_pos_bias=args.rel_pos_bias, 
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value, 
            qkv_bias=args.qkv_bias)

def main(args):
    print(args)
    device = torch.device(args.device)
    
    # 固定随机种子以保证可复现性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # 1. 初始化 TensorBoard Logger
    os.makedirs(args.log_dir, exist_ok=True)
    log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    # 2. 加载一次测试数据
    dataset_test_list, test_ch_names_list, metrics, used_ints = prepare_test_data(args)
    if not dataset_test_list: print("No datasets to evaluate. Exiting."); return

    # 3. 创建数据加载器 (使用 SequentialSampler)
    data_loader_test_list = []
    for dataset in dataset_test_list:
        sampler_test = torch.utils.data.SequentialSampler(dataset)
        data_loader_test_list.append(torch.utils.data.DataLoader(
            dataset, sampler=sampler_test, batch_size=args.batch_size, num_workers=2,
            pin_memory=args.pin_mem, drop_last=False, collate_fn=partial(custom_collate_fn, multi_label=True)
        ))
            
    # 4. 开始10折评估循环
    all_folds_stats = {}
    for fold_num in range(10):
        print(f"\n{'='*20} Evaluating Fold {fold_num}/9 {'='*20}")
        
        # 4.1 动态构建模型路径
        checkpoint_path = Path(args.checkpoint_base_dir) / f"fold{fold_num}" / "checkpoint-best.pth"
        if not checkpoint_path.is_file():
            print(f"Warning: Checkpoint not found for fold {fold_num} at {checkpoint_path}. Skipping.")
            continue
        
        # 4.2 重新创建模型实例
        model = get_models(args)
        
        # 4.3 加载权重
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_model = checkpoint.get('model', checkpoint)
        if any(key.startswith("student.") for key in checkpoint_model.keys()):
            checkpoint_model = OrderedDict((k[8:], v) for k, v in checkpoint_model.items() if k.startswith("student."))
        utils.load_state_dict(model, checkpoint_model, prefix="")
        
        # 4.4 将模型发送到指定设备 (无DDP包装)
        model.to(device)
            
        # 4.5 执行评估
        test_stats = evaluate_for_embedding(
            data_loader_test_list, model, device, header=f"Test Fold {fold_num}:",
            ch_names_list=test_ch_names_list, metrics=metrics,
            is_binary=args.nb_classes == 1, used_ints=used_ints, args=args,
        )
        all_folds_stats[f"fold_{fold_num}"] = test_stats
        
        # 4.6 将结果写入 TensorBoard
        log_writer.set_step(fold_num)
        for key, value in test_stats.items():
            log_writer.update(**{key: value}, head="evaluation")
        print(f"Fold {fold_num} metrics logged to TensorBoard.")

    # # 5. 循环结束后，统一处理和打印所有结果
    # print(f"\n{'='*25} All Folds Evaluated Summary {'='*25}")
    # print(json.dumps(all_folds_stats, indent=4))
    
    # 计算并打印平均指标
    avg_stats = {}
    for metric_name in metrics + ["loss", "accuracy_old", "balanced_accuracy_old"]:
        values = [all_folds_stats[f"fold_{i}"][metric_name] for i in range(10) if f"fold_{i}" in all_folds_stats and metric_name in all_folds_stats[f"fold_{i}"]]
        if values:
            avg_stats[f"avg_{metric_name}"] = np.mean(values)
            avg_stats[f"std_{metric_name}"] = np.std(values)
    
    print("\n--- Average Performance Across Folds ---")
    print(json.dumps(avg_stats, indent=4))
    
    # 将平均值写入TensorBoard
    log_writer.set_step(10) 
    for key, value in avg_stats.items():
        log_writer.update(**{key: value}, head="summary")
    log_writer.flush()
    print("\nAverage metrics logged to TensorBoard at step 10.")

if __name__ == "__main__":
    opts = get_args_eval()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
    main(opts)