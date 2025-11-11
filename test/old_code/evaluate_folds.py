import argparse
import os
import torch
from functools import partial
from pathlib import Path  # <--- 修正1: 正确导入 Path

# 从你现有的项目中导入必要的函数和类
from run_class_finetune_for_embedding import get_models, get_dataset, custom_collate_fn
from engine_for_finetuning import evaluate_for_embedding
import utils


def get_eval_args():
    parser = argparse.ArgumentParser("LaBraM 10-Fold Evaluation Script", add_help=False)

    # --- 核心评估参数 ---
    parser.add_argument("--base_checkpoint_dir", required=True, type=str,
                        help="包含fold0, fold1...子文件夹的基础路径")
    parser.add_argument("--output_dir", default="./evaluation_results", type=str,
                        help="保存.npy文件和混淆矩阵图像的目录")
        # --- 新增参数: 指定要评估的epoch ---
    parser.add_argument("--epochs", type=str, nargs='+', default=None,
                        help="要评估的特定epoch列表。如果未提供,则默认使用'checkpoint-best.pth'")

    # --- 数据集相关参数 (必须和训练时保持一致) ---
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--td", action='store_true', default=False, help="是否使用TDBrain数据集")
    parser.add_argument("--pin_mem", action='store_true', default=True)
    parser.add_argument("--no_pin_mem", action='store_false', dest="pin_mem")

    # --- 设备参数 ---
    parser.add_argument("--device", default="cuda", help="设备 (cuda or cpu)")
    parser.add_argument("--gpu_id", default="0", type=str, help="GPU ID")

    # --- 模型和分类参数 ---
    parser.add_argument("--model", default="labram_base_patch200_200", type=str, help="模型名称")
    parser.add_argument("--nb_classes", default=1, type=int, help="分类类别数")
    parser.add_argument("--input_size", default=200, type=int)

    parser.add_argument("--qkv_bias", action="store_true")
    parser.add_argument("--disable_qkv_bias", action="store_false", dest="qkv_bias")
    parser.set_defaults(qkv_bias=True)

    parser.add_argument("--rel_pos_bias", action="store_true")
    parser.add_argument("--disable_rel_pos_bias", action="store_false", dest="rel_pos_bias")
    parser.set_defaults(rel_pos_bias=True)

    parser.add_argument("--abs_pos_emb", action="store_true")
    parser.set_defaults(abs_pos_emb=True)

    parser.add_argument("--layer_scale_init_value", default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, metavar="PCT")
    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT")

    parser.add_argument("--use_mean_pooling", action="store_true")
    parser.set_defaults(use_mean_pooling=True)

    parser.add_argument("--init_scale", default=0.001, type=float)
    parser.add_argument("--seed", default=12345, type=int)
    return parser.parse_args()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- 核心修正 ---
    # 1. 确定要评估的epoch列表。
    #    如果命令行没有提供`--epochs`，`args.epochs`为None，`epochs_to_evaluate`则为`['best']`。
    raw_epochs = args.epochs if args.epochs is not None else ['best']

    # 2. 【新增逻辑】处理类型：将数字字符串转为整数，保持 'best' 为字符串
    epochs_to_evaluate = []
    for epoch_str in raw_epochs:
        if epoch_str.isdigit():
            epochs_to_evaluate.append(int(epoch_str))
        elif epoch_str.lower() == 'best':
            epochs_to_evaluate.append('best')
        else:
            print(f"警告: 无法识别的epoch值 '{epoch_str}'，将被忽略。")
            continue
    # 2. **使用for循环遍历列表**。这样 `epoch` 变量就是单个整数（如 59），而不是列表（[59]）
    for epoch in epochs_to_evaluate:
        print(f"\n{'#' * 25} EVALUATING EPOCH: {epoch} {'#' * 25}")

        for fold_idx in range(10):
            print(f"\n{'=' * 20} Processing Fold {fold_idx} for Epoch {epoch} {'=' * 20}")

            # 3. **正确构建模型文件名**。因为 `epoch` 是整数，f-string会正确生成 "checkpoint-59.pth"
            checkpoint_filename = f"checkpoint-{epoch}.pth" if epoch != 'best' else "checkpoint-best.pth"
            model_path = os.path.join(args.base_checkpoint_dir, f"fold{fold_idx}", checkpoint_filename)

            if not os.path.exists(model_path):
                print(f"警告: 找不到模型 {model_path}, 跳过。")
                continue

            # 4. **将单个epoch信息传递给评估函数**，这是让文件名正确保存的关键
            data_args = argparse.Namespace(**vars(args))
            data_args.fold = fold_idx
            data_args.eval = True # 确保getattr(args, 'eval', False)为True
            data_args.output_dir = args.output_dir
            data_args.epoch = epoch # **将单个epoch整数传递下去**

            # (其余代码保持不变)
            (_, _, dataset_val_list, val_ch_names_list, metrics, used_ints) = get_dataset(data_args)
            if not dataset_val_list or not any(dataset_val_list):
                print(f"警告: 第 {fold_idx} 折没有验证数据, 跳过。")
                continue

            data_loader_val_list = []
            for dataset in dataset_val_list:
                if len(dataset) > 0:
                    loader = torch.utils.data.DataLoader(
                        dataset, sampler=torch.utils.data.SequentialSampler(dataset),
                        batch_size=args.batch_size, num_workers=2,
                        pin_memory=args.pin_mem, drop_last=False,
                        collate_fn=partial(custom_collate_fn, multi_label=True)
                    )
                    data_loader_val_list.append(loader)

            if not data_loader_val_list:
                print(f"警告: 第 {fold_idx} 折的数据加载器为空, 跳过。")
                continue

            model = get_models(args)
            checkpoint = torch.load(model_path, map_location='cpu')
            model_state_dict = checkpoint.get('model', checkpoint.get('model_ema', checkpoint))
            utils.load_state_dict(model, model_state_dict)
            print(f"成功从 {model_path} 加载模型。")
            model.to(device)
            model.eval()

            # 调用评估函数，此时的 data_args 包含了正确的 .epoch 属性
            evaluate_for_embedding(
                data_loader_val_list, model, device,
                ch_names_list=val_ch_names_list,
                metrics=metrics, is_binary=(args.nb_classes == 1),
                used_ints=used_ints, args=data_args
            )

    print(f"\n{'=' * 20} 所有折和epoch评估完成 {'=' * 20}")
    print(f"预测结果和真实标签 (.npy文件) 已保存至: {args.output_dir}")


if __name__ == "__main__":
    args = get_eval_args()
    main(args)