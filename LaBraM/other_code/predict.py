# predict.py (已修正参数兼容性问题)

import argparse
import os
import torch
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from pathlib import Path
from einops import rearrange
from torch.utils.data import Dataset, DataLoader

# 从您的项目中导入必要的模块和函数
from run_class_finetune_for_embedding import get_models
import utils
from utils import get_input_chans


# =============================================================================
#  专门为预测任务创建的数据集类 (无需修改)
# =============================================================================
class PredictionDataset(Dataset):
    def __init__(self, subject_folders):
        self.file_paths = []
        self.id2participant_id = {}
        self.ch_names = None

        for subj_path in subject_folders:
            if not os.path.isdir(subj_path):
                continue
            try:
                numeric_id = int(os.path.basename(subj_path))
                info_file_path = os.path.join(subj_path, f"{numeric_id}_info.pkl")
                if os.path.exists(info_file_path):
                    with open(info_file_path, 'rb') as f:
                        info = pickle.load(f)
                        self.id2participant_id[numeric_id] = info['subject_id_dateset']
                else:
                    print(f"警告: 在 {subj_path} 中找不到 info.pkl 文件，跳过。")
                    continue
                for file in os.listdir(subj_path):
                    if file.endswith('.pkl') and 'data' in file:
                        self.file_paths.append(os.path.join(subj_path, file))
            except Exception as e:
                print(f"处理文件夹 {subj_path} 时出错: {e}")

        if not subject_folders:
            raise ValueError("错误: subject_folders列表为空,无法确定父目录。")
        parent_dir = os.path.dirname(subject_folders[0])
        channel_file_path = os.path.join(parent_dir, 'channel_name.pkl')
        if os.path.exists(channel_file_path):
            with open(channel_file_path, 'rb') as f:
                self.ch_names = pickle.load(f)
        else:
            raise FileNotFoundError(f"错误: 无法在 {parent_dir} 中找到 'channel_name.pkl'。")
        print(f"成功找到 {len(self.file_paths)} 个数据切片，对应 {len(self.id2participant_id)} 个独立被试。")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        with open(file_path, 'rb') as f:
            eeg_data = pickle.load(f)
        numeric_id = int(os.path.basename(os.path.dirname(file_path)))
        participant_id_str = self.id2participant_id[numeric_id]
        return torch.tensor(eeg_data, dtype=torch.float32), participant_id_str

    def get_ch_names(self):
        return self.ch_names


def custom_prediction_collate(batch):
    eeg_data = torch.stack([item[0] for item in batch], dim=0)
    participant_ids = [item[1] for item in batch]
    return eeg_data, participant_ids


# =============================================================================
#  核心修正: 一个与 get_models 兼容的参数解析器
# =============================================================================
def get_args():
    parser = argparse.ArgumentParser("LaBraM Inference Script", add_help=False)

    # --- 预测专用参数 ---
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained model checkpoint')
    parser.add_argument('--data_path', required=True, type=str, help='Path to the directory containing subject folders')
    parser.add_argument('--template_file', required=True, type=str, help='Path to the participant template TSV file')
    parser.add_argument('--output_file', default='./submission.tsv', type=str,
                        help='Path to save the final submission file')

    # --- 与 get_models 兼容的模型架构参数 (从训练脚本复制而来) ---
    parser.add_argument('--model', default='labram_base_patch200_200', type=str)
    parser.add_argument('--input_size', default=200, type=int)
    parser.add_argument('--nb_classes', default=1, type=int)

    # QKV Bias
    parser.add_argument('--qkv_bias', action='store_true')
    parser.add_argument('--disable_qkv_bias', action='store_false', dest='qkv_bias')
    parser.set_defaults(qkv_bias=False)  # 对应训练时的 --disable_qkv_bias

    # Relative Position Bias
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)  # 对应训练时的 --disable_rel_pos_bias

    # Absolute Position Embedding
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)  # 对应训练时的 --abs_pos_emb

    # Layer Scale
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float)

    # Dropout and DropPath
    parser.add_argument('--drop', type=float, default=0.0, help="Dropout rate")
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help="Attention dropout rate")
    parser.add_argument('--drop_path', type=float, default=0.1, help="Drop path rate")

    # Pooling and Head
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--init_scale', default=0.001, type=float)

    # --- 运行参数 ---
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', help='e.g., cuda:0')

    return parser.parse_args()


def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. 加载模型 (现在可以安全地调用get_models)
    print("Building model...")
    model = get_models(args)  # 使用导入的函数

    checkpoint = torch.load(args.model_path, map_location='cpu')
    print(f"Loading checkpoint from: {args.model_path}")
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 2. 准备数据集
    subject_folders = [os.path.join(args.data_path, d) for d in os.listdir(args.data_path) if
                       os.path.isdir(os.path.join(args.data_path, d))]
    if not subject_folders:
        print(f"错误: 在 {args.data_path} 目录中没有找到任何被试子文件夹。")
        return

    dataset_predict = PredictionDataset(subject_folders=subject_folders)
    if len(dataset_predict) == 0:
        print(f"错误: 数据集为空。")
        return

    data_loader = DataLoader(dataset_predict, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                             collate_fn=custom_prediction_collate)
    ch_names = dataset_predict.get_ch_names()
    input_chans = get_input_chans(ch_names)

    # 3. 执行推理并收集所有切片的预测结果
    print("Starting inference...")
    all_predictions = {}
    with torch.no_grad():
        for eeg_data, participant_ids in data_loader:
            eeg_data = eeg_data.to(device) / 100.0
            eeg_data = rearrange(eeg_data, 'B N (A T) -> B N A T', T=200)
            outputs = model(eeg_data, input_chans=input_chans)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().flatten()
            for pid, p in zip(participant_ids, preds):
                if pid not in all_predictions:
                    all_predictions[pid] = []
                all_predictions[pid].append(p)

    # 4. 对每个被试的所有切片预测进行投票
    print("Aggregating results by majority vote...")
    final_subject_labels = {pid: Counter(preds).most_common(1)[0][0] for pid, preds in all_predictions.items()}
    print(f"Aggregated predictions for {len(final_subject_labels)} subjects.")

    # 5. 读取模板，填充结果并保存
    print(f"Loading template file from {args.template_file}")
    submission_df = pd.read_csv(args.template_file, sep='\t')
    int_to_label = {0: 'HC', 1: 'MDD'}
    predicted_labels_map = {pid: int_to_label.get(label) for pid, label in final_subject_labels.items()}

    submission_df['diagnosis'] = submission_df['participant_id'].map(predicted_labels_map)
    submission_df['diagnosis'].fillna('n/a', inplace=True)

    submission_df.to_csv(args.output_file, sep='\t', index=False)
    print(f"Submission file saved to {args.output_file}")
    print("\nPreview of the output file:\n", submission_df.head())


if __name__ == '__main__':
    args = get_args()
    main(args)