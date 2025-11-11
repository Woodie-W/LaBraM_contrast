#!/bin/bash

# --- 配置参数 ---
# 数据集名称，用于构建路径
dataset_name="finetune_wy_1+2+3+4-2_2cls_b64_base"

# 定义要为这个数据集评估的epoch列表
# 只需要在这里修改，用空格隔开多个epoch
epochs_to_run="69"

# 评估脚本的核心参数
BASE_CHECKPOINT_DIR="./checkpoints/${dataset_name}/"
OUTPUT_DIR="./evaluation_results/${dataset_name}/"
MODEL_NAME="labram_base_patch200_200"
NUM_CLASSES=1
GPU_ID=3

# --- 主执行逻辑 ---
# 使用for循环，为列表中的每一个epoch单独运行一次评估脚本
echo "Starting evaluation for dataset: ${dataset_name}"

for epoch in ${epochs_to_run}
do
    echo "----------------------------------------------------"
    echo "--- Running evaluation for EPOCH: ${epoch} ---"
    echo "----------------------------------------------------"

    # 运行评估脚本，注意 --epochs 参数现在只接收一个值: ${epoch}
    python evaluate_folds.py \
        --base_checkpoint_dir ${BASE_CHECKPOINT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --model ${MODEL_NAME} \
        --nb_classes ${NUM_CLASSES} \
        --gpu_id ${GPU_ID} \
        --batch_size 64 \
        --epochs ${epoch} \
        --drop_path 0.1 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --disable_qkv_bias \
        --seed 12345
done

echo "*********************************************"
echo "All evaluations for ${dataset_name} are complete."
echo "*********************************************"

# --- (可选) 评估完成后，为每个epoch单独绘制混淆矩阵 ---
echo "Starting to plot confusion matrices..."

for epoch in ${epochs_to_run}
do
    echo "--- Plotting for EPOCH: ${epoch} ---"
    python plot_matrices.py \
        --results_dir ${OUTPUT_DIR} \
        --nb_classes ${NUM_CLASSES} \
        --epoch ${epoch}
done

echo "All plotting is complete."