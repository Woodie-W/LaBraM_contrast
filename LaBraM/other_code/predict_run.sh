#!/bin/bash

# =================================================================
#  对10折交叉验证训练出的所有模型进行预测的脚本
# =================================================================

# --- 固定参数 (在这里修改一次即可) ---

# 1. 包含所有被试文件夹的、预处理好的数据目录
DATA_FOR_PREDICTION="/data/sxk/pretrain_set/EEG/all/26chs/"

# 2. 用于生成结果的TSV模板文件
TEMPLATE_FILE="./participants.tsv"

# 3. 包含所有折模型(fold0, fold1...)的根目录
#    注意：请仔细核对这个路径是否与您训练时使用的 `output_dir` 匹配
BASE_CHECKPOINT_DIR="/data/sxk/LaBraM/checkpoints/finetune_wy_2-2_b64_8_7_base"
# --- 循环执行预测 ---

# 循环变量 'fold' 将从 0 遍历到 9
for fold in {0..9}
do
    echo "================================================="
    echo "======          正在处理 Fold ${fold}          ======"
    echo "================================================="

    # 动态构建每一折的模型路径和输出文件名
    MODEL_CHECKPOINT="${BASE_CHECKPOINT_DIR}/fold${fold}/checkpoint-59.pth"
    OUTPUT_FILE="./result_fold${fold}.tsv"

    # 在运行前，检查模型文件是否存在
    if [ ! -f "${MODEL_CHECKPOINT}" ]; then
        echo "警告: 模型文件未找到: ${MODEL_CHECKPOINT}"
        echo "跳过 Fold ${fold}..."
        echo ""
        continue # 跳过当前循环，继续下一个
    fi

    echo "模型路径: ${MODEL_CHECKPOINT}"
    echo "数据路径: ${DATA_FOR_PREDICTION}"
    echo "输出文件: ${OUTPUT_FILE}"
    echo "-------------------------------------------------"

    # 运行您的 predict.py 脚本
    # 包含了所有与训练时匹配的模型结构参数
    python predict.py \
        --model_path "${MODEL_CHECKPOINT}" \
        --data_path "${DATA_FOR_PREDICTION}" \
        --template_file "${TEMPLATE_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --batch_size 64 \
        --device "cuda:0" \
        --nb_classes 1

    echo ""
    echo "====== Fold ${fold} 处理完成. 结果已保存. ======"
    echo ""

done

echo "================================================="
echo "======      所有 10 折均已处理完毕!      ======"
echo "================================================="