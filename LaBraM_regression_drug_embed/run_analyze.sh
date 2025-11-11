#!/bin/bash

# 设置为 "当任何命令失败时，立即退出脚本"
set -e

DATASETS=(
    "labram_0_regression_drug_embed"
    "labram_1_regression_drug_embed"
    "labram_2_regression_drug_embed"
    "labram_3_regression_drug_embed"
    "labram_4_regression_drug_embed"
)

# 循环遍历上面定义的所有数据集
for dataset_name in "${DATASETS[@]}"; do
    echo "处理数据集: $dataset_name"
    python analyze_results.py --dataset "$dataset_name"
    echo "" 
done
