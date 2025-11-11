#!/bin/bash

dataset=wy_td1_2cls_b64_lr2
scenario=3
# 运行评估脚本
python run_class_eval.py \
    --checkpoint_base_dir ./checkpoints/finetune_${dataset}_base \
    --log_dir ./log/eval_${dataset}_s${scenario}/ \
    --gpu_id 0 \
    --nb_classes 1 \
    --model labram_base_patch200_200 \
    --batch_size 64 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --disable_qkv_bias \
    --scenario ${scenario} \
    --seed 12345