#!/bin/bash

# 1. 定义一个你自己有权限且空间足够的临时目录
export MY_TEMP_DIR="/data/sxk/my_tmp_for_python"
mkdir -p $MY_TEMP_DIR
# 2. 设置环境变量，让所有 Python 子进程都使用这个新目录
export TMPDIR=$MY_TEMP_DIR

dataset=wy_1+2+3+4-2+4_2cls_b64
for fold in {2..3}
do
    echo "Running training for fold ${fold}..."
       python  1_2_3_4-2_4.py \
        --fold ${fold} \
        --dataset ${dataset} \
        --gpu_id 3 \
        --world_size 1 \
        --nb_classes 1 \
        --output_dir ./checkpoints/finetune_${dataset}_base/fold${fold}/ \
        --log_dir ./log/finetune_${dataset}_base/fold${fold} \
        --model labram_base_patch200_200 \
        --finetune finetune_model/checkpoint-4.pth \
        --weight_decay 0.05 \
        --batch_size 64 \
        --lr 2e-4 \
        --update_freq 2 \
        --warmup_epochs 3 \
        --epochs 60 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 5 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --disable_qkv_bias \
        --seed 12345    #添加 & 使得命令在后台运行，就不用等待fold1结束才会运行fold2了

done

