#!/bin/bash

dataset=wy_td1_2cls_b64_lr2
for fold in {0..9}
do
    echo "Running training for fold ${fold}..."
       python  run_class_finetune_for_embedding.py \
        --fold ${fold} \
        --dataset ${dataset} \
        --gpu_id 4 \
        --world_size 1 \
        --nb_classes 1 \
        --output_dir ./checkpoints/finetune_${dataset}_base/fold${fold}/ \
        --log_dir ./log/finetune_${dataset}_base/fold${fold} \
        --model labram_base_patch200_200 \
        --finetune finetune_model/checkpoint-4.pth \
        --weight_decay 0.05 \
        --batch_size 64 \
        --lr 2.5e-4 \
        --update_freq 2 \
        --warmup_epochs 3 \
        --epochs 60 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 2 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --disable_qkv_bias \
        --td \
        --seed 12345    #添加 & 使得命令在后台运行，就不用等待fold1结束才会运行fold2了

done

