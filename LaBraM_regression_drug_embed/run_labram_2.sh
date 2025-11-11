#!/bin/bash

dataset=labram_2_regression_drug_embed
for fold in {0..9}
do
    echo "Running training for fold ${fold}..."
       python  run_class_finetune_for_embedding.py \
        --fold ${fold} \
        --dataset ${dataset} \
        --gpu_id 7 \
        --output_dir ./checkpoints/${dataset}/fold${fold}/ \
        --log_dir ./log/${dataset}/fold${fold} \
        --model labram_base_patch200_200 \
        --finetune finetune_model/checkpoint-4.pth \
        --swanlab_project LaBraM_regression_drug_embed\
        --weight_decay 0.0005 \
        --batch_size 256 \
        --lr 2e-5 \
        --update_freq 2 \
        --warmup_epochs 7 \
        --epochs 120 \
        --layer_decay 0.85 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 2 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --disable_qkv_bias \
        --seed 998244353 

done

