#!/bin/bash

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 ./LEM_forlabram/LaBraM/run_class_finetuning.py \
        --world_size 8 \
        --output_dir ./LEM_forlabram/LaBraM/checkpoints/finetune_TUHEEG_PROCESSED_base/ \
        --log_dir ./LEM_forlabram/LaBraM/log/finetune_TUHEEG_PROCESSED_base \
        --model labram_base_patch200_200 \
        --finetune ./LEM_forlabram/LaBraM/checkpoints/labram_base_lossnan/checkpoint-4.pth \
        --weight_decay 0.05 \
        --batch_size 128 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 3 \
        --epochs 30 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --dist_eval \
        --save_ckpt_freq 5 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUHEEG_PROCESSED \
        --disable_qkv_bias \
        --seed 0 \