#!/bin/bash

# datasets=("Td_eyecloseandopen" "Parkinson_eyes_open_PROCESSED" "PREDICT-PD_LPC_Rest_2" "Parkinson" "PREDICT-PD_LPC_Rest")
datasets=("Td_eyecloseandopen")
for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,5,6,7,9 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=7 ./LEM_forlabram/LaBraM/run_class_finetuning_for_parkinson.py \
        --world_size 7 \
        --output_dir ./LEM_forlabram/LaBraM/checkpoints/finetune_${dataset}_base_1102/ \
        --log_dir ./LEM_forlabram/LaBraM/log/finetune_${dataset}_base_1102 \
        --model labram_base_patch200_200 \
        --finetune ./LEM_forlabram/LaBraM/checkpoints/labram_base_lossnan/checkpoint-4.pth \
        --weight_decay 0.05 \
        --batch_size 32 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 3 \
        --epochs 30 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --dataset ${dataset} \
        --save_ckpt_freq 5 \
        --dist_eval \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --disable_qkv_bias \
        --seed 0 \
         
done
