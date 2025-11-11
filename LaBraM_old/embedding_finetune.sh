#!/bin/bash
for fold in {3..9}
do
    echo "Running training for fold ${fold}..."
    CUDA_VISIBLE_DEVICES=4,5,6,8 OMP_NUM_THREADS=64 torchrun --nnodes=1 --nproc_per_node=4 ./LEM_forlabram/LaBraM/run_class_finetune_for_embedding.py \
        --fold ${fold} \
        --world_size 4 \
        --output_dir ./LEM_forlabram/LaBraM/checkpoints/finetune_base_embedding_1112/fold${fold}/ \
        --log_dir ./LEM_forlabram/LaBraM/log/finetune_base_embedding_1112/fold${fold} \
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
        --disable_qkv_bias \
        --seed 0
done




