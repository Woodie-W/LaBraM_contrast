CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8,9 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=9 ./run_vqnsp_training.py \
    --world_size 9 \
    --output_dir ./checkpoints/vqnsp/ \
    --log_dir ./log/vqnsp/ \
    --model vqnsp_encoder_base_decoder_3x200x12 \
    --codebook_n_emd 8192 \
    --codebook_emd_dim 64 \
    --quantize_kmeans_init \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.99 \
    --weight_decay 1e-4  \
    --warmup_epochs 10 \
    --epochs 100 \
    --save_ckpt_freq 20
    