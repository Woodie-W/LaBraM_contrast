# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 OMP_NUM_THREADS=32 torchrun --nnodes=1 --nproc_per_node=8 ./LEM/LaBraM/run_labram_pretraining.py \
#         --world_size 8 \
#         --output_dir ./LEM/LaBraM/checkpoints/labram_base \
#         --log_dir ./LEM/LaBraM/log/labram_base \
#         --model labram_base_patch200_1600_8k_vocab \
#         --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 \
#         --tokenizer_weight ./LEM/LaBraM/checkpoints/vqnsp.pth \
#         --batch_size 96 \
#         --lr 1e-4 \
#         --warmup_epochs 5 \
#         --clip_grad 0.2 \
#         --drop_path 0. \
#         --layer_scale_init_value 0.1 \
#         --opt_betas 0.9 0.98 \
#         --opt_eps 1e-8  \
#         --epochs 50 \
#         --save_ckpt_freq 5 \
#         --codebook_dim 64 \
#         --gradient_accumulation_steps 8
#         # --lr 5e-4 \
#         # --clip_grad 3.0 \
#         # --opt_betas 0.9 0.98 \

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 OMP_NUM_THREADS=32 torchrun --nnodes=1 --nproc_per_node=8 ./LEM/LaBraM/run_labram_pretraining.py \
        --output_dir ./LEM/LaBraM/checkpoints/labram_base \
        --log_dir ./LEM/LaBraM/log/labram_base \
        --model labram_base_patch200_1600_8k_vocab \
        --tokenizer_model vqnsp_encoder_base_decoder_3x200x12 \
        --tokenizer_weight ./LEM/LaBraM/checkpoints/vqnsp/checkpoint-99.pth \
        --batch_size 64 \
        --lr 5e-4 \
        --warmup_epochs 50 \
        --clip_grad 3.0 \
        --drop_path 0. \
        --layer_scale_init_value 0.1 \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 50 \
        --save_ckpt_freq 2 \
        --codebook_dim 64 \
        --gradient_accumulation_steps 1