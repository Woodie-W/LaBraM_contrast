#!/bin/bash

# 设置CUDA设备(如果需要指定GPU,取消注释并设置)
# export CUDA_VISIBLE_DEVICES=0

# 创建日志目录
mkdir -p logs

# 循环运行10折、从2开始
for fold in {2..9}
do
    echo "Starting fold ${fold}"

    # 使用nohup后台运行
    nohup python3 run_class_finetune_for_embedding.py \
        --fold $fold \
        --batch_size 64 \
        --epochs 30 \
        > logs/labram_tdbrain_fold${fold}.log 2>&1 &

    # 记录进程ID
    echo "Fold ${fold} started with PID $!"

    # 等待几秒，避免同时启动太多进程
    sleep 5
done

echo "All folds submitted to background!"
echo "Use 'ps aux | grep run_class_finetune' to check running processes"
echo "Use 'tail -f logs/labram_tdbrain_fold*.log' to monitor logs"