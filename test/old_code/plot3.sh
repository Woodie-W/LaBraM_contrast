#!/bin/bash
dataset=finetune_wy_1+2+3-4_2cls_b64_base
python plot_matrices.py \
    --results_dir ./evaluation_results/${dataset}/ \
    --nb_classes 2 \
    --class_names HC MDD \
    --epoch 59
    
python plot_matrices.py \
    --results_dir ./evaluation_results/${dataset}/ \
    --nb_classes 2 \
    --class_names HC MDD \
    --epoch 34   