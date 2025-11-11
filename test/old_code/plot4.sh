#!/bin/bash
dataset=finetune_wy_td1_2cls_b64_lr2_base
python plot_matrices.py \
    --results_dir ./evaluation_results/${dataset}/ \
    --nb_classes 2 \
    --class_names HC MDD \
    --epoch 59
    
python plot_matrices.py \
    --results_dir ./evaluation_results/${dataset}/ \
    --nb_classes 2 \
    --class_names HC MDD \
    --epoch 44       
       
python plot_matrices.py \
    --results_dir ./evaluation_results/${dataset}/ \
    --nb_classes 2 \
    --class_names HC MDD \
    --epoch 54  