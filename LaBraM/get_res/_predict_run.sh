# _predict_run.sh (您的脚本，可直接使用)

fold=3
# 1. 选择你训练好的模型路径
MODEL_CHECKPOINT="/data/sxk/LaBraM/checkpoints/finetune_wy_1+2+3+4-2_2cls_b64_base/fold${fold}/checkpoint-79.pth"

# 2. 指定由 split_data.py 生成的、待预测的数据所在的目录
#    路径指向包含被试文件夹(如'28304', '28310'等)的目录，这是正确的
DATA_FOR_PREDICTION="/data/sxk/pretrain_set/EEG/all/26chs/"

# 3. 指定您提供的 participants.tsv 模板文件
TEMPLATE_FILE="./participants.tsv"

# 4. 指定最终输出文件的名称
OUTPUT_FILE="./result_${fold}.tsv"

# 运行脚本
python predict.py \
    --model_path "${MODEL_CHECKPOINT}" \
    --data_path "${DATA_FOR_PREDICTION}" \
    --template_file "${TEMPLATE_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --batch_size 64 \
    --device "cuda:0"