#!/bin/bash

# 创建结果存放目录
mkdir -p results_noisy

# 定义公共参数 (已更新为 6 分类配置)
MODEL_PATH="models/classifier_6class_baseline.bin"
VOCAB_PATH="models/fixed_length_vocab.txt"
CONFIG_PATH="bert_base_config.json"
LABELS_NUM=6 

echo "[*] 开始执行 6分类 鲁棒性压力测试推理流水线..."

# 1. 基准测试 (纯净测试集)
echo "[1/6] 正在推理: 纯净基准环境 (Baseline)..."
CUDA_VISIBLE_DEVICES="6" PYTHONPATH=./ python3 inference/run_classifier_infer.py \
    --load_model_path $MODEL_PATH --vocab_path $VOCAB_PATH --config_path $CONFIG_PATH \
    --test_path datasets/test_dataset.tsv \
    --prediction_path results_noisy/baseline_pred.tsv \
    --labels_num $LABELS_NUM --embedding word_pos_seg --encoder transformer --tokenizer space

# 2. 纯随机丢包 (Drop 10%)
echo "[2/6] 正在推理: 纯随机丢包 (Drop 10%)..."
CUDA_VISIBLE_DEVICES="6" PYTHONPATH=./ python3 inference/run_classifier_infer.py \
    --load_model_path $MODEL_PATH --vocab_path $VOCAB_PATH --config_path $CONFIG_PATH \
    --test_path datasets_noisy/drop_10/test_dataset.tsv \
    --prediction_path results_noisy/drop_10_pred.tsv \
    --labels_num $LABELS_NUM --embedding word_pos_seg --encoder transformer --tokenizer space

# 3. 突发连续丢包 (Span Drop)
echo "[3/6] 正在推理: 突发连续丢包 (Span Drop)..."
CUDA_VISIBLE_DEVICES="6" PYTHONPATH=./ python3 inference/run_classifier_infer.py \
    --load_model_path $MODEL_PATH --vocab_path $VOCAB_PATH --config_path $CONFIG_PATH \
    --test_path datasets_noisy/span_drop/test_dataset.tsv \
    --prediction_path results_noisy/span_drop_pred.tsv \
    --labels_num $LABELS_NUM --embedding word_pos_seg --encoder transformer --tokenizer space

# 4. 载荷长度变异 (Padding)
echo "[4/6] 正在推理: 载荷长度变异 (Padding)..."
CUDA_VISIBLE_DEVICES="6" PYTHONPATH=./ python3 inference/run_classifier_infer.py \
    --load_model_path $MODEL_PATH --vocab_path $VOCAB_PATH --config_path $CONFIG_PATH \
    --test_path datasets_noisy/padding/test_dataset.tsv \
    --prediction_path results_noisy/padding_pred.tsv \
    --labels_num $LABELS_NUM --embedding word_pos_seg --encoder transformer --tokenizer space

# 5. 报文重传 (Duplicate)
echo "[5/6] 正在推理: 报文重传 (Duplicate)..."
CUDA_VISIBLE_DEVICES="6" PYTHONPATH=./ python3 inference/run_classifier_infer.py \
    --load_model_path $MODEL_PATH --vocab_path $VOCAB_PATH --config_path $CONFIG_PATH \
    --test_path datasets_noisy/duplicate/test_dataset.tsv \
    --prediction_path results_noisy/duplicate_pred.tsv \
    --labels_num $LABELS_NUM --embedding word_pos_seg --encoder transformer --tokenizer space

# 6. 混合恶劣环境 (Mixed Extreme)
echo "[6/6] 正在推理: 混合恶劣环境 (Mixed Extreme)..."
CUDA_VISIBLE_DEVICES="6" PYTHONPATH=./ python3 inference/run_classifier_infer.py \
    --load_model_path $MODEL_PATH --vocab_path $VOCAB_PATH --config_path $CONFIG_PATH \
    --test_path datasets_noisy/mixed_extreme/test_dataset.tsv \
    --prediction_path results_noisy/mixed_extreme_pred.tsv \
    --labels_num $LABELS_NUM --embedding word_pos_seg --encoder transformer --tokenizer space

echo "[+] 全部推理完成！结果已保存在 results_noisy/ 目录中。"