#!/bin/bash
# 噪声后缀列表
NOISES=("drop_10" "duplicate" "mixed_extreme" "padding" "span_drop")
GPU="6"
BASE_DIR="/mnt/nfs/kaleid/ET-BERT"

echo "================ 开始批量执行抗噪鲁棒性推理 ================"

# 1. IoT 任务 (6分类)
for noise in "${NOISES[@]}"; do
    echo "[*] 正在执行 IoT 推理: 噪声类型 -> $noise"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=./ python3 inference/run_classifier_infer.py \
        --load_model_path ${BASE_DIR}/models/iot_task/classifier_baseline.bin \
        --vocab_path ${BASE_DIR}/models/fixed_length_vocab.txt \
        --config_path ${BASE_DIR}/bert_base_config.json \
        --test_path ${BASE_DIR}/results_iot/test_${noise}.tsv \
        --prediction_path ${BASE_DIR}/results_iot/pred_${noise}.tsv \
        --labels_num 6 --embedding word_pos_seg --encoder transformer --tokenizer space
done

# 2. VPN-App 任务 (14分类)
for noise in "${NOISES[@]}"; do
    echo "[*] 正在执行 VPN-App 推理: 噪声类型 -> $noise"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=./ python3 inference/run_classifier_infer.py \
        --load_model_path ${BASE_DIR}/models/vpn_app_task/classifier_baseline.bin \
        --vocab_path ${BASE_DIR}/models/fixed_length_vocab.txt \
        --config_path ${BASE_DIR}/bert_base_config.json \
        --test_path ${BASE_DIR}/results_vpn_app/test_${noise}.tsv \
        --prediction_path ${BASE_DIR}/results_vpn_app/pred_${noise}.tsv \
        --labels_num 14 --embedding word_pos_seg --encoder transformer --tokenizer space
done

# 3. VPN-Service 任务 (6分类)
for noise in "${NOISES[@]}"; do
    echo "[*] 正在执行 VPN-Service 推理: 噪声类型 -> $noise"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=./ python3 inference/run_classifier_infer.py \
        --load_model_path ${BASE_DIR}/models/vpn_service_task/classifier_baseline.bin \
        --vocab_path ${BASE_DIR}/models/fixed_length_vocab.txt \
        --config_path ${BASE_DIR}/bert_base_config.json \
        --test_path ${BASE_DIR}/results_vpn_service/test_${noise}.tsv \
        --prediction_path ${BASE_DIR}/results_vpn_service/pred_${noise}.tsv \
        --labels_num 6 --embedding word_pos_seg --encoder transformer --tokenizer space
done

echo "================ 所有推理任务执行完毕！ ================"