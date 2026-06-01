#!/bin/bash

# 定义基础路径和噪声类型
BASE_DIR="/mnt/nfs/kaleid/ET-BERT"
NOISES=("drop_10" "duplicate" "mixed_extreme" "padding" "span_drop")

# 定义一个临时文件夹用于存放生成的原始 TSV
TMP_DATASET_DIR="${BASE_DIR}/tmp_noisy_dataset"
TMP_CACHE_DIR="${BASE_DIR}/tmp_noisy_cache"

echo "================ 开始批量提取噪声 PCAP 的序列特征 ================"

for noise in "${NOISES[@]}"; do
    echo "---------------------------------------------------"
    echo "[*] 正在处理噪声环境: $noise"
    
    # ================= 1. IoT 任务 (6分类) =================
    IN_IOT="${BASE_DIR}/dataset_noisy_${noise}/IoT"
    OUT_IOT="${BASE_DIR}/results_iot/test_${noise}.tsv"
    
    echo "  -> [1/3] 提取 IoT 任务..."
    PYTHONPATH=./ python3 data_process/main_fast.py \
        --pcap_path "$IN_IOT" \
        --dataset_dir "$TMP_DATASET_DIR" \
        --cache_dir "$TMP_CACHE_DIR" \
        --force_clear
    
    # 将临时目录中生成的 test_dataset.tsv 移动并重命名到结果目录
    if [ -f "${TMP_DATASET_DIR}/test_dataset.tsv" ]; then
        mv "${TMP_DATASET_DIR}/test_dataset.tsv" "$OUT_IOT"
        echo "    [+] IoT 测试集提取成功 -> $OUT_IOT"
    else
        echo "    [!] IoT 测试集生成失败！"
    fi

    # ================= 2. VPN-App 任务 (14分类) =================
    IN_VPN_APP="${BASE_DIR}/dataset_noisy_${noise}/ISCX-VPN-App"
    OUT_VPN_APP="${BASE_DIR}/results_vpn_app/test_${noise}.tsv"
    
    echo "  -> [2/3] 提取 VPN-App 任务..."
    PYTHONPATH=./ python3 data_process/main_fast.py \
        --pcap_path "$IN_VPN_APP" \
        --dataset_dir "$TMP_DATASET_DIR" \
        --cache_dir "$TMP_CACHE_DIR" \
        --force_clear
    
    if [ -f "${TMP_DATASET_DIR}/test_dataset.tsv" ]; then
        mv "${TMP_DATASET_DIR}/test_dataset.tsv" "$OUT_VPN_APP"
        echo "    [+] VPN-App 测试集提取成功 -> $OUT_VPN_APP"
    else
        echo "    [!] VPN-App 测试集生成失败！"
    fi

    # ================= 3. VPN-Service 任务 (6分类) =================
    IN_VPN_SERVICE="${BASE_DIR}/dataset_noisy_${noise}/ISCX-VPN-Service"
    OUT_VPN_SERVICE="${BASE_DIR}/results_vpn_service/test_${noise}.tsv"
    
    echo "  -> [3/3] 提取 VPN-Service 任务..."
    PYTHONPATH=./ python3 data_process/main_fast.py \
        --pcap_path "$IN_VPN_SERVICE" \
        --dataset_dir "$TMP_DATASET_DIR" \
        --cache_dir "$TMP_CACHE_DIR" \
        --force_clear
    
    if [ -f "${TMP_DATASET_DIR}/test_dataset.tsv" ]; then
        mv "${TMP_DATASET_DIR}/test_dataset.tsv" "$OUT_VPN_SERVICE"
        echo "    [+] VPN-Service 测试集提取成功 -> $OUT_VPN_SERVICE"
    else
        echo "    [!] VPN-Service 测试集生成失败！"
    fi

done

# 清理临时文件夹
rm -rf "$TMP_DATASET_DIR" "$TMP_CACHE_DIR"

echo "================ 所有噪声数据集 TSV 提取完毕！ ================"
echo "现在你可以直接去运行 ./run_noisy_inference.sh 了！"