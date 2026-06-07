# SPL-BERT: 基于预训练模型的加密流量分类与鲁棒性提升

## 📖 项目简介
本项目为本科毕业设计代码仓库，主要基于经典的加密流量预训练模型 [ET-BERT] 进行二次开发与重构，提出了一种新的加密流量分类模型 —— **SPL-BERT (Signed Packet-Length BERT)**。

针对传统预训练模型在真实网络物理链路波动（如丢包、重传、乱序等网络噪声）中表现出的特征失真和决策脆弱等问题，本项目在词汇表构建、数据增强机制以及鲁棒性评估体系三个核心维度进行了修改。

## 🚀 与原始 ET-BERT 的核心差异及修改原因

### 1. 特征输入重构：从原始字节 (Raw Bytes) 转为双向包长序列
* **原始 ET-BERT**：高度依赖应用层载荷的前 N 个原始字节作为特征输入。
* **SPL-BERT 修改**：剥离应用层加密载荷，提取基于会话重组的**有符号双向包长序列**（正数表示客户端发送，负数表示服务端响应）。
* **修改原因**：包长序列特征独立于有效载荷内容，天然具备高抗扰动性，还能直接反映网络交互行为的本质和长程通信规律。

### 2. 词汇表重构：基于 MTU 物理约束的确定性词汇表
* **原始 ET-BERT**：使用 BPE 算法在字节流上动态生成词汇表。
* **SPL-BERT 修改**：构建了映射范围严格锁定在 `[-1600, 1600]` 的固定整数词表。
* **修改原因**：由于输入特征变为了包含负数的十进制包长，原有的 `0~255` 字节词表完全失效。基于标准以太网最大传输单元（MTU）的物理边界约束，将包长数值进行一对一确定性硬映射，大幅降低了模型嵌入层参数规模。

### 3. 数据增强：新增网络传输噪声增强模块 (PCAP Augmenter)
* **原始 ET-BERT**：在理想的纯净数据集上进行模型的训练与测试。
* **SPL-BERT 修改**：开发了独立的数据增强模块，模拟四类网络噪声：随机丢包（Drop）、突发连续拥塞（Span Mask）、载荷变异（Padding）以及报文重传（Duplication）。
* **修改原因**：通过流量特征掩码（TFM）策略引入人工扰动，迫使模型在自监督预训练阶段学习从受干扰的分布中复原通信行为的本质特征，从而显著增强模型在真实非理想信道下的泛化能力与抗干扰鲁棒性。

### 4. 模型评估：引入 PA-curve 综合鲁棒性量化体系
* **原始 ET-BERT**：采用宏观分类准确率（Accuracy / F1-score）作为评价标准。
* **SPL-BERT 修改**：设计了基于蒙特卡洛采样与概率-准确率曲线（PA-curve）及积分面积（PA-area）的综合评估体系。
* **修改原因**：网络噪声具有高度的随机性，传统的单一准确率指标极易掩盖模型在局部扰动下决策边界的脆弱性。PA-curve 能够准确估计测试样本的识别概率差异（Δp），并将多维度稳定性转化为综合标量指标（PA-area），为实际部署提供严谨的理论支撑。

---

## 🛠️ 项目完整运行指令示例 (Linux Terminal Pipeline)

本项目提供了一条端到端的完整流水线。请在 Linux 终端中按照以下步骤顺序执行：

### 1. 数据增强阶段 (Data Augmentation)
使用 `pcap_augmenter.py` 或提供的 shell 脚本，向原始 PCAP 文件注入多维度网络噪声。

```bash
# 示例 1: 执行单一维度网络噪声注入 (例如：10% 随机丢包)
python pcap_augmenter.py \
    --input_dir ./datasets/raw_pcap \
    --output_dir ./datasets/augmented_pcap/drop_10 \
    --drop_prob 0.1

# 示例 2: 执行复合极端网络噪声注入 (Mixed Extreme)
python pcap_augmenter.py \
    --input_dir ./datasets/raw_pcap \
    --output_dir ./datasets/augmented_pcap/mixed_extreme \
    --drop_prob 0.1 \
    --span_drop_prob 0.05 \
    --pad_prob 0.15 \
    --dup_prob 0.05

```

### 2. 特征提取与词表构建 (Preprocessing & Vocab)

剥离加密载荷，提取符号化包长序列，并生成固定词汇表。

```bash
# 提取流级别的双向有符号包长序列，生成训练文本语料
python data_process/dataset_generation_raw.py \
    --input_dir ./datasets/augmented_pcap/mixed_extreme \
    --output_dir ./datasets/corpus/pretrain_corpus_mixed.txt \
    --seq_len 512

# 构建基于 MTU 约束的确定性长度词汇表 [-1600, 1600]
python vocab_process/build_fixed_vocab.py \
    --output_file ./models/fixed_length_vocab.txt

```

### 3. 基座模型预训练 (Pre-training)

利用大规模无标签包长序列语料，执行掩码语言模型（MLM）自监督预训练。

```bash
python pre-training/pretrain.py \
    --dataset_path ./datasets/corpus/pretrain_corpus_mixed.txt \
    --vocab_path ./models/fixed_length_vocab.txt \
    --output_model_path ./models/spl_bert_base.bin \
    --config_path ./models/bert_base_config.json \
    --target mlm \
    --batch_size 32 \
    --total_steps 500000 \
    --save_checkpoint_steps 50000

```

### 4. 下游微调与分类 (Fine-Tuning & Classification)

加载预训练权重，在特定任务（如 IoT 6分类、VPN-App 14分类）上执行有监督微调。

```bash
# 以 IoT 6分类任务为例进行微调
python fine-tuning/run_classifier.py \
    --pretrained_model_path ./models/spl_bert_base.bin \
    --vocab_path ./models/fixed_length_vocab.txt \
    --config_path ./models/bert_base_config.json \
    --train_path ./datasets/IoT_6class/train.tsv \
    --dev_path ./datasets/IoT_6class/dev.tsv \
    --test_path ./datasets/IoT_6class/test_baseline.tsv \
    --epochs 5 \
    --batch_size 64 \
    --output_model_path ./models/finetuned/spl_bert_iot_6class.bin

```

### 5. 鲁棒性量化评估 (Robustness Assessment)

通过蒙特卡洛采样生成识别概率差异 (Δp)，计算 PA-Curve 与 PA-Area。

```bash
# Step 1: 蒙特卡洛采样 (N=500)，在带噪测试集上生成预测分布
python 1_generate_mc_data.py \
    --model_path ./models/finetuned/spl_bert_iot_6class.bin \
    --test_data ./datasets/IoT_6class/test_baseline.tsv \
    --num_samples 500 \
    --noise_type mixed_extreme \
    --output_mc_results ./results/mc_inference_iot.pkl

# Step 2: 基于 Clopper-Pearson 估计，计算概率边界及 PA-area
python 2_calc_pa_curve.py \
    --mc_results_path ./results/mc_inference_iot.pkl \
    --confidence_level 0.999 \
    --output_pa_metrics ./results/pa_metrics_iot.json
```

### 6. 数据可视化 (Visualization)

基于生成的评估指标文件，一键渲染论文所需的各项数据图表。

```bash
# 绘制特定噪声下的混淆矩阵热力图
python plot_separate_heatmaps.py \
    --metrics_path ./results/pa_metrics_iot.json \
    --output_img ./images/confusion_matrix_iot.png

# 绘制多维度噪声下的准确率衰减横向对比柱状图
python plot_accuracy_bar.py \
    --results_dir ./results/ \
    --task_name IoT_6class \
    --output_img ./images/accuracy_bar_iot.png

# 绘制最终的 PA-curve 概率-准确率曲线图
python plot_robustness_grid.py \
    --metrics_path ./results/pa_metrics_iot.json \
    --output_img ./images/pa_curve_iot.png

```

## 📝 核心修改目录结构简要说明

* `pcap_augmenter.py`: 核心网络噪声生成引擎。
* `data_process/`: 包含流重组与双向包长序列提取代码。
* `vocab_process/`: 确定性 MTU 词汇表生成脚本。
* `pre-training/`: 掩码语言模型（MLM）自监督预训练代码。
* `fine-tuning/`: 针对各类下游应用分类任务的有监督微调代码。
* `1_generate_mc_data.py` & `2_calc_pa_curve.py`: PA-curve 鲁棒性量化核心代码。
* `plot_*.py`: 系列评估结果可视化脚本。

```
