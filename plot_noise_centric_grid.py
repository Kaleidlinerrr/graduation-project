#!/usr/bin/python3
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# 全局字体设置 (支持中文)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  

# 各任务类别名称定义
CLASSES = {
    'IoT': ['Benign', 'SYN-Flood', 'SlowLoris', 'HostDiscovery', 'OSScan', 'PortScan'],
    'VPN-App': ['aim', 'email', 'facebook', 'ftps', 'hangouts', 'icq', 'netflix', 'sftp', 'skype', 'spotify', 'torrent', 'vimeo', 'voipbuster', 'youtube'],
    'VPN-Service': ['Chat', 'Email', 'FileTransfer', 'P2P', 'Streaming', 'VOIP']
}

# 任务配置：(标题, 文件夹名, 内部键值)
TASKS = [
    ("IoT 分类任务 (6类)", "results_iot", "IoT"),
    ("VPN-App 分类任务 (14类)", "results_vpn_app", "VPN-App"),
    ("VPN-Service 分类任务 (6类)", "results_vpn_service", "VPN-Service")
]

# 噪声环境配置：(主标题, 文件后缀, 颜色映射)
CONDITIONS = [
    ("纯净基准环境 (Baseline)", "", "Blues"),
    ("随机丢包环境 (Drop 10%)", "_drop_10", "Blues"),
    ("连续丢包环境 (Span Drop)", "_span_drop", "Blues"),
    ("载荷变异环境 (Padding)", "_padding", "Blues"),
    ("报文重传环境 (Duplicate)", "_duplicate", "Blues"),
    ("混合极端恶劣环境 (Mixed Extreme)", "_mixed_extreme", "Reds")
]

def load_data(truth_path, pred_path):
    try:
        truth_df = pd.read_csv(truth_path, sep='\t', header=0)
        y_true = truth_df['label'].tolist()
        pred_df = pd.read_csv(pred_path, sep='\t', header=None)
        if "label" in str(pred_df.iloc[0].values) or pred_df.columns[0] == "label":
            pred_df = pd.read_csv(pred_path, sep='\t', header=0)
        y_pred = pred_df.iloc[:, 0].tolist()
        return y_true, y_pred
    except Exception as e:
        print(f"[!] 无法加载数据: {truth_path} 或 {pred_path}")
        return None, None

def plot_grid_for_noise(noise_title, suffix, cmap_color):
    print(f"[*] 正在绘制横向对比图: {noise_title} ...")
    
    # 动态分配宽度比例 (由于 VPN-App 有 14 类，我们需要让中间的子图更宽一些)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(28, 9), gridspec_kw={'width_ratios': [1, 1.8, 1]})
    
    for idx, (task_title, folder_name, task_key) in enumerate(TASKS):
        ax = axes[idx]
        class_names = CLASSES[task_key]
        num_classes = len(class_names)
        
        # 纯净基准与噪声的数据路径区分
        if suffix == "":
            truth_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/test_dataset.tsv"
            pred_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/prediction.tsv"
        else:
            truth_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/test{suffix}.tsv"
            pred_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/pred{suffix}.tsv"
            
        y_true, y_pred = load_data(truth_path, pred_path)
        if y_true is None:
            ax.set_title(f"{task_title} (Data Missing)")
            ax.axis('off')
            continue
            
        cm = confusion_matrix(y_true, y_pred)
        # 补齐维度保护
        full_cm = np.zeros((num_classes, num_classes))
        min_dim = min(cm.shape[0], num_classes)
        full_cm[:min_dim, :min_dim] = cm[:min_dim, :min_dim]
        
        cm_normalized = full_cm.astype('float') / (full_cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        annot_data = np.empty_like(full_cm).astype(str)
        for i in range(num_classes):
            for j in range(num_classes):
                count = int(full_cm[i, j])
                percent = cm_normalized[i, j] * 100
                annot_data[i, j] = f"{count}\n({percent:.1f}%)" if count > 0 else "0"

        annot_size = 9 if num_classes > 10 else 12
        
        sns.heatmap(cm_normalized, annot=annot_data, fmt='', cmap=cmap_color,
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, linewidths=0.5, linecolor='gray',
                    annot_kws={"size": annot_size}, ax=ax, cbar=(idx == 2)) # 仅在最右侧子图显示颜色条
        
        ax.set_title(task_title, fontsize=18, fontweight='bold', pad=15)
        if idx == 0:
            ax.set_ylabel('真实标签 (True Label)', fontsize=16, fontweight='bold')
        ax.set_xlabel('预测标签 (Predicted Label)', fontsize=16, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    # 主标题
    fig.suptitle(f'SPL-BERT特征模型在【{noise_title}】下的多任务性能对比', fontsize=26, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 根据噪声类型生成输出文件名
    safe_suffix = suffix if suffix != "" else "_baseline"
    out_file = f"/mnt/nfs/kaleid/ET-BERT/Noise_Centric_Grid{safe_suffix}.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] 保存成功: {out_file}\n")

if __name__ == "__main__":
    print("================ 开始生成以噪声为中心的 1x3 横向对比图 ================")
    for title, suffix, cmap in CONDITIONS:
        plot_grid_for_noise(title, suffix, cmap)
    print("================ 所有图表生成完毕！ ================")