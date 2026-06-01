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

# 子图配置: (标题, 文件后缀标识)
# 对应 2x3 网格的 6 个位置
CONDITIONS = [
    ("纯净基准 (Baseline)", ""),
    ("随机丢包 (Drop 10%)", "_drop_10"),
    ("连续丢包 (Span Drop)", "_span_drop"),
    ("载荷变异 (Padding)", "_padding"),
    ("报文重传 (Duplicate)", "_duplicate"),
    ("混合极端 (Mixed Extreme)", "_mixed_extreme")
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

def plot_grid_for_task(task_key, folder_name):
    print(f"[*] 正在绘制 {task_key} 任务的鲁棒性对比图...")
    class_names = CLASSES[task_key]
    num_classes = len(class_names)
    
    # 动态调整画布大小 (14分类需要更大的画布)
    fig_size = (28, 16) if num_classes > 10 else (22, 13)
    annot_size = 8 if num_classes > 10 else 11
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=fig_size)
    axes = axes.flatten()
    
    for idx, (title, suffix) in enumerate(CONDITIONS):
        ax = axes[idx]
        
        # 纯净基准与噪声的数据路径区分
        if suffix == "":
            truth_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/test_dataset.tsv"
            pred_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/prediction.tsv"
        else:
            truth_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/test{suffix}.tsv"
            pred_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/pred{suffix}.tsv"
            
        y_true, y_pred = load_data(truth_path, pred_path)
        if y_true is None:
            ax.set_title(f"{title} (Data Missing)")
            ax.axis('off')
            continue
            
        cm = confusion_matrix(y_true, y_pred)
        # 补齐维度(防止某些噪声下某类完全未被预测出导致矩阵变小)
        full_cm = np.zeros((num_classes, num_classes))
        min_dim = min(cm.shape[0], num_classes)
        full_cm[:min_dim, :min_dim] = cm[:min_dim, :min_dim]
        
        cm_normalized = full_cm.astype('float') / (full_cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # 生成带百分比的文字注解
        annot_data = np.empty_like(full_cm).astype(str)
        for i in range(num_classes):
            for j in range(num_classes):
                count = int(full_cm[i, j])
                percent = cm_normalized[i, j] * 100
                annot_data[i, j] = f"{count}\n({percent:.1f}%)" if count > 0 else "0"

        # 混合极端使用红色系警示，其他使用蓝色系
        cmap = 'Reds' if 'Mixed' in title else 'Blues'
        
        sns.heatmap(cm_normalized, annot=annot_data, fmt='', cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    vmin=0, vmax=1, linewidths=0.5, linecolor='gray',
                    annot_kws={"size": annot_size}, ax=ax, cbar=(idx % 3 == 2)) # 只在最右侧显示颜色条
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
        if idx % 3 == 0:
            ax.set_ylabel('真实标签 (True Label)', fontsize=14, fontweight='bold')
        if idx >= 3:
            ax.set_xlabel('预测标签 (Predicted Label)', fontsize=14, fontweight='bold')
            
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # 主标题与排版调整
    fig.suptitle(f'{task_key} 分类任务在不同物理信道噪声下的鲁棒性对比', fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为主标题留出空间
    
    out_file = f"/mnt/nfs/kaleid/ET-BERT/Robustness_Grid_{task_key}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[+] 保存成功: {out_file}\n")

if __name__ == "__main__":
    tasks = [
        ('IoT', 'results_iot'),
        ('VPN-App', 'results_vpn_app'),
        ('VPN-Service', 'results_vpn_service')
    ]
    for task_name, folder in tasks:
        plot_grid_for_task(task_name, folder)