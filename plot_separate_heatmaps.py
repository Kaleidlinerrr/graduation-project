#!/usr/bin/python3
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import os

# ================= 全局字体设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  
# ================================================

# 根据前期数据处理阶段的映射，定义各个任务的真实类名标签
CLASS_NAMES_IOT = ['Benign', 'SYN-Flood', 'SlowLoris', 'HostDiscovery', 'OSScan', 'PortScan']
CLASS_NAMES_VPN_APP = ['aim', 'email', 'facebook', 'ftps', 'hangouts', 'icq', 'netflix', 'sftp', 'skype', 'spotify', 'torrent', 'vimeo', 'voipbuster', 'youtube']
CLASS_NAMES_VPN_SERVICE = ['Chat', 'Email', 'FileTransfer', 'P2P', 'Streaming', 'VOIP']

def load_data(truth_path, pred_path):
    """安全读取真实标签和预测结果"""
    try:
        truth_df = pd.read_csv(truth_path, sep='\t', header=0)
        y_true = truth_df['label'].tolist()
        
        pred_df = pd.read_csv(pred_path, sep='\t', header=None)
        if "label" in str(pred_df.iloc[0].values) or pred_df.columns[0] == "label":
            pred_df = pd.read_csv(pred_path, sep='\t', header=0)
            
        y_pred = pred_df.iloc[:, 0].tolist()
        
        if len(y_true) != len(y_pred):
            print(f"[!] 警告: 行数不一致! 真值:{len(y_true)}, 预测:{len(y_pred)}")
            return None, None
            
        return y_true, y_pred
    except Exception as e:
        print(f"[!] 无法加载数据 {truth_path}: {e}")
        return None, None

def plot_separate_heatmaps():
    # 任务配置列表：(图表标题, 真实标签路径, 预测标签路径, 保存后缀, 类名列表)
    experiments = [
        ("IoT 纯净基准测试 (6分类)", "/mnt/nfs/kaleid/ET-BERT/results_iot/test_dataset.tsv", "/mnt/nfs/kaleid/ET-BERT/results_iot/prediction.tsv", "IoT_Baseline", CLASS_NAMES_IOT),
        ("VPN-App 纯净基准测试 (14分类)", "/mnt/nfs/kaleid/ET-BERT/results_vpn_app/test_dataset.tsv", "/mnt/nfs/kaleid/ET-BERT/results_vpn_app/prediction.tsv", "VPN_App_Baseline", CLASS_NAMES_VPN_APP),
        ("VPN-Service 纯净基准测试 (7分类)", "/mnt/nfs/kaleid/ET-BERT/results_vpn_service/test_dataset.tsv", "/mnt/nfs/kaleid/ET-BERT/results_vpn_service/prediction.tsv", "VPN_Service_Baseline", CLASS_NAMES_VPN_SERVICE)
    ]

    print("================ 正在生成下游分类任务独立热力图 ================")
    
    for name, truth_path, pred_path, file_suffix, class_names in experiments:
        if not os.path.exists(truth_path) or not os.path.exists(pred_path):
            print(f"[!] 跳过 {name}: 找不到对应文件，请确保推理指令已执行完成。")
            continue

        y_true, y_pred = load_data(truth_path, pred_path)
        if y_true is None: continue

        # 打印一下该任务的核心指标
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        print(f"\n[*] {name} 指标: Acc: {acc:.4f} | Pre: {p:.4f} | Rec: {r:.4f} | F1: {f1:.4f}")

        cm = confusion_matrix(y_true, y_pred)
        
        # 保护机制：如果预测类别数小于定义的类别数（某些类没被预测出），截断或扩展维度
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) # 避免除零

        annot_data = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                count = cm[i, j]
                percent = cm_normalized[i, j] * 100
                annot_data[i, j] = f"{count}\n({percent:.1f}%)" if count > 0 else "0"

        # 根据分类数量动态调整画布大小 (14分类需要更大的图)
        if len(class_names) > 10:
            plt.figure(figsize=(14, 11))
            annot_size = 8
        else:
            plt.figure(figsize=(9, 7.5))
            annot_size = 11
        
        # 统一使用学术蓝色
        cmap = 'Blues'

        # 绘制热力图
        sns.heatmap(cm_normalized, annot=annot_data, fmt='', cmap=cmap,
                    xticklabels=class_names[:ncols], yticklabels=class_names[:nrows],
                    vmin=0, vmax=1, linewidths=0.5, linecolor='gray',
                    annot_kws={"size": annot_size, "weight": "bold"})

        plt.title(f'{name}', fontsize=16, fontweight='bold', pad=15)
        plt.ylabel('真实标签 (True Label)', fontsize=14, fontweight='bold')
        plt.xlabel('预测标签 (Predicted Label)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        plt.subplots_adjust(bottom=0.2)

        # 保存图片
        output_filename = f'/mnt/nfs/kaleid/ET-BERT/Heatmap_{file_suffix}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close() 
        print(f"[+] 成功绘制并保存: {output_filename}")

    print("\n================ 全部图表生成完毕！ ================")

if __name__ == "__main__":
    plot_separate_heatmaps()