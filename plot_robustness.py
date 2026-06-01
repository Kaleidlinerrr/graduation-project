#!/usr/bin/python3
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

# ================= 全局字体设置 (极其重要，防止中文乱码) =================
# Windows系统优先使用黑体(SimHei)，Mac系统优先使用Arial Unicode MS，Linux优先使用文泉驿等
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# =====================================================================

# 你的 6 细分类标签
CLASS_NAMES = ['Benign', 'SYN-Flood', 'SlowLoris', 'VPN-App', 'VPN-Service', 'Recon']

def load_data(truth_path, pred_path):
    """安全读取真值和预测值，兼容含有表头和无表头的情况"""
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

def plot_degradation_bar(results_dict):
    """绘制带衰减标注的柱状图"""
    envs = list(results_dict.keys())
    acc_scores = [results_dict[env]['acc'] * 100 for env in envs]
    baseline_acc = acc_scores[0]

    plt.figure(figsize=(11, 6))
    colors = ['#2ca02c'] + ['#d62728' if x < baseline_acc - 5 else '#1f77b4' for x in acc_scores[1:]]
    bars = plt.bar(envs, acc_scores, color=colors, width=0.6, alpha=0.85)

    plt.ylim(max(0, min(acc_scores) - 10), min(100, max(acc_scores) + 5))
    plt.ylabel('准确率 (%)', fontsize=13, fontweight='bold')
    plt.title('物理噪声环境下的经验鲁棒性评估', fontsize=16, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        if i == 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", 
                     ha='center', fontweight='bold', fontsize=11)
        else:
            drop = yval - baseline_acc
            label_text = f"{yval:.2f}%\n({drop:.2f}%)"
            color = 'red' if drop < -5 else 'black'
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, label_text, 
                     ha='center', color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Robustness_Degradation_Bar.png', dpi=300, bbox_inches='tight')
    print("[+] 准确率衰减图已保存: Robustness_Degradation_Bar.png")

def plot_cm_comparison(y_true_base, y_pred_base, y_true_extreme, y_pred_extreme):
    """绘制基准环境与极限恶劣环境的混淆矩阵对比"""
    cm_base = confusion_matrix(y_true_base, y_pred_base)
    cm_ext = confusion_matrix(y_true_extreme, y_pred_extreme)

    cm_base_pct = cm_base.astype('float') / cm_base.sum(axis=1)[:, np.newaxis]
    cm_ext_pct = cm_ext.astype('float') / cm_ext.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(cm_base_pct, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=1)
    axes[0].set_title('混淆矩阵：纯净基准环境', fontsize=15, fontweight='bold', pad=10)
    axes[0].set_ylabel('真实标签', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('预测标签', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(cm_ext_pct, annot=True, fmt='.2f', cmap='Reds', ax=axes[1], 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=1)
    axes[1].set_title('混淆矩阵：混合极端恶劣环境', fontsize=15, fontweight='bold', pad=10)
    axes[1].set_xlabel('预测标签', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('Confusion_Matrix_Comparison.png', dpi=300, bbox_inches='tight')
    print("[+] 混淆矩阵对比图已保存: Confusion_Matrix_Comparison.png")

if __name__ == "__main__":
    experiments = [
        ("基准环境\n(Baseline)", "datasets/test_dataset.tsv", "results_noisy/baseline_pred.tsv"),
        ("随机丢包\n(Drop 10%)", "datasets_noisy/drop_10/test_dataset.tsv", "results_noisy/drop_10_pred.tsv"),
        ("连续丢包\n(Span Drop)", "datasets_noisy/span_drop/test_dataset.tsv", "results_noisy/span_drop_pred.tsv"),
        ("载荷变异\n(Padding)", "datasets_noisy/padding/test_dataset.tsv", "results_noisy/padding_pred.tsv"),
        ("报文重传\n(Duplicate)", "datasets_noisy/duplicate/test_dataset.tsv", "results_noisy/duplicate_pred.tsv"),
        ("混合恶劣\n(Mixed Extreme)", "datasets_noisy/mixed_extreme/test_dataset.tsv", "results_noisy/mixed_extreme_pred.tsv")
    ]
    
    results = {}
    base_true, base_pred, ext_true, ext_pred = None, None, None, None

    print("================ 6分类模型 鲁棒性评估报告 ================")
    for name, truth_path, pred_path in experiments:
        # 修复 Python 3.8 的 f-string 报错: 先把 \n 替换掉存入变量
        clean_name = name.replace('\n', ' ')
        
        if not os.path.exists(truth_path) or not os.path.exists(pred_path):
            print(f"[!] 跳过 {clean_name}: 找不到对应文件，请确认推理脚本是否已运行完毕。")
            continue
            
        y_true, y_pred = load_data(truth_path, pred_path)
        if y_true is None: continue
            
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        results[name] = {'acc': acc, 'f1': f1}
        
        print(f"[{clean_name.ljust(20)}] 准确率: {acc:.4f} | 宏F1: {f1:.4f}")

        if "基准环境" in name:
            base_true, base_pred = y_true, y_pred
        elif "混合恶劣" in name:
            ext_true, ext_pred = y_true, y_pred
    print("==========================================================")

    if len(results) > 0:
        plot_degradation_bar(results)
    
    if base_true is not None and ext_true is not None:
        plot_cm_comparison(base_true, base_pred, ext_true, ext_pred)