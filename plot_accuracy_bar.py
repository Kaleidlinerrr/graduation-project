#!/usr/bin/python3
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import os

# ================= 全局字体与样式设置 =================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  
sns.set_theme(style="whitegrid", font='SimHei') # 使用学术网格背景

# 任务与文件夹映射
TASKS = [
    ("IoT (6分类)", "results_iot"),
    ("VPN-App (14分类)", "results_vpn_app"),
    ("VPN-Service (6分类)", "results_vpn_service")
]

# 噪声条件映射: (图表显示名称, 文件后缀)
CONDITIONS = [
    ("Baseline\n(纯净基准)", ""),
    ("Drop 10%\n(随机丢包)", "_drop_10"),
    ("Span Drop\n(连续丢包)", "_span_drop"),
    ("Padding\n(载荷变异)", "_padding"),
    ("Duplicate\n(报文重传)", "_duplicate"),
    ("Mixed Extreme\n(混合恶劣)", "_mixed_extreme")
]

def get_accuracy(truth_path, pred_path):
    """安全读取文件并计算准确率(返回百分比形式)"""
    try:
        truth_df = pd.read_csv(truth_path, sep='\t', header=0)
        y_true = truth_df['label'].tolist()
        
        pred_df = pd.read_csv(pred_path, sep='\t', header=None)
        if "label" in str(pred_df.iloc[0].values) or pred_df.columns[0] == "label":
            pred_df = pd.read_csv(pred_path, sep='\t', header=0)
        y_pred = pred_df.iloc[:, 0].tolist()
        
        return accuracy_score(y_true, y_pred) * 100
    except Exception as e:
        print(f"[!] 警告: 无法加载或计算 {truth_path} -> {e}")
        return 0.0

def main():
    print("================ 正在自动计算并绘制准确率柱状图 ================")
    
    # 1. 创建 1x3 的子图画布
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6.5))
    
    # 定义柱子的颜色 (Baseline用深色，Mixed Extreme用警示色，其他用渐变蓝)
    colors = ['#4A4A4A', '#5DADE2', '#3498DB', '#2874A6', '#1B4F72', '#E74C3C']
    
    for idx, (task_title, folder_name) in enumerate(TASKS):
        ax = axes[idx]
        accuracies = []
        labels = []
        
        # 2. 遍历该任务下的所有噪声条件，计算准确率
        for cond_name, suffix in CONDITIONS:
            labels.append(cond_name)
            if suffix == "":
                truth_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/test_dataset.tsv"
                pred_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/prediction.tsv"
            else:
                truth_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/test{suffix}.tsv"
                pred_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/pred{suffix}.tsv"
            
            acc = get_accuracy(truth_path, pred_path)
            accuracies.append(acc)
            print(f"[*] {task_title} - {cond_name.replace(chr(10), ' ')}: {acc:.2f}%")
            
        # 3. 绘制柱状图
        bars = ax.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1, alpha=0.9)
        
        # 4. 在每个柱子顶部添加具体数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0: # 只有大于0才显示
                ax.annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 向上偏移3个像素
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 5. 子图样式设置
        ax.set_title(task_title, fontsize=16, fontweight='bold', pad=15)
        ax.set_ylim(0, 110) # Y轴固定为0-110，留出顶部空间显示文字
        if idx == 0:
            ax.set_ylabel('模型准确率 Accuracy (%)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # 为 X 轴标签设置适当的倾斜以防止文字重叠
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # 添加一条水平基准虚线 (以Baseline的准确率为基准)，直观展示性能下降幅度
        baseline_acc = accuracies[0]
        if baseline_acc > 0:
            ax.axhline(y=baseline_acc, color='#4A4A4A', linestyle='--', alpha=0.5, zorder=0)

    # 整体标题与排版
    fig.suptitle('不同物理层噪声环境下各分类任务的准确率衰减对比', fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 6. 保存图片
    output_path = "/mnt/nfs/kaleid/ET-BERT/Accuracy_Comparison_BarChart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[+] 柱状图生成成功，已保存至: {output_path}")
    print("========================================================================")

if __name__ == "__main__":
    main()