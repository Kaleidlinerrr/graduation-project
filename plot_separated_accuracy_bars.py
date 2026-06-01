#!/usr/bin/python3
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import accuracy_score
import os

# ================= ⚡ 核心修复：动态中文字体自适应定位器 ⚡ =================
def configure_chinese_font():
    # 预设常见的 Linux/Mac/Windows 健壮中文字体列表
    font_names = ['Noto Sans CJK SC', 'Source Han Sans CN', 'WenQuanYi Micro Hei', 
                  'SimHei', 'Microsoft YaHei', 'STHeiti', 'DejaVu Sans']
    
    # 动态扫描系统已安装的字体，过滤出所有可能支持中文的字体
    system_fonts = [f.name for f in fm.fontManager.ttflist 
                    if any(kw in f.name.lower() for kw in ['hei', 'cjk', 'han', 'sim', 'kai', 'yahei'])]
    
    # 将系统内检测到的中文字体置顶，后面拼接预设列表
    combined_fonts = system_fonts + font_names + ['sans-serif']
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = combined_fonts
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    if system_fonts:
        print(f"[+] 检测到系统可用中文字体: {system_fonts[0]}，已成功绑定。")
    else:
        print("[!] 警告: 系统中未扫描到标准中文字体，若出现方框，请参考提示安装字体包。")

# 激活字体配置
sns.set_theme(style="whitegrid") # 使用学术网格背景
configure_chinese_font()
# =========================================================================

# 任务与文件夹映射：(图表标题, 对应文件夹, 保存文件后缀)
TASKS = [
    ("IoT 分类任务 (6分类)", "results_iot", "IoT"),
    ("VPN-App 分类任务 (14分类)", "results_vpn_app", "VPN_App"),
    ("VPN-Service 分类任务 (6分类)", "results_vpn_service", "VPN_Service")
]

# 噪声条件映射: (中文显示名称, 文件后缀)
CONDITIONS = [
    ("纯净基准 (Baseline)", ""),
    ("随机丢包 10%", "_drop_10"),
    ("连续丢包 (Span Drop)", "_span_drop"),
    ("载荷变异 (Padding)", "_padding"),
    ("报文重传 (Duplicate)", "_duplicate"),
    ("混合极端恶劣", "_mixed_extreme")
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
    print("================ 正在分别生成 3 张独立的中文准确率柱状图 ================")
    
    # 定义柱子的颜色 (Baseline用深色，Mixed Extreme用警示色，其他用渐变蓝)
    colors = ['#4A4A4A', '#5DADE2', '#3498DB', '#2874A6', '#1B4F72', '#E74C3C']
    
    for task_title, folder_name, save_suffix in TASKS:
        accuracies = []
        labels = []
        
        # 1. 遍历收集该任务在不同噪声下的准确率
        for cond_name, file_suffix in CONDITIONS:
            labels.append(cond_name)
            if file_suffix == "":
                truth_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/test_dataset.tsv"
                pred_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/prediction.tsv"
            else:
                truth_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/test{file_suffix}.tsv"
                pred_path = f"/mnt/nfs/kaleid/ET-BERT/{folder_name}/pred{file_suffix}.tsv"
            
            acc = get_accuracy(truth_path, pred_path)
            accuracies.append(acc)
            print(f"[*] {task_title} - {cond_name}: {acc:.2f}%")
            
        # 2. 为当前任务创建独立画布
        plt.figure(figsize=(9.5, 6.5))
        
        # 3. 绘制柱状图
        bars = plt.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1, alpha=0.9, width=0.55)
        
        # 4. 在每个柱子顶部添加具体数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4),  # 向上偏移4个像素
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 5. 图表中文样式设置
        plt.title(f'不同噪声环境下的准确率衰减: {task_title}', fontsize=15, fontweight='bold', pad=15)
        plt.ylim(0, 115) # Y轴固定，留出顶部空间显示文字
        plt.ylabel('模型准确率 Accuracy (%)', fontsize=13, fontweight='bold')
        plt.xlabel('网络信道物理噪声类型', fontsize=13, fontweight='bold', labelpad=10)
        plt.xticks(rotation=25, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        
        # 6. 添加基准水平虚线
        baseline_acc = accuracies[0]
        if baseline_acc > 0:
            plt.axhline(y=baseline_acc, color='#4A4A4A', linestyle='--', alpha=0.6, zorder=0)

        plt.tight_layout()
        
        # 7. 保存当前图表并关闭画布
        output_path = f"/mnt/nfs/kaleid/ET-BERT/Accuracy_Bar_ZH_{save_suffix}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[+] 中文图表保存成功: {output_path}\n")

    print("================ 所有中文图表生成完毕！ ================")

if __name__ == "__main__":
    main()