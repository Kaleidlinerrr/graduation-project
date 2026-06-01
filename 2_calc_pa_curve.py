#!/usr/bin/python3
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
from collections import Counter
from scipy.stats import beta

# ================= ⚡ 中文字体自适应定位器 ⚡ =================
def configure_chinese_font():
    system_fonts = [f.name for f in fm.fontManager.ttflist 
                    if any(kw in f.name.lower() for kw in ['hei', 'cjk', 'han', 'sim', 'kai', 'yahei', 'wqy'])]
    combined_fonts = system_fonts + ['sans-serif']
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = combined_fonts
    plt.rcParams['axes.unicode_minus'] = False 
    
    if not system_fonts:
        print("[!] 警告: 系统中未扫描到标准中文字体！")
configure_chinese_font()
# =========================================================================

def calc_pa_metrics(stabilities, thresholds):
    """
    根据论文公式计算不同阈值下的样本占比，以及曲线下方的梯形积分面积
    """
    stabilities = np.array(stabilities)
    proportions_percentage = []
    proportions_normalized = []
    
    # 统计满足 \Delta p >= x 且分类正确的样本比例
    for t in thresholds:
        prop = np.mean(stabilities >= t)
        proportions_percentage.append(prop * 100.0)
        proportions_normalized.append(prop)
        
    # 严格按照论文 Eq (4) 计算 PA-Area (梯形积分)
    pa_area = 0.0
    for i in range(len(thresholds) - 1):
        dx = thresholds[i+1] - thresholds[i]
        dy = proportions_normalized[i+1] + proportions_normalized[i]
        pa_area += (dx * dy) / 2.0
        
    return proportions_percentage, pa_area

def plot_real_pa_curve(thresholds, proportions, pa_area, task_name):
    plt.figure(figsize=(10, 6.5))
    plt.plot(thresholds, proportions, color='#1f77b4', linewidth=3.0, 
             label=f'PA-Curve (PA-Area = {pa_area:.4f})')
    plt.fill_between(thresholds, proportions, color='#1f77b4', alpha=0.15)
    
    plt.title(f'基于蒙特卡洛采样的 PA-Curve (严格概率区间分布)\n任务: {task_name}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('正确决策稳定性阈值 $\Delta \hat{p}$ (Confidence Probability Diff)', fontsize=14, fontweight='bold')
    plt.ylabel('样本占比 (%) (Proportion of Samples $\geq \Delta \hat{p}$)', fontsize=14, fontweight='bold')
    
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', fontsize=14, framealpha=0.9, edgecolor='gray')
    
    explanation = "注：横坐标表示基于Clopper-Pearson区间估计的Top-1与Top-2类别置信概率差下界；\n纵坐标为准确识别且概率差 $\geq \Delta \hat{p}$ 的样本百分比。PA-Area 严格量化了鲁棒性。"
    plt.figtext(0.5, -0.06, explanation, ha="center", fontsize=11, color='dimgrey',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.5'))

    safe_task_name = task_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    output_filename = f'Real_PA_Curve_{safe_task_name}.png'
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[+] 真正的学术级 PA-Curve (严谨复现版) 中文图表已保存为: {output_filename}")
    print(f"[+] 计算所得 PA-Area (鲁棒性量化核心指标): {pa_area:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="模型推理后的预测结果 TSV")
    parser.add_argument("--truth", required=True, help="第一步生成的真值表 CSV")
    parser.add_argument("--task_name", type=str, required=True, help="任务名称，用于图表标题和文件名")
    parser.add_argument("--alpha", type=float, default=0.001, help="置信区间显著性水平 (论文默认为0.001)")
    args = parser.parse_args()

    print(f"[*] 正在加载 {args.task_name} 的大批量推理数据...")
    truth_df = pd.read_csv(args.truth)
    actual_n = len(truth_df)
    print(f"[*] 自动推测出该任务的真实独立样本总数为: {actual_n}")
    
    pred_df = pd.read_csv(args.pred, sep='\t', header=None)
    if "label" in str(pred_df.iloc[0].values) or pred_df.columns[0] == "label":
        pred_df = pd.read_csv(args.pred, sep='\t', header=0)
    
    preds = pred_df.iloc[:, 0].tolist()
    N = len(preds) // actual_n
    
    if len(preds) % actual_n != 0:
        print(f"[!] 严重警告: 预测结果总数 ({len(preds)}) 依然无法被真实样本数 ({actual_n}) 整除！请检查数据完整性。")
        return

    print(f"[*] 成功识别配置: 真实样本数 n={actual_n}, 蒙特卡洛变异次数 N={N}")

    delta_p_list = []
    print("[*] 正在基于 Clopper-Pearson 方法计算每个样本的概率差下界 (Delta p)...")
    
    for i in range(actual_n):
        true_label = truth_df.iloc[i]['true_label']
        sample_preds = preds[i*N : (i+1)*N]
        
        # 获取最高频(A)和次高频(B)的预测结果
        counter = Counter(sample_preds)
        most_common = counter.most_common(2)
        
        c_A, n_A = most_common[0]
        if len(most_common) > 1:
            c_B, n_B = most_common[1]
        else:
            c_B, n_B = None, 0
            
        # 论文 Eq(3) 核心约束: 如果最高频预测类别不是真实标签，直接判定为 0 贡献
        if c_A != true_label:
            delta_p_list.append(-1.0) # 设置为负数，使得在任何 >= x (x \in [0,1]) 的判定中均为 False
            continue
            
        # Clopper-Pearson 置信区间计算
        # 计算 c_A 概率的下界 \underline{p_A}
        p_A_lower = beta.ppf(args.alpha / 2, n_A, N - n_A + 1) if n_A > 0 else 0.0
        
        # 计算 c_B 概率的上界 \overline{p_B}
        p_B_upper = beta.ppf(1 - args.alpha / 2, n_B + 1, N - n_B) if n_B < N else 1.0
        
        # 计算概率差
        delta_p = p_A_lower - p_B_upper
        delta_p_list.append(delta_p)

    # 论文中曲线的 x 轴是 0 到 1
    thresholds = np.linspace(0.0, 1.0, 101)
    
    # 统计并计算积分
    proportions, pa_area = calc_pa_metrics(delta_p_list, thresholds)

    plot_real_pa_curve(thresholds, proportions, pa_area, args.task_name)

if __name__ == "__main__":
    main()