#!/usr/bin/python3
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
import os

def random_loss_augment(sequence, drop_rate):
    """模拟信道极其恶劣时的随机丢包现象"""
    tokens = sequence.strip().split()
    # 根据 drop_rate 随机丢弃 token，但保证至少留下一个特征
    kept_tokens = [tok for tok in tokens if random.random() > drop_rate]
    if not kept_tokens and tokens:
        kept_tokens = [random.choice(tokens)]
    return " ".join(kept_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始纯净测试集 (TSV)")
    parser.add_argument("--output", required=True, help="输出的蒙特卡洛测试集 (TSV)")
    parser.add_argument("--truth", required=True, help="用于对照的真值表 (CSV)")
    parser.add_argument("--type", type=str, default="loss", help="变异类型")
    parser.add_argument("--rate", type=float, default=0.1, help="丢包率/变异率")
    parser.add_argument("--N", type=int, default=500, help="每个样本的蒙特卡洛采样次数")
    parser.add_argument("--n", type=int, default=1000, help="随机抽取的总样本数")
    args = parser.parse_args()

    print(f"[*] 正在读取原始数据集: {args.input}")
    df = pd.read_csv(args.input, sep='\t', header=0)
    
    if len(df) < args.n:
        print(f"[!] 警告: 测试集总数 ({len(df)}) 小于请求抽取数 ({args.n})，将使用全部数据。")
        args.n = len(df)
        
    # 随机抽取 n 个样本
    sampled_df = df.sample(n=args.n, random_state=42).reset_index(drop=True)
    
    mc_test_data = []
    mc_truth_data = []

    print(f"[*] 正在生成蒙特卡洛变异样本 (共 {args.n} * {args.N} = {args.n * args.N} 条)...")
    for idx, row in tqdm(sampled_df.iterrows(), total=args.n):
        true_label = row['label']
        original_seq = str(row['text_a'])
        
        # 记录真值，用于后续核对
        mc_truth_data.append({"sample_id": idx, "true_label": true_label})
        
        # 生成 N 次物理变异
        for _ in range(args.N):
            if args.type == "loss":
                mutated_seq = random_loss_augment(original_seq, args.rate)
            else:
                mutated_seq = original_seq # 预留其他变异方法
            
            mc_test_data.append({"label": true_label, "text_a": mutated_seq})

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 写入 TSV (给模型推理用)
    print(f"[*] 正在保存变异测试集: {args.output}")
    pd.DataFrame(mc_test_data).to_csv(args.output, sep='\t', index=False)
    
    # 写入 CSV (给绘图脚本计算准确率用)
    print(f"[*] 正在保存真值对照表: {args.truth}")
    pd.DataFrame(mc_truth_data).to_csv(args.truth, index=False)
    print("[+] 蒙特卡洛数据生成完毕！")

if __name__ == "__main__":
    main()