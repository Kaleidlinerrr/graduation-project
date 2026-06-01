#!/usr/bin/python3
#-*- coding:utf-8 -*-

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd

def evaluate(true_path, pred_path):
    true_df = pd.read_csv(true_path, sep='\t')
    y_true = true_df['label'].tolist()
    
    pred_df = pd.read_csv(pred_path, sep='\t')
    y_pred = pred_df['label'].tolist()
    
    print("\n" + "="*50)
    print("=== 鲁棒性测试 (Drop 噪声) 评估报告 ===")
    print("="*50)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"[*] 准确率 (Accuracy) : {acc:.4f}")
    
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"[*] F1 Macro          : {f1_macro:.4f}")
    
    precision = precision_score(y_true, y_pred, average='macro')
    print(f"[*] Precision Macro   : {precision:.4f}")
    
    recall = recall_score(y_true, y_pred, average='macro')
    print(f"[*] Recall Macro      : {recall:.4f}")
    
    print("\n[*] 混淆矩阵 (Confusion Matrix):")
    print(confusion_matrix(y_true, y_pred))
    print("="*50 + "\n")

if __name__ == '__main__':
    true_file = "datasets_noisy/test_dataset.tsv"
    pred_file = "datasets_noisy/prediction_noisy_drop.tsv"
    evaluate(true_file, pred_file)