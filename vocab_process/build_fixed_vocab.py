#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os

def generate_fixed_vocab(save_path, max_mtu=1600):
    """
    生成适用于包长序列的固定词汇表 (Fixed Vocabulary)
    """
    special_tokens = [
        "[PAD]", 
        "[UNK]", 
        "[CLS]", 
        "[SEP]", 
        "[MASK]", 
        "<S>", 
        "<T>"
    ]
    
    with open(save_path, 'w', encoding='utf-8') as f:
        # 1. 写入所有特殊 Tokens
        for token in special_tokens:
            f.write(f"{token}\n")
            
        # 2. 写入所有的负方向包长 (代表 Server 发往 Client)
        for i in range(-max_mtu, 0):
            f.write(f"{i}\n")
            
        # 3. 写入所有的正方向包长 (代表 Client 发往 Server)
        for i in range(1, max_mtu + 1):
            f.write(f"{i}\n")

    abs_path = os.path.abspath(save_path)

if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_root_dir = os.path.dirname(current_script_dir)
    
    models_dir = os.path.join(project_root_dir, "models")
    
    os.makedirs(models_dir, exist_ok=True)
    
    vocab_file_path = os.path.join(models_dir, "fixed_length_vocab.txt")
    
    generate_fixed_vocab(vocab_file_path, max_mtu=1600)