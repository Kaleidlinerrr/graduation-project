#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
import os
import csv
import dpkt
import socket
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
from tqdm import tqdm
import random
import shutil  

# ================= 论文标准配置区 =================
MAX_SEQ_LENGTH = 256     # 截断序列最大长度 (对应时序序列的标准化对齐尺寸)
MAX_SAMPLES_PER_CLASS = 5000 
# ==================================================

parser = argparse.ArgumentParser()
parser.add_argument("--pcap_path", required=True, help="输入: 增强后的 PCAP 文件夹路径")
parser.add_argument("--dataset_dir", required=True, help="输出: TSV 存放目录")
parser.add_argument("--cache_dir", required=True, help="缓存: npy 缓存存放目录")
parser.add_argument("--force_clear", action="store_true", help="强制清空旧缓存并重新提取") 
args, _ = parser.parse_known_args()

pcap_path = args.pcap_path if args.pcap_path.endswith('/') else args.pcap_path + '/'
dataset_dir = args.dataset_dir if args.dataset_dir.endswith('/') else args.dataset_dir + '/'
cache_root_dir = args.cache_dir if args.cache_dir.endswith('/') else args.cache_dir + '/'

# ----------------- 极速 DPKT 核心引擎 (含三重回退机制) -----------------
def get_bidi_sessions_fast(file_path):
    """底层物理层双向流解析器，引入三重协议回退机制，完美兼容 VPN 裸 IP 流"""
    sessions = {}
    with open(file_path, 'rb') as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            pcap = dpkt.pcapng.Reader(f)

        for ts, buf in pcap:
            try:
                ip = None
                
                # [尝试 1]：作为常规以太网帧解析 (Ethernet)
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                        ip = eth.data
                except Exception:
                    pass

                # [尝试 2]：作为裸 IP 解析 (Raw IP) - VPN 虚拟网卡克星
                if ip is None:
                    try:
                        raw_ip = dpkt.ip.IP(buf)
                        if raw_ip.v in (4, 6):
                            ip = raw_ip
                    except Exception:
                        pass
                
                # [尝试 3]：作为 Linux SLL 解析 ("any" 网卡)
                if ip is None:
                    try:
                        sll = dpkt.sll.SLL(buf)
                        if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                            ip = sll.data
                    except Exception:
                        pass

                if ip is None or not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue
                
                if isinstance(ip, dpkt.ip.IP):
                    src_ip = socket.inet_ntoa(ip.src)
                    dst_ip = socket.inet_ntoa(ip.dst)
                else:
                    src_ip = socket.inet_ntop(socket.AF_INET6, ip.src)
                    dst_ip = socket.inet_ntop(socket.AF_INET6, ip.dst)

                trans = ip.data
                if isinstance(trans, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    sport = trans.sport
                    dport = trans.dport
                else:
                    continue

                key = tuple(sorted([f"{src_ip}:{sport}", f"{dst_ip}:{dport}"]))
                
                if key not in sessions:
                    sessions[key] = []
                
                sessions[key].append({
                    "length": len(buf),
                    "src_ip": src_ip
                })
            except Exception:
                continue 
    return sessions

def extract_features_from_pcap_dir(pcap_dir):
    """深度递归穿透扫描，自适应绕过外壳目录提取真实类别"""
    print(f"[*] 开始使用动态感知引擎深度递归扫描: {pcap_dir}")
    X_all, Y_all = [], []
    
    # 1. 深度遍历所有文件，建立 "真实类别 -> [文件列表]" 的映射
    class_to_files = {}
    for root, dirs, files in os.walk(pcap_dir):
        for f in files:
            if f.lower().endswith('.pcap') or f.lower().endswith('.pcapng'):
                rel_path = os.path.relpath(root, pcap_dir)
                
                # 将路径拆解为层级节点 (例如: ['splitcap', 'Chat'] 或 ['aim'])
                parts = []
                curr = rel_path
                while curr and curr != '.':
                    curr, tail = os.path.split(curr)
                    if tail:
                        parts.insert(0, tail)
                
                # 寻找第一个真正有分类意义的目录名（跳过 'splitcap' 等外壳）
                class_name = "Unknown"
                for part in parts:
                    if part.lower() != 'splitcap':
                        class_name = part
                        break
                
                if class_name != "Unknown":
                    if class_name not in class_to_files:
                        class_to_files[class_name] = []
                    class_to_files[class_name].append(os.path.join(root, f))

    subdirs = list(class_to_files.keys())
    subdirs.sort() 
    
    if not subdirs:
        print("[!] 警告：目标路径及其所有深层子目录中均未发现 PCAP 文件！")
        return [], []

    # ================= 动态多分类自适应标签映射 =================
    label_map = {}
    is_iot_task = any('benign' in d.lower() or 'recon' in d.lower() or 'flood' in d.lower() or 'loris' in d.lower() for d in subdirs)
    
    if is_iot_task:
        print("[*] 检测到当前任务为 IoT 细分类，自动进行独立 6 分类标签映射...")
        iot_mapped_indices = {}
        for d in subdirs:
            name = d.lower()
            if 'benign' in name: iot_mapped_indices[d] = 0
            elif 'syn_flood' in name: iot_mapped_indices[d] = 1
            elif 'slowloris' in name: iot_mapped_indices[d] = 2
            elif 'hostdiscovery' in name: iot_mapped_indices[d] = 3
            elif 'osscan' in name: iot_mapped_indices[d] = 4
            elif 'portscan' in name: iot_mapped_indices[d] = 5
        label_map = iot_mapped_indices
        total_categories = 6
    else:
        print("[*] 检测到标准 VPN 数据集任务，启用自适应递增标签映射...")
        for idx, d in enumerate(subdirs):
            label_map[d] = idx
        total_categories = len(subdirs)
            
    print(f"[+] 经深度推断确定的最终类别映射关系 (共 {total_categories} 类):")
    for folder, idx in label_map.items():
        file_count = len(class_to_files.get(folder, []))
        print(f"    真实业务类: {folder.ljust(25)} ----> Token ID: {idx} (扫出 {file_count} 个底层文件)")

    dataset_statistic = [0] * total_categories

    # 2. 开始核心特征提取
    for label_name, label_idx in label_map.items():
        pcap_files = class_to_files.get(label_name, [])
        class_sequences = []
        
        if not pcap_files:
            continue
            
        for file_path in tqdm(pcap_files, desc=f"特征转换 [{label_name}]"):
            sessions = get_bidi_sessions_fast(file_path)
            
            for key, packets_meta in sessions.items():
                if len(packets_meta) < 3:
                    continue
                    
                client_ip = packets_meta[0]["src_ip"]
                lengths = []
                for p_meta in packets_meta[:MAX_SEQ_LENGTH]:
                    direction = 1 if p_meta["src_ip"] == client_ip else -1
                    lengths.append(str(direction * p_meta["length"]))
                    
                class_sequences.append(" ".join(lengths))

        if len(class_sequences) > MAX_SAMPLES_PER_CLASS:
            class_sequences = random.sample(class_sequences, MAX_SAMPLES_PER_CLASS)
            
        dataset_statistic[label_idx] += len(class_sequences)
        X_all.extend(class_sequences)
        Y_all.extend([label_idx] * len(class_sequences))

    print("\n" + "="*20 + " 数据集提取统计报告 " + "="*20)
    for idx, count in enumerate(dataset_statistic):
        print(f"  类别索引 ID {idx}:\t{count} 条有符号序列")
    print(f"  全局总流数统计:\t{sum(dataset_statistic)} 条")
    print("="*60 + "\n")
    
    return X_all, Y_all

# ----------------- 数据集格式化写入 -----------------
def write_dataset_tsv(data, label, file_dir, data_type):
    dataset_file = [["label", "text_a"]]
    for index in range(len(label)):
        dataset_file.append([label[index], data[index]])
    
    os.makedirs(file_dir, exist_ok=True)
    out_path = os.path.join(file_dir, f"{data_type}_dataset.tsv")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)

def unlabel_data(label_data_path, dataset_out_dir):
    nolabel_data = ""
    with open(label_data_path, newline='', encoding='utf-8') as f:
        data = csv.reader(f, delimiter='\t')
        next(data) 
        for row in data:
            if len(row) > 1:
                nolabel_data += row[1] + '\n'
    nolabel_file = os.path.join(dataset_out_dir, "nolabel_test_dataset.tsv")
    with open(nolabel_file, 'w', newline='', encoding='utf-8') as f:
        f.write(nolabel_data)

def models_deal(x_length_train, x_length_test, x_length_valid, y_train, y_test, y_valid):
    print("[*] 正在向指定物理路径生成标准化 TSV 数据集...")
    write_dataset_tsv(x_length_train, y_train, dataset_dir, "train")
    write_dataset_tsv(x_length_test, y_test, dataset_dir, "test")
    write_dataset_tsv(x_length_valid, y_valid, dataset_dir, "valid")
    
    test_tsv_path = os.path.join(dataset_dir, "test_dataset.tsv")
    unlabel_data(test_tsv_path, dataset_dir)
    print(f"[+] 序列化 TSV 语料生产成功！目标存储路径: {dataset_dir}")

def dataset_extract():
    task_name = [p for p in pcap_path.split('/') if p][-1]
    cache_path = os.path.join(cache_root_dir, f"dataset_{task_name}/")

    if args.force_clear and os.path.exists(cache_path):
        print(f"[!] 接收到强制清理指令，正在移除历史隔离缓存: {cache_path} ...")
        shutil.rmtree(cache_path)

    try:
        if os.path.exists(cache_path) and os.listdir(cache_path):
            print(f"[*] 检测到历史隔离缓存，正在极速载入: {cache_path} ...")
            x_length_train = np.load(cache_path + "x_length_train.npy", allow_pickle=True)
            x_length_test = np.load(cache_path + "x_length_test.npy", allow_pickle=True)
            x_length_valid = np.load(cache_path + "x_length_valid.npy", allow_pickle=True)
            y_train = np.load(cache_path + "y_train.npy", allow_pickle=True)
            y_test = np.load(cache_path + "y_test.npy", allow_pickle=True)
            y_valid = np.load(cache_path + "y_valid.npy", allow_pickle=True)
            
            models_deal(x_length_train, x_length_test, x_length_valid, y_train, y_test, y_valid)
            return
    except Exception as e:
        print(f"[!] 隔离缓存不可读 ({e})，启动原始链路数据流解析引擎...")

    X_length, Y_all = extract_features_from_pcap_dir(pcap_path)

    if not X_length:
        print("[!] 严重错误: 目标路径下未提取到有效会话序列，请确认输入数据源是否正确。")
        return

    x_length = np.array(X_length)
    dataset_label = np.array(Y_all)

    # 规避小样本切分崩溃
    unique_labels, label_counts = np.unique(dataset_label, return_counts=True)
    min_class_size = np.min(label_counts) if len(label_counts) > 0 else 0

    if min_class_size < 2:
        print(f"[!] 警报: 检测到存在极度匮乏的孤立类别（最小样本数={min_class_size}），自动降级为纯随机切分。")
        indices = np.arange(len(x_length))
        random.seed(42)
        random.shuffle(indices)
        
        train_bound = int(len(indices) * 0.8)
        val_bound = int(len(indices) * 0.9)
        
        x_length_train, y_train = x_length[indices[:train_bound]], dataset_label[indices[:train_bound]]
        x_length_valid, y_valid = x_length[indices[train_bound:val_bound]], dataset_label[indices[train_bound:val_bound]]
        x_length_test, y_test = x_length[indices[val_bound:]], dataset_label[indices[val_bound:]]
    else:
        split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41) 
        split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42) 

        for train_index, test_index in split_1.split(x_length, dataset_label):
            x_length_train, y_train = x_length[train_index], dataset_label[train_index]
            x_length_test, y_test = x_length[test_index], dataset_label[test_index]
            
        for test_index, valid_index in split_2.split(x_length_test, y_test):
            x_length_valid, y_valid = x_length_test[valid_index], y_test[valid_index]
            x_length_test = x_length_test[test_index]
            y_test = y_test[test_index]

    os.makedirs(cache_path, exist_ok=True)
    np.save(os.path.join(cache_path, 'x_length_train.npy'), x_length_train)
    np.save(os.path.join(cache_path, 'x_length_test.npy'), x_length_test)
    np.save(os.path.join(cache_path, 'x_length_valid.npy'), x_length_valid)
    np.save(os.path.join(cache_path, 'y_train.npy'), y_train)
    np.save(os.path.join(cache_path, 'y_test.npy'), y_test)
    np.save(os.path.join(cache_path, 'y_valid.npy'), y_valid)

    models_deal(x_length_train, x_length_test, x_length_valid, y_train, y_test, y_valid)

if __name__ == '__main__':
    dataset_extract()