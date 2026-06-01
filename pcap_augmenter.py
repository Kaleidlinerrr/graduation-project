#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
import random
import argparse
from tqdm import tqdm
import scapy.all as scapy

def augment_packet(pkt, args):
    """
    对单个数据包进行概率性增强
    返回: 增强后的包列表 (可能为空[丢包], 单个[正常/变异], 多个[重传])
    """
    # 1. Deletion (随机丢包)
    if random.random() < args.drop_prob:
        return []

    # 2. Random Replace / Padding (载荷长度变异)
    if random.random() < args.pad_prob:
        pad_len = random.randint(1, args.max_pad_len)
        # 在包尾追加全 0 字节模拟 Padding
        padding = scapy.Raw(load=b'\x00' * pad_len)
        if pkt.haslayer(scapy.Raw):
            pkt[scapy.Raw].load += b'\x00' * pad_len
        elif pkt.haslayer(scapy.IP) or pkt.haslayer(scapy.IPv6):
            pkt = pkt / padding
        
        # 删除原始长度和校验和，强制 Scapy 在 wrpcap 时重新计算新长度
        if pkt.haslayer(scapy.IP):
            del pkt[scapy.IP].len
            del pkt[scapy.IP].chksum
        elif pkt.haslayer(scapy.IPv6):
            del pkt[scapy.IPv6].plen
        if pkt.haslayer(scapy.TCP):
            del pkt[scapy.TCP].chksum
        elif pkt.haslayer(scapy.UDP):
            del pkt[scapy.UDP].chksum
            del pkt[scapy.UDP].len

    # 3. Insertion / Duplication (模拟重传)
    if random.random() < args.dup_prob:
        return [pkt, pkt] # 返回两次，模拟网络中的 Duplicate Packet

    return [pkt]

def process_pcap(input_file, output_file, args):
    try:
        packets = scapy.rdpcap(input_file)
        augmented_packets = []
        
        skip_span = 0 # 用于处理 Span Mask (突发丢包)
        
        for pkt in packets:
            # Span Mask (突发丢包逻辑)
            if skip_span > 0:
                skip_span -= 1
                continue
            if random.random() < args.span_drop_prob:
                skip_span = random.randint(2, args.max_span_len)
                continue

            # 单包级别增强
            aug_pkts = augment_packet(pkt, args)
            augmented_packets.extend(aug_pkts)
            
        scapy.wrpcap(output_file, augmented_packets)
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="PCAP Data Augmenter Engine")
    parser.add_argument("--input_dir", required=True, help="源 PCAP 文件夹路径")
    parser.add_argument("--output_dir", required=True, help="增强后 PCAP 存放路径")
    
    # 噪声参数配置 (默认设置了一些微小的扰动概率)
    parser.add_argument("--drop_prob", type=float, default=0.05, help="单包随机丢包概率 (Deletion)")
    parser.add_argument("--span_drop_prob", type=float, default=0.01, help="突发连续丢包概率 (Span Mask)")
    parser.add_argument("--max_span_len", type=int, default=5, help="突发丢包的最大连续长度")
    parser.add_argument("--pad_prob", type=float, default=0.1, help="长度变异概率 (Random Replace)")
    parser.add_argument("--max_pad_len", type=int, default=100, help="最大 Padding 字节数")
    parser.add_argument("--dup_prob", type=float, default=0.02, help="数据包重复概率 (Insertion)")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 递归遍历输入文件夹
    pcap_files = []
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.endswith('.pcap') or f.endswith('.pcapng'):
                pcap_files.append(os.path.join(root, f))

    print(f"[*] Found {len(pcap_files)} PCAP files. Starting augmentation engine...")
    
    for pcap_path in tqdm(pcap_files):
        # 保持原始目录结构
        rel_path = os.path.relpath(pcap_path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        out_dir = os.path.dirname(out_path)
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        process_pcap(pcap_path, out_path, args)
        
    print("[+] Data Augmentation Complete!")

if __name__ == '__main__':
    main()