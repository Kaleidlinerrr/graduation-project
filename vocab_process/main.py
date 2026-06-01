#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
import scapy.all as scapy

# 【核心功能】：自定义双向流聚类函数，确保双向数据包被分入同一会话
def get_bidi_sessions(packets):
    sessions = {}
    for p in packets:
        try:
            if p.haslayer(scapy.IP):
                src, dst = p[scapy.IP].src, p[scapy.IP].dst
            elif p.haslayer(scapy.IPv6):
                src, dst = p[scapy.IPv6].src, p[scapy.IPv6].dst
            else:
                continue

            if p.haslayer(scapy.TCP):
                sport, dport = p[scapy.TCP].sport, p[scapy.TCP].dport
            elif p.haslayer(scapy.UDP):
                sport, dport = p[scapy.UDP].sport, p[scapy.UDP].dport
            else:
                continue

            # 使用排序后的端点对作为键，保证 A->B 和 B->A 的键完全一致
            key = tuple(sorted([f"{src}:{sport}", f"{dst}:{dport}"]))
            
            if key not in sessions:
                sessions[key] = []
            sessions[key].append(p)
        except:
            continue
    return sessions

def main():
    dataset_dir = "/mnt/nfs/kaleid/ET-BERT/dataset/"
    corpus_dir = "/mnt/nfs/kaleid/ET-BERT/corpora/"
    # 保持预训练脚本一致的文件名
    corpus_file = os.path.join(corpus_dir, "length_tls13_burst.txt") 
    
    if not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir)
        
    # 如果旧的语料文件已存在，先将其删除，避免新旧数据重复追加混合
    if os.path.exists(corpus_file):
        os.remove(corpus_file)

    print(f"Begin scanning for PCAP files in: {dataset_dir}")
    
    pcap_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".pcap") or file.endswith(".pcapng"):
                pcap_files.append(os.path.join(root, file))
                
    total_used_packets = 0
    file_count = 0
    
    for pcap_path in pcap_files:
        file_count += 1
        file_name = os.path.basename(pcap_path)
        print(f"No.{file_count} pacp is processed ... {file_name} ...")
        
        try:
            # 载入数据包并进行双向流聚类
            packets = scapy.rdpcap(pcap_path)
            sessions = get_bidi_sessions(packets)
            
            burst_txt = ""
            for session_key, session_packets in sessions.items():
                # 过滤掉交互包数过少的无效流
                if len(session_packets) < 3:
                    continue
                
                # 获取建连第一包的源IP，作为正方向基准
                client_ip = None
                if session_packets[0].haslayer(scapy.IP):
                    client_ip = session_packets[0][scapy.IP].src
                elif session_packets[0].haslayer(scapy.IPv6):
                    client_ip = session_packets[0][scapy.IPv6].src

                lengths = []
                # 截断长度设为256（与预训练 --seq_length 256 保持一致）
                for p in session_packets[:256]: 
                    plen = len(p)
                    direction = 1
                    
                    # 判断当前包的源IP是否为客户端IP，如果不是则打上负号
                    if client_ip:
                        if p.haslayer(scapy.IP) and p[scapy.IP].src != client_ip:
                            direction = -1
                        elif p.haslayer(scapy.IPv6) and p[scapy.IPv6].src != client_ip:
                            direction = -1
                            
                    lengths.append(str(direction * plen))
                    total_used_packets += 1
                
                burst_txt += " ".join(lengths) + "\n\n"
                
            # 将处理完的当前 PCAP 的所有流特征追加写入 txt
            if burst_txt:
                with open(corpus_file, 'a') as f:
                    f.write(burst_txt)
                    
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print(f"finish preprocessing {file_count} pcaps")
    print(f"used packets {total_used_packets}")
    print("finish generating pretrain dataset.")
    print(f" please check in {corpus_dir}")

if __name__ == '__main__':
    main()