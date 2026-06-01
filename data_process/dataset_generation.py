#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
import json
import tqdm
import random
import scapy.all as scapy

random.seed(40)

word_dir = "/mnt/nfs/kaleid/ET-BERT/corpora/"
word_name = "length_burst.txt"

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

            # 使用排序后的端点对作为键，这样 A->B 和 B->A 的键将完全一致
            key = tuple(sorted([f"{src}:{sport}", f"{dst}:{dport}"]))
            
            if key not in sessions:
                sessions[key] = []
            sessions[key].append(p)
        except:
            continue
    return sessions

def get_burst_feature(label_pcap, payload_pac=256):
    try:
        packets = scapy.rdpcap(label_pcap)
        sessions = get_bidi_sessions(packets)
        
        burst_txt = ""
        for session_key, session_packets in sessions.items():
            if len(session_packets) < 3:
                continue
            
            # 第一包的源 IP 作为“客户端”（正方向）
            client_ip = None
            if session_packets[0].haslayer(scapy.IP):
                client_ip = session_packets[0][scapy.IP].src
            elif session_packets[0].haslayer(scapy.IPv6):
                client_ip = session_packets[0][scapy.IPv6].src

            lengths = []
            for p in session_packets[:payload_pac]:
                plen = len(p)
                direction = 1
                if client_ip:
                    # 如果后续包的源 IP 不是建连的第一包 IP，标记为反向（负数）
                    if p.haslayer(scapy.IP) and p[scapy.IP].src != client_ip:
                        direction = -1
                    elif p.haslayer(scapy.IPv6) and p[scapy.IPv6].src != client_ip:
                        direction = -1
                lengths.append(str(direction * plen))
            
            burst_txt += " ".join(lengths) + "\n\n"
            
        if burst_txt:
            if not os.path.exists(word_dir):
                os.makedirs(word_dir)
            with open(os.path.join(word_dir, word_name), 'a') as f:
                f.write(burst_txt)
    except Exception as e:
        print(f"Skipping {label_pcap} due to error: {e}")
    return 0

def get_feature_packet(label_pcap, payload_len=256):
    feature_data = []
    try:
        packets = scapy.rdpcap(label_pcap)
        packet_data_string = ''  
        for packet in packets:
            packet_len = len(packet)
            packet_data_string += str(packet_len)
            break
        feature_data.append(packet_data_string)
    except:
        return -1
    return feature_data

def get_multiple_feature_flows(label_pcap, max_flows, payload_pac=128):
    feature_data = []
    try:
        print(f"\n[INFO] Loading large PCAP into memory: {label_pcap} ...")
        packets = scapy.rdpcap(label_pcap)
        print(f"[INFO] Clustering bidirectional sessions for {label_pcap} ...")
        
        # 使用修复后的双向流聚类
        sessions = get_bidi_sessions(packets)
        
        for session_key, session_packets in sessions.items():
            if len(session_packets) < 3:
                continue
            
            client_ip = None
            if session_packets[0].haslayer(scapy.IP):
                client_ip = session_packets[0][scapy.IP].src
            elif session_packets[0].haslayer(scapy.IPv6):
                client_ip = session_packets[0][scapy.IPv6].src

            lengths = []
            for p in session_packets[:payload_pac]:
                plen = len(p)
                direction = 1
                if client_ip:
                    # 如果源 IP 与发起方不同，则打上负号
                    if p.haslayer(scapy.IP) and p[scapy.IP].src != client_ip:
                        direction = -1
                    elif p.haslayer(scapy.IPv6) and p[scapy.IPv6].src != client_ip:
                        direction = -1
                lengths.append(str(direction * plen))
            
            flow_data_string = " ".join(lengths)
            feature_data.append(flow_data_string)
            
            if len(feature_data) >= max_flows:
                break 
                
        return feature_data
    except Exception as e:
        print(f"Error parsing {label_pcap}: {e}")
        return []

def generation(pcap_path, samples, features, splitcap = False, payload_length = 128, payload_packet = 128, dataset_save_path = "/mnt/nfs/kaleid/ET-BERT/ex_results/", dataset_level = "flow"):
    if os.path.exists(os.path.join(dataset_save_path, "dataset.json")):
        print("the pcap file of %s is finished generating."%pcap_path)
        X, Y = obtain_data(pcap_path, samples, features, dataset_save_path)
        return X,Y

    dataset = {}
    label_name_list = []
    session_pcap_path  = {}

    for parent, dirs, files in os.walk(pcap_path):
        if label_name_list == []:
            label_name_list.extend(dirs)
        for dir in label_name_list:
            session_pcap_path[dir] = os.path.join(pcap_path, dir)
        break

    label_id = {}
    for index in range(len(label_name_list)):
        label_id[label_name_list[index]] = index

    r_file_record = []
    print("\nBegin to generate features.")

    label_count = 0
    for key in session_pcap_path.keys():
        if dataset_level == "flow":
            if label_id[key] not in dataset:
                dataset[label_id[key]] = {"samples": 0, "length": {}, "payload": {}}

        target_all_files = [os.path.join(p, f) for p, d, files in os.walk(session_pcap_path[key]) for f in files if f.endswith('.pcap') or f.endswith('.pcapng')]
        
        random.shuffle(target_all_files)
        
        for r_f in target_all_files:
            if dataset[label_id[key]]["samples"] >= samples[label_count]:
                break
                
            r_file_record.append(r_f)
            
            if dataset_level == "flow":
                needed_samples = samples[label_count] - dataset[label_id[key]]["samples"]
                flow_features = get_multiple_feature_flows(r_f, max_flows=needed_samples, payload_pac=payload_packet)
                
                for flow_string in flow_features:
                    dataset[label_id[key]]["samples"] += 1
                    sample_idx = str(dataset[label_id[key]]["samples"])
                    dataset[label_id[key]]["length"][sample_idx] = flow_string
                    
        label_count += 1

    all_data_number = 0
    print("\nExtraction Summary:")
    for index in range(len(label_name_list)):
        print("%s\t%s\t%d"%(label_id[label_name_list[index]], label_name_list[index], dataset[label_id[label_name_list[index]]]["samples"]))
        all_data_number += dataset[label_id[label_name_list[index]]]["samples"]
    print("all\t%d\n"%(all_data_number))

    if not os.path.exists(dataset_save_path):
        os.makedirs(dataset_save_path)

    with open(os.path.join(dataset_save_path, "picked_file_record"),"w") as p_f:
        for i in r_file_record:
            p_f.write(i + "\n")
    with open(os.path.join(dataset_save_path, "dataset.json"), "w") as f:
        json.dump(dataset,fp=f,ensure_ascii=False,indent=4)

    X,Y = obtain_data(pcap_path, samples, features, dataset_save_path, json_data = dataset)
    return X,Y

def read_data_from_json(json_data, features, samples):
    X,Y = [], []
    for feature_index in range(len(features)):
        x = []
        label_count = 0
        for label in json_data.keys():
            sample_num = json_data[label]["samples"]
            if X == []:
                y = [label] * sample_num
                Y.append(y)
            
            x_label = []
            for sample_index in json_data[label][features[feature_index]].keys():
                x_label.append(json_data[label][features[feature_index]][sample_index])
            x.append(x_label)
            label_count += 1
        X.append(x)
    return X,Y

def obtain_data(pcap_path, samples, features, dataset_save_path, json_data = None):
    if json_data:
        X,Y = read_data_from_json(json_data,features,samples)
    else:
        with open(os.path.join(dataset_save_path, "dataset.json"),"r") as f:
            dataset = json.load(f)
        X,Y = read_data_from_json(dataset,features,samples)

    for index in range(len(X)):
        if len(X[index]) != len(Y):
            print("data and labels are not properly associated.")
            return -1
    return X,Y