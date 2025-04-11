#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict

def compare_entity_differences(file1_path, file2_path, output_path=None):
    """
    比较两个JSON文件中实体跨度和类型的差异
    
    Args:
        file1_path: 第一个JSON文件路径 (eeg_input.json)
        file2_path: 第二个JSON文件路径 (eeg_results.json)
        output_path: 输出结果文件路径（可选）
    """
    # 读取两个文件
    with open(file1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    with open(file2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # 构建第一个文件的映射 {content -> {entity_span -> type}}
    entity_map1 = {}
    for item in data1:
        content = item.get('Text_sentence')  # eeg_input.json使用Text_sentence字段
        if not content:
            continue
            
        entity_types = {}
        for entity in item.get('NER_results', []):
            if 'Entity_span' in entity and 'Type' in entity:
                entity_types[entity['Entity_span']] = entity['Type']
        
        entity_map1[content] = entity_types
    
    # 构建第二个文件的映射 {content -> {entity_span -> type}}
    entity_map2 = {}
    for item in data2:
        content = item.get('content')  # eeg_results.json使用content字段
        if not content:
            continue
            
        entity_types = {}
        for entity in item.get('entities', []):
            if 'Entity_span' in entity and 'Type' in entity:
                entity_types[entity['Entity_span']] = entity['Type']
        
        entity_map2[content] = entity_types
    
    # 比较两个映射
    all_contents = set(entity_map1.keys()) | set(entity_map2.keys())
    
    # 结果统计
    total_contents = len(all_contents)
    matched_contents = 0
    different_type_counts = 0
    different_span_counts = 0
    type_differences = []
    type_difference_stats = defaultdict(int)  # 统计不同类型转换的数量
    span_differences = []
    
    # 查找差异
    for content in all_contents:
        if content in entity_map1 and content in entity_map2:
            matched_contents += 1
            
            # 获取两个文件中此内容的实体映射
            entities1 = entity_map1[content]
            entities2 = entity_map2[content]
            
            # 查找共同的实体名称
            common_entities = set(entities1.keys()) & set(entities2.keys())
            only_in_file1 = set(entities1.keys()) - set(entities2.keys())
            only_in_file2 = set(entities2.keys()) - set(entities1.keys())
            
            # 统计实体跨度差异
            different_span_counts += len(only_in_file1) + len(only_in_file2)
            
            # 记录实体跨度差异
            if only_in_file1 or only_in_file2:
                span_differences.append({
                    'content': content,
                    'only_in_file1': [{'Entity_span': span, 'Type': entities1[span]} for span in only_in_file1],
                    'only_in_file2': [{'Entity_span': span, 'Type': entities2[span]} for span in only_in_file2],
                })
            
            # 检查共同实体的类型差异
            for entity_span in common_entities:
                type1 = entities1[entity_span]
                type2 = entities2[entity_span]
                
                if type1 != type2:
                    different_type_counts += 1
                    type_pair = f"{type1} -> {type2}"
                    type_difference_stats[type_pair] += 1
                    
                    type_differences.append({
                        'content': content,
                        'entity_span': entity_span,
                        'type_in_file1': type1,
                        'type_in_file2': type2
                    })
    
    # 按差异数量排序类型转换统计
    sorted_type_stats = sorted(type_difference_stats.items(), key=lambda x: x[1], reverse=True)
    
    # 输出统计结果
    result = {
        'summary': {
            'total_contents': total_contents,
            'matched_contents': matched_contents,
            'different_type_count': different_type_counts,
            'different_span_count': different_span_counts,
        },
        'type_conversion_stats': dict(sorted_type_stats),
        'span_differences': span_differences,
        'type_differences': type_differences
    }
    
    # 打印统计摘要
    print(f"总计内容条目: {total_contents}")
    print(f"匹配内容条目: {matched_contents}")
    print(f"类型不同的实体数量: {different_type_counts}")
    print(f"实体跨度不同的数量: {different_span_counts}")
    print("\n最常见的类型转换:")
    for type_pair, count in sorted_type_stats[:10]:  # 显示前10个最常见的类型转换
        print(f"  {type_pair}: {count}次")
    
    # 输出实体跨度差异示例
    if span_differences:
        print("\n实体跨度差异示例(前3个):")
        for i, diff in enumerate(span_differences[:3]):
            print(f"\n例 {i+1}: {diff['content'][:50]}..." if len(diff['content']) > 50 else f"\n例 {i+1}: {diff['content']}")
            print(f"  只在文件1中: {', '.join([e['Entity_span'] for e in diff['only_in_file1']])}" if diff['only_in_file1'] else "  只在文件1中: 无")
            print(f"  只在文件2中: {', '.join([e['Entity_span'] for e in diff['only_in_file2']])}" if diff['only_in_file2'] else "  只在文件2中: 无")
    
    # 如果指定了输出路径，保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n详细比较结果已保存至: {output_path}")
    
    return result

if __name__ == "__main__":
    file1_path = "/Users/xxxxxx/Downloads/论文/gmner/result/eeg/eeg_input.json"
    file2_path = "/Users/xxxxxx/Downloads/论文/gmner/result/eeg/eeg_results.json"
    output_path = "/Users/xxxxxx/Downloads/论文/gmner/result/eeg/entity_differences.json"
    
    compare_entity_differences(file1_path, file2_path, output_path)