import json

def filter_invalid_entities(data):
    """删除Entity_valid为false的实体"""
    filtered_count = 0
    
    for item in data:
        if 'entities' in item:
            # 过滤掉Entity_valid为false的实体
            valid_entities = []
            for entity in item['entities']:
                # 检查Entity_valid字段，可能是字符串"false"或布尔值false
                if 'Entity_valid' in entity:
                    valid_value = entity['Entity_valid']
                    # 处理不同类型的false值
                    if isinstance(valid_value, str):
                        valid_value = valid_value.lower() != "false"
                    
                    if valid_value:  # 如果有效，保留此实体
                        valid_entities.append(entity)
                    else:
                        filtered_count += 1
                else:
                    # 如果没有Entity_valid字段，默认保留
                    valid_entities.append(entity)
            
            # 更新实体列表
            item['entities'] = valid_entities
    
    return filtered_count

def merge_entity_types(val_results_path, converted_results_path, output_path):
    # 读取验证结果文件
    with open(val_results_path, 'r', encoding='utf-8') as f:
        val_results = json.load(f)
    
    # 读取包含Type信息的NER结果文件
    with open(converted_results_path, 'r', encoding='utf-8') as f:
        converted_results = json.load(f)
    
    # 创建映射字典，键为(content, url)，值为Type列表
    type_mapping = {}
    for item in converted_results:
        if 'content' not in item or 'url' not in item:
            continue
            
        content = item['content']
        url = item['url']
        
        # 直接提取Type列表，按实体顺序
        type_list = []
        for entity in item.get('entities', []):
            if 'Type' in entity:
                type_list.append(entity['Type'])
        
        type_mapping[(content, url)] = type_list
    
    # 为val_results中的每个实体按顺序添加Type字段
    match_count = 0
    mismatch_count = 0
    for val_item in val_results:
        if 'content' not in val_item or 'url' not in val_item:
            continue
            
        content = val_item['content']
        url = val_item['url']
        key = (content, url)
        
        if key in type_mapping:
            type_list = type_mapping[key]
            
            # 按顺序添加Type
            for i, entity in enumerate(val_item.get('entities', [])):
                if i < len(type_list):
                    entity['Type'] = type_list[i]
                    match_count += 1
                else:
                    # 如果val_results中的实体数量多于converted_results
                    entity['Type'] = "UNKNOWN"
                    mismatch_count += 1
        else:
            # 如果找不到匹配的条目
            for entity in val_item.get('entities', []):
                entity['Type'] = "UNKNOWN"
                mismatch_count += 1
    
    # 过滤无效实体
    filtered_count = filter_invalid_entities(val_results)
    
    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(val_results, f, ensure_ascii=False, indent=4)
    
    print(f"已将Type字段添加到{len(val_results)}个条目中")
    print(f"成功匹配: {match_count}个实体, 未匹配: {mismatch_count}个实体")
    print(f"已删除 {filtered_count} 个无效实体")
    print(f"结果保存到{output_path}")

if __name__ == "__main__":
    # 文件路径
    val_results_path = 'result/zero-shot-demo/format_val/format_val_results.json'
    converted_results_path = 'result/zero-shot-demo/ner/origin-cot/sota/converted_ner_results.json'
    output_path = 'result/zero-shot-demo/format_val/val_results_with_type.json'
    
    merge_entity_types(val_results_path, converted_results_path, output_path)