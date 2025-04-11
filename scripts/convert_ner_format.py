import json
import re
import os

def convert_ner_format(input_file, output_file):
    """转换NER结果格式，确保正确提取所有实体预测结果"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"原始数据：共{len(data)}条")
    
    converted_data = []
    entity_count = 0
    items_with_entities = 0
    empty_entities_count = 0
    
    for idx, item in enumerate(data):
        new_item = {
            "content": item.get("content", ""),
            "url": item.get("url", "")
        }
        
        # 提取实体信息
        entities = []
        found_entity = False
        
        if "entities" in item:
            try:
                # 场景1: entities是列表
                if isinstance(item["entities"], list):
                    for entity_info in item["entities"]:
                        if isinstance(entity_info, dict) and "output" in entity_info:
                            output_text = entity_info["output"]
                            input_text = entity_info.get("input", "")
                            
                            # 如果输入和输出完全相同，说明没有找到实体
                            if output_text.strip() == input_text.strip():
                                continue
                                
                            # 提取实体标注
                            matches = re.findall(r'\{([^{}]+?),\s*([^{}]+?)\}', output_text)
                            if matches:
                                found_entity = True
                                for entity_span, entity_type in matches:
                                    entities.append({
                                        "Entity_span": entity_span.strip(),
                                        "Type": entity_type.strip()
                                    })
                
                # 场景2: entities是字符串，可能是JSON代码块
                elif isinstance(item["entities"], str):
                    # 检查是否包含JSON代码块
                    if "```json" in item["entities"]:
                        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', item["entities"])
                        if json_match:
                            try:
                                json_str = json_match.group(1).strip()
                                parsed_json = json.loads(json_str)
                                
                                # 处理解析后的JSON
                                no_entities_found = True  # 标记是否找到了实体
                                
                                if isinstance(parsed_json, list):
                                    for entry in parsed_json:
                                        if isinstance(entry, dict) and "output" in entry:
                                            output_text = entry["output"]
                                            input_text = entry.get("input", "")
                                            
                                            # 关键检查：输入输出相同，表示没有实体
                                            if output_text.strip() == input_text.strip():
                                                print(f"条目{idx}: 输入输出相同，没有找到实体")
                                                continue
                                                
                                            # 提取实体
                                            matches = re.findall(r'\{([^{}]+?),\s*([^{}]+?)\}', output_text)
                                            if matches:
                                                no_entities_found = False
                                                found_entity = True
                                                for entity_span, entity_type in matches:
                                                    entities.append({
                                                        "Entity_span": entity_span.strip(),
                                                        "Type": entity_type.strip()
                                                    })
                                                    
                                # 调试输出
                                if no_entities_found:
                                    print(f"条目{idx}: JSON代码块中未找到实体标注")
                                    
                            except json.JSONDecodeError as e:
                                print(f"JSON解析错误(条目{idx}): {e}")
                                # 仅在明确包含实体标记时才尝试提取
                                text = item["entities"]
                                if re.search(r'\{[^{}]+,[^{}]+\}', text):
                                    matches = re.findall(r'\{([^{}]+?),\s*([^{}]+?)\}', text)
                                    if matches:
                                        found_entity = True
                                        for entity_span, entity_type in matches:
                                            entities.append({
                                                "Entity_span": entity_span.strip(),
                                                "Type": entity_type.strip()
                                            })
                    
                    # 如果不是JSON代码块，但直接包含实体标记
                    elif re.search(r'\{[^{}]+,[^{}]+\}', item["entities"]):
                        matches = re.findall(r'\{([^{}]+?),\s*([^{}]+?)\}', item["entities"])
                        if matches:
                            found_entity = True
                            for entity_span, entity_type in matches:
                                entities.append({
                                    "Entity_span": entity_span.strip(),
                                    "Type": entity_type.strip()
                                })
                    else:
                        # 处理没有实体标记的情况
                        print(f"条目{idx}: 没有找到实体标记")
            
            except Exception as e:
                print(f"处理条目{idx}时出错: {str(e)[:100]}")
        
        # 更新统计信息
        if entities:
            items_with_entities += 1
            entity_count += len(entities)
        else:
            empty_entities_count += 1
        
        # 添加提取的实体（即使是空列表）
        new_item["entities"] = entities
        converted_data.append(new_item)
    
    # 保存转换后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)
    
    print(f"转换完成。处理了{len(data)}条记录")
    print(f"- {items_with_entities}条有实体，共{entity_count}个实体")
    print(f"- {empty_entities_count}条没有实体")
    print(f"结果保存至: {output_file}")
    return converted_data


if __name__ == "__main__":
    input_file = r"result/zero-shot-demo/gpt-4o-mini/eeg(MNER)/eeg_results.json"
    output_file = r"result/zero-shot-demo/gpt-4o-mini/eeg(MNER)/converted_ner_results.json"
    convert_ner_format(input_file, output_file)