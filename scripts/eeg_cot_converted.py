import json
import copy

def convert_eeg_format():
    # 读取输入文件
    input_path = "result/zero-shot-demo/gpt-4o-mini/eeg(MNER)/eeg_results.json"
    output_path = "result/zero-shot-demo/gpt-4o-mini/eeg(MNER)/eeg_cot_converted.json"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        cot_data = json.load(f)
    
    # 创建新的转换后数据列表
    converted_data = []
    
    # 遍历每个条目进行格式转换
    for item in cot_data:
        new_item = {
            "content": item["content"],
            "url": item["url"]
        }
        
        # 处理entities部分
        new_entities = []
        
        if "entities" in item:
            for entity in item["entities"]:
                # 检查是否已经是目标格式
                if isinstance(entity, dict) and "Grounding_Image_Area_Description" in entity:
                    # 已经是目标格式，直接复制
                    new_entities.append(copy.deepcopy(entity))
                elif isinstance(entity, dict) and "input" in entity:
                    # COT格式，检查是否有output
                    if "output" in entity and entity["output"]:
                        for output_entity in entity["output"]:
                            if isinstance(output_entity, dict):
                                new_entities.append(copy.deepcopy(output_entity))
                    # 否则创建一个空实体数组
        
        new_item["entities"] = new_entities
        converted_data.append(new_item)
    
    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)
    
    print(f"格式转换完成，已保存到 {output_path}")

if __name__ == "__main__":
    convert_eeg_format()