import json

def convert_eeg_results(input_file, output_file):
    """
    将EEG结果文件转换为简化格式
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换格式
    converted_data = []
    for item in data:
        # 创建新的格式
        new_item = {
            "entities": [],
            "Image_URL": item.get("url", "")
        }
        
        # 处理实体
        for entity in item.get("entities", []):
            if "Entity_span" in entity:
                new_entity = {
                    "Entity_span": entity["Entity_span"],
                    "Grounding_Image_Area_Description": entity.get("Grounding_Image_Area_Description", "")
                }
                new_item["entities"].append(new_entity)
        
        # 添加到结果列表
        converted_data.append(new_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=4)
    
    print(f"转换完成！共处理 {len(data)} 条数据，输出到 {output_file}")

if __name__ == "__main__":
    input_file = "result/zero-shot-demo/gpt-4o-mini/eeg(MNER)/eeg_cot_converted.json"
    output_file = "result/zero-shot-demo/gpt-4o-mini/vg/vg_input.json"
    convert_eeg_results(input_file, output_file)