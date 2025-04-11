import json

def convert_ner_to_eeg(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        ner_data = json.load(f)
    
    # 转换格式
    eeg_data = []
    for item in ner_data:
        if "content" in item and "url" in item and "entities" in item:
            eeg_item = {
                "Text_sentence": item["content"],
                "NER_results": item["entities"],
                "Image_URL": item["url"]
            }
            eeg_data.append(eeg_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eeg_data, f, ensure_ascii=False, indent=3)
    
    print(f"转换完成！共转换{len(eeg_data)}条数据。")

## 只用于示例的转换
def convert_demo_to_eeg(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        ner_data = json.load(f)
    
    # 转换格式
    eeg_data = []
    for item in ner_data:
        if "content" in item and "url" in item and "entities" in item:
            eeg_item = {
                "Text_sentence": item["content"],
                "NER_results": item["entities"],
                "iamge_id": item["image_id"]
            }
            eeg_data.append(eeg_item)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eeg_data, f, ensure_ascii=False, indent=3)
    
    print(f"转换完成！共转换{len(eeg_data)}条数据。")

if __name__ == "__main__":
    input_file = "result/zero-shot-demo/gpt-4o-mini/ner/origin-cot/converted_ner_results.json"
    output_file = "result/zero-shot-demo/gpt-4o-mini/eeg(MNER)/eeg-cot/eeg_input.json"
    convert_ner_to_eeg(input_file, output_file)
    # convert_demo_to_eeg(input_file, output_file)