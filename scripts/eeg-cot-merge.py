import json

# 读取原始输入文件
with open("result/zero-shot-demo/eeg-demo/eeg_demo_origin_input.json", "r", encoding="utf-8") as f:
    origin_data = json.load(f)

# 读取输出结果文件
with open("result/zero-shot-demo/eeg-demo/eeg_demo2.json", "r", encoding="utf-8") as f:
    output_data = json.load(f)

# 创建合并后的数据列表
merged_data = []

for i, input_item in enumerate(origin_data):
    output_item = output_data[i]
    
    # 提取推理内容和实体输出
    reasoning_content = ""
    output_entities = []
    
    for entity in output_item["output"]:
        if "Reasoning_content" in entity:
            reasoning_content = entity["Reasoning_content"]
        else:
            output_entities.append(entity)
    
    # 创建合并后的项目
    merged_item = {
        "input": input_item,
        "Reasoning_content": reasoning_content,
        "output": output_entities
    }
    
    merged_data.append(merged_item)

# 保存合并后的数据
output_path = "result/zero-shot-demo/eeg-demo/eeg_cot.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, indent=3, ensure_ascii=False)

print(f"合并数据已保存至 {output_path}")