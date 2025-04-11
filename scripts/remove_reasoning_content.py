import json
import os

def clean_ner_data_preserve_structure(input_file, output_file):
    """从NER数据文件中移除reasoning_content，保留其他结构"""
    
    print(f"正在处理文件: {input_file}")
    
    try:
        # 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 处理每个条目
        for item in data:
            if "entities" in item:
                # 直接删除reasoning_content键
                item["entities"] = [entity for entity in item["entities"] 
                                   if "reasoning_content" not in entity]
                
        # 保存清理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print(f"处理完成! 清理后的数据已保存到: {output_file}")
        print(f"原始文件大小: {os.path.getsize(input_file) / (1024*1024):.2f} MB")
        print(f"清理后文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    input_file = "result/zero-shot-demo/ner-demo/ner_demo_correct.json"
    output_file = "/Users/xxxxxx/Downloads/论文/gmner/result/zero-shot-demo/ner-demo/ner_demo_correct_no_reaning.json"
    
    clean_ner_data_preserve_structure(input_file, output_file)