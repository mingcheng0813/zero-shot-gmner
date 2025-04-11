import json
import re

def parse_output_results(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for item in data:
        if 'entities' in item and len(item['entities']) > 1:
            output_text = item['entities'][1].get('output', '')
            
            # 使用正则表达式删除 {实体名称,实体类型} 格式的标记
            # 将其替换为仅保留实体名称
            clean_text = re.sub(r'\{([^,]+),\w+\}', r'\1', output_text)
            
            results.append({
                'original_output': output_text,
                'cleaned_text': clean_text,
                'url': item.get('url', '')
            })
    
    return results

def main():
    file_path = '/Users/xxxxxx/Downloads/论文/gmner/result/NER/ner_results.json'
    results = parse_output_results(file_path)
    
    # 打印结果
    for i, result in enumerate(results):
        print(f"条目 {i+1}:")
        print(f"原始输出: {result['original_output']}")
        print(f"清理后文本: {result['cleaned_text']}")
        print(f"URL: {result['url']}")
        print("-" * 50)
    
    # 将结果保存到新文件
    output_file = '/Users/xxxxxx/Downloads/论文/gmner/result/NER/cleaned_outputs.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"结果已保存到: {output_file}")

if __name__ == '__main__':
    main()