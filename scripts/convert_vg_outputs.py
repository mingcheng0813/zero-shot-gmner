import json
import re
import os
from tqdm import tqdm

def extract_entities_from_raw_output(raw_output):
    """从原始输出字符串中提取实体信息"""
    entities = []
    
    # 如果为空，返回空列表
    if not raw_output or not isinstance(raw_output, str):
        return entities
    
    # 尝试提取JSON部分
    json_match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
    if not json_match:
        return entities
    
    json_str = json_match.group(1).strip()
    
    # 尝试解析JSON
    try:
        parsed_entities = json.loads(json_str)
        
        # 处理不同的实体格式
        if isinstance(parsed_entities, list):
            for entity in parsed_entities:
                if isinstance(entity, dict) and "Entity_span" in entity and entity["Entity_span"]:
                    clean_entity = {
                        "Entity_span": entity.get("Entity_span", ""),
                        "bndbox": []
                    }
                    
                    # 处理边界框
                    if "bndbox" in entity and entity["bndbox"]:
                        bndboxes = entity["bndbox"]
                        if isinstance(bndboxes, list):
                            for bbox in bndboxes:
                                if isinstance(bbox, dict):
                                    # 确保所有坐标都是字符串
                                    clean_bbox = {}
                                    has_all_coords = True
                                    
                                    for key in ["xmin", "ymin", "xmax", "ymax"]:
                                        if key in bbox and bbox[key]:
                                            # 处理可能包含逗号的坐标
                                            value = str(bbox[key])
                                            if "," in value:
                                                value = value.split(",")[0]
                                            # 处理带引号的数字
                                            if value.startswith('"') and value.endswith('"'):
                                                value = value[1:-1]
                                            # 验证是否是有效值
                                            try:
                                                float(value)  # 验证数字格式
                                                clean_bbox[key] = value
                                            except ValueError:
                                                has_all_coords = False
                                        else:
                                            has_all_coords = False
                                    
                                    # 只有当所有坐标都存在时才添加边界框
                                    if has_all_coords and len(clean_bbox) == 4:
                                        clean_entity["bndbox"].append(clean_bbox)
                    
                    # 只添加有效实体
                    if clean_entity["Entity_span"]:
                        entities.append(clean_entity)
    except Exception as e:
        print(f"JSON解析错误: {e}")
        # 尝试使用正则表达式进行更灵活的解析
        try:
            entity_matches = re.finditer(r'{\s*"Entity_span"\s*:\s*"([^"]*)"', json_str)
            for match in entity_matches:
                entity_name = match.group(1)
                if entity_name:
                    entities.append({
                        "Entity_span": entity_name,
                        "bndbox": []
                    })
        except Exception as e2:
            print(f"正则解析也失败: {e2}")
    
    return entities

def validate_entity(entity):
    """验证实体格式，确保每个边界框都有完整的坐标"""
    if not isinstance(entity, dict) or "Entity_span" not in entity or not entity["Entity_span"]:
        return None
    
    valid_entity = {
        "Entity_span": entity["Entity_span"],
        "bndbox": []
    }
    
    # 验证边界框
    if "bndbox" in entity and isinstance(entity["bndbox"], list):
        for bbox in entity["bndbox"]:
            if not isinstance(bbox, dict):
                continue
            
            # 确保所有必要的坐标都存在
            valid_bbox = {}
            has_all_coords = True
            
            for key in ["xmin", "ymin", "xmax", "ymax"]:
                if key in bbox and bbox[key]:
                    valid_bbox[key] = str(bbox[key])
                else:
                    has_all_coords = False
                    break
            
            # 只添加完整的边界框
            if has_all_coords and len(valid_bbox) == 4:
                valid_entity["bndbox"].append(valid_bbox)
    
    return valid_entity

def process_outputs(input_file, content_list, url_list):
    """处理输出文件并返回结构化结果"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            outputs = json.load(f)
    except Exception as e:
        print(f"读取输入文件时出错: {e}")
        return []
    
    results = []
    
    # 确保三个列表长度一致
    min_length = min(len(outputs), len(content_list), len(url_list))
    print(f"处理 {min_length} 个样本...")
    
    for i in tqdm(range(min_length), desc="处理视觉接地输出"):
        output = outputs[i]
        content = content_list[i] if i < len(content_list) else ""
        url = url_list[i] if i < len(url_list) else ""
        
        # 提取实体并验证
        raw_output = output.get("raw_output", "")
        extracted_entities = extract_entities_from_raw_output(raw_output)
        
        # 验证和修复实体格式
        valid_entities = []
        for entity in extracted_entities:
            valid_entity = validate_entity(entity)
            if valid_entity:
                valid_entities.append(valid_entity)
        
        # 创建结果对象
        result = {
            "content": content,
            "entities": valid_entities,
            "url": url
        }
        
        results.append(result)
    
    return results

def get_content_and_url_list():
    """从最终结果文件中获取内容和URL列表"""
    try:
        with open("result/zero-shot-no-demo/samples1500/vg/vg_results_final.json", 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        
        content_list = []
        url_list = []
        
        for item in final_results:
            if isinstance(item, dict):
                content_list.append(item.get("content", ""))
                url_list.append(item.get("url", ""))
        
        print(f"找到 {len(content_list)} 条内容和URL")
        return content_list, url_list
    except Exception as e:
        print(f"获取内容和URL列表时出错: {e}")
        return [], []

def verify_results(results):
    """最终验证结果格式"""
    valid_results = []
    anomalies = 0
    
    for result in results:
        # 检查基本字段
        if not isinstance(result, dict) or "content" not in result or "url" not in result:
            anomalies += 1
            continue
            
        # 重建结果对象
        clean_result = {
            "content": result["content"],
            "entities": [],
            "url": result["url"]
        }
        
        # 验证实体
        if "entities" in result and isinstance(result["entities"], list):
            for entity in result["entities"]:
                valid_entity = validate_entity(entity)
                if valid_entity:
                    clean_result["entities"].append(valid_entity)
        
        valid_results.append(clean_result)
    
    print(f"验证完成: 发现并修复了 {anomalies} 个异常结果")
    return valid_results

def main():
    # 设置输入和输出文件路径
    input_file = "result/zero-shot-no-demo/samples1500/vg/vg_outputs_all.json"
    output_file = "result/zero-shot-no-demo/samples1500/vg/vg_results_final_clean.json"
    backup_file = "result/zero-shot-no-demo/samples1500/vg/vg_results_final_backup.json"
    
    # 备份现有文件
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f_in:
                with open(backup_file, 'w', encoding='utf-8') as f_out:
                    f_out.write(f_in.read())
            print(f"已备份现有结果文件到 {backup_file}")
        except Exception as e:
            print(f"备份文件时出错: {e}")
    
    # 获取内容和URL列表
    content_list, url_list = get_content_and_url_list()
    
    # 处理输出
    results = process_outputs(input_file, content_list, url_list)
    
    # 最终验证
    verified_results = verify_results(results)
    
    # 写入结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(verified_results, f, ensure_ascii=False, indent=4)
        print(f"处理完成! 已保存结果到 {output_file}")
        print(f"样本总数: {len(verified_results)}")
    except Exception as e:
        print(f"保存结果文件时出错: {e}")

if __name__ == "__main__":
    main()