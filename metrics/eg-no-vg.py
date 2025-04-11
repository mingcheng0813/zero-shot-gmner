import json
import os

def calculate_entity_grounding_metrics(pred_file, gold_file, output_dir=None):
    """
    计算实体级别的接地性能指标
    
    Args:
        pred_file: 预测结果文件路径
        gold_file: 真实标记文件路径
        output_dir: 输出结果目录（可选）
    """
    # 读取文件
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    # 构建真实集的索引，使用context作为键
    gold_index = {}
    for item in gold_data:
        if "context" in item:
            gold_index[item["context"]] = item
    
    # 初始化计数器和结果列表
    tp = 0  # 真正例
    fp = 0  # 假正例
    fn = 0  # 假负例
    
    total_pred_entities = 0
    matched_contents = 0
    
    tp_list = []
    fp_list = []
    fn_list = []
    
    # 第一遍循环：计算TP和FP，并记录已匹配的金标准实体
    for pred_item in pred_data:
        content = pred_item.get("content", "")
        if not content or content not in gold_index:
            # 对于无法匹配到金标准的内容，将其所有实体标为FP
            for entity in pred_item.get("entities", []):
                if "Entity_span" in entity:
                    total_pred_entities += 1
                    # 检查是否有接地描述
                    grounding_desc = entity.get("Grounding_Image_Area_Description", "")
                    if grounding_desc and grounding_desc != "None":
                        fp += 1
                        fp_list.append({
                            "content": content,
                            "entity": entity.get("Entity_span", ""),
                            "pred_grounding": grounding_desc,
                            "reason": "未找到对应内容"
                        })
            continue
        
        matched_contents += 1
        gold_item = gold_index[content]
        
        # 获取金标准实体
        gold_entities_dict = {}
        for gold_entity in gold_item.get("entities", []):
            if "entity_names" in gold_entity:
                entity_name = gold_entity["entity_names"]
                if entity_name not in gold_entities_dict:
                    gold_entities_dict[entity_name] = []
                gold_entities_dict[entity_name].append(gold_entity)
        
        # 记录已处理的金标准实体ID
        processed_gold_entities = set()
        
        # 处理预测的每个实体
        for pred_entity in pred_item.get("entities", []):
            total_pred_entities += 1
            
            # 获取实体名和接地描述
            entity_span = pred_entity.get("Entity_span", "")
            grounding_desc = pred_entity.get("Grounding_Image_Area_Description", "")
            pred_groundable = bool(grounding_desc) and grounding_desc != "None"
            
            # 在金标准中查找匹配的实体
            if entity_span in gold_entities_dict:
                # 找到匹配的且未处理的金标准实体
                matched = False
                for gold_entity in gold_entities_dict[entity_span]:
                    gold_entity_id = id(gold_entity)
                    if gold_entity_id not in processed_gold_entities:
                        matched = True
                        processed_gold_entities.add(gold_entity_id)
                        
                        # 获取金标准的接地性
                        # gold_groundable = gold_entity.get("groundability", "") == "Groundable"
                        gold_groundable = bool(gold_entity.get("bndbox", None))
                        
                        if pred_groundable and gold_groundable:
                            # 预测为可接地且真实可接地 -> TP
                            tp += 1
                            tp_list.append({
                                "content": content,
                                "entity": entity_span,
                                "pred_grounding": grounding_desc
                            })
                        elif not pred_groundable and not gold_groundable:
                            # 预测为不可接地且真实也不可接地 -> TP
                            tp += 1
                            tp_list.append({
                                "content": content,
                                "entity": entity_span,
                                "pred_grounding": grounding_desc
                            })
                        elif pred_groundable and not gold_groundable:
                            # 预测为可接地但真实不可接地 -> FP
                            fp += 1
                            fp_list.append({
                                "content": content,
                                "entity": entity_span,
                                "pred_grounding": grounding_desc,
                                "reason": "实际不可接地"
                            })
                        elif not pred_groundable and gold_groundable:
                            # 预测为不可接地但真实可接地 -> FN
                            fp += 1
                            fp_list.append({
                                "content": content,
                                "entity": entity_span,
                                "reason": "未提供接地描述但实际可接地"
                            })
                               
                        break
                
                # 预测了多个同名实体
                if not matched:
                    # if pred_groundable:
                    fp += 1
                    fp_list.append({
                        "content": content,
                        "entity": entity_span,
                        "pred_grounding": grounding_desc,
                        "reason": "预测了多个同名实体"
                    })
            else:
                # 在金标准中找不到对应实体
                # if pred_groundable:
                fp += 1
                fp_list.append({
                    "content": content,
                    "entity": entity_span,
                    "pred_grounding": grounding_desc,
                    "reason": "过度预测的不存在实体"
                })
        
        # 检查遗漏的金标准实体（假负例）
        for entity_name, gold_entities in gold_entities_dict.items():
            for gold_entity in gold_entities:
                gold_entity_id = id(gold_entity)
                if gold_entity_id not in processed_gold_entities:
                    fn += 1
                    fn_list.append({
                        "content": content,
                        "entity": entity_name,
                        "reason": "未预测"
                    })
    
    # 计算评估指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 输出结果
    print(f"===== 实体接地评估结果 =====")
    print(f"总预测实体数: {total_pred_entities}")
    print(f"匹配内容条目数: {matched_contents}")
    print(f"TP (真正例): {tp}")
    print(f"FP (假正例): {fp}")
    print(f"FN (假负例): {fn}")
    print(f"TP + FP: {tp + fp} (应等于所有预测的实体数)")
    print(f"精确率(P): {precision:.4f}")
    print(f"召回率(R): {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print(f"===========================")
    
    # 保存结果到文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存评估结果摘要
        summary = {
            "total_pred_entities": total_pred_entities,
            "matched_contents": matched_contents,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
        
        
        # 保存详细结果
        with open(os.path.join(output_dir, 'tp.json'), 'w', encoding='utf-8') as f:
            json.dump(tp_list, f, indent=4, ensure_ascii=False)
        
        with open(os.path.join(output_dir, 'fp.json'), 'w', encoding='utf-8') as f:
            json.dump(fp_list, f, indent=4, ensure_ascii=False)
        
        with open(os.path.join(output_dir, 'fn.json'), 'w', encoding='utf-8') as f:
            json.dump(fn_list, f, indent=4, ensure_ascii=False)
        
        print(f"详细结果已保存至: {output_dir}")
    
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP_list": tp_list,
        "FP_list": fp_list,
        "FN_list": fn_list
    }

def main():
    # 文件路径
    pred_file = "result/zero-shot-demo/eeg(MNER)/eeg_results.json"
    gold_file = "/Users/xiaomingcheng/Downloads/论文/gmner/data/Twitter10000_v2.0/mrc/merged-test-mrc-updated-with-bndbox.json"
    
    # 输出目录
    output_dir = "result/zero-shot-demo/eg_no_vg"
    
    # 计算指标
    calculate_entity_grounding_metrics(pred_file, gold_file, output_dir)

if __name__ == "__main__":
    main()