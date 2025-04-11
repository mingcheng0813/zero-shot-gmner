import json
import os
from collections import defaultdict

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)
    box格式: [xmin, ymin, xmax, ymax]
    """
    # 确保box1和box2是数值而非字符串
    # try:
    box1 = [float(box1['xmin']), float(box1['ymin']), float(box1['xmax']), float(box1['ymax'])]
    box2 = [float(box2['xmin']), float(box2['ymin']), float(box2['xmax']), float(box2['ymax'])]
    print(f"box1: {box1}, box2: {box2}")   
    # except (ValueError, TypeError):
    #     box1 = [float(box1['xmin'].split(',')[0].strip()), float(box1['ymin'].split(',')[0].strip()), float(box1['xmax'].split(',')[0].strip()), float(box1['ymax'].split(',')[0].strip())]
    #     box2 = [float(box2['xmin']), float(box2['ymin']), float(box2['xmax']), float(box2['ymax'])]
        # print(f"Invalid box format: {box1}, {box2}")
    
    # 计算交集的坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # 如果没有交集，返回0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    
    return iou

def evaluate_visual_grounding(gold_file, pred_file, iou_threshold=0.5):
    """
    评估视觉接地性能
    
    Args:
        gold_file: 真实集文件路径
        pred_file: 预测集文件路径
        iou_threshold: IoU阈值，默认0.5
        
    Returns:
        包含各项指标的字典
    """
    # 读取文件
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # 构建真实集的索引，使用context作为键
    gold_index = {}
    for item in gold_data:
        if "context" in item:
            gold_index[item["context"]] = item
    
    # 初始化计数器
    tp = 0  # 真正例
    fp = 0  # 假正例
    fn = 0  # 假负例
    
    tp_list = []
    fp_list = []
    fn_list = []
    
    # 对每个预测样本进行评估
    for pred_item in pred_data:
        content = pred_item.get("content", "")
        
        # 查找对应的金标准样本
        if content in gold_index:
            gold_item = gold_index[content]
            gold_entities = gold_item.get("entities", [])
            pred_entities = pred_item.get("entities", [])
            
            # 构建金标准实体的索引，使用entity_names作为键
            gold_entities_dict = defaultdict(list)
            for entity in gold_entities:
                # if "entity_names" in entity and "bndbox" in entity: # 未添加不可接地实体
                if "entity_names" in entity: # 未添加不可接地实体

                    gold_entities_dict[entity["entity_names"]].append(entity)
            
            # 记录已处理的金标准实体
            processed_gold_entities = set()
            
            # 检查每个预测实体
            for pred_entity in pred_entities:
                # 跳过pred_entity为字符串的样本
                if isinstance(pred_entities, str):
                    print(f"跳过pred_entity为字符串的样本: {pred_entity}")
                    continue
                print(f"pred_entity: {pred_entity}")
                entity_span = pred_entity.get("Entity_span", "")
                pred_bndboxes = pred_entity.get("bndbox", [])
                
                # 如果预测的边界框是字典而非列表，将其转换为列表
                if isinstance(pred_bndboxes, dict):
                    pred_bndboxes = [pred_bndboxes]
                
                # 在金标准实体中寻找对应实体
                if entity_span in gold_entities_dict:
                    # 检查是否有任何一个预测边界框与真实边界框的IoU > 阈值
                    matched = False
                    matched_gold_entity = None
                    
                    for gold_entity in gold_entities_dict[entity_span]:
                        # 如果不是同一实体则跳过，不用计算指标
                        if entity_span != gold_entity.get("entity_names", ""):
                            continue
                        
                        # 同一实体的情况
                        gold_entity_id = id(gold_entity)
                        if gold_entity_id in processed_gold_entities:
                            continue
                            
                        gold_bndboxes = gold_entity.get("bndbox", [])
                        
                        # 如果真实边界框为空，则跳过该实体
                        if not gold_bndboxes and not pred_bndboxes:
                            matched = True # 预测正确
                            matched_gold_entity = gold_entity
                            processed_gold_entities.add(gold_entity_id)
                            tp += 1
                            tp_list.append({
                                "content": content,
                                "entity": entity_span,
                                "pred_bndbox": pred_bndboxes,
                                "gold_bndbox": gold_bndboxes,
                                "reason": "真实和预测都不能接地"
                            })
                            break
                           
                        # 对每个预测边界框，检查是否与任意一个真实边界框的IoU > 阈值
                        for pred_box in pred_bndboxes:
                            for gold_box in gold_bndboxes:
                                iou = calculate_iou(pred_box, gold_box)
                                print(iou)
                                if iou > iou_threshold:
                                    matched_gold_entity = gold_entity
                                    processed_gold_entities.add(gold_entity_id)
                                    tp += 1
                                    tp_list.append({
                                        "content": content,
                                        "entity": entity_span,
                                        "pred_bndbox": pred_bndboxes,
                                        "gold_bndbox": matched_gold_entity.get("bndbox", [])
                                    })
                                    matched = True # 预测正确
                                    break
                            if matched:
                                break
                    if not matched:
                        fp += 1
                        fp_list.append({
                            "content": content,
                            "entity": entity_span,
                            "pred_bndbox": pred_bndboxes,
                            "reason": "没有与真实边界框匹配的预测边界框"
                        })

                else:
                    # 预测了不存在的实体
                    fp += 1
                    fp_list.append({
                        "content": content,
                        "entity": entity_span,
                        "pred_bndbox": pred_bndboxes,
                        "reason": "预测实体未存在于金标准"
                    })
            
            # 查找漏掉的实体（假负例）
            for entity_name, gold_entity_list in gold_entities_dict.items():
                for gold_entity in gold_entity_list:
                    gold_entity_id = id(gold_entity)
                    if gold_entity_id not in processed_gold_entities:
                        fn += 1
                        fn_list.append({
                            "content": content,
                            "entity": entity_name,
                            "gold_bndbox": gold_entity.get("bndbox", []),
                            "reason": "漏掉的实体"
                        })
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
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
    gold_file = "/Users/xiaomingcheng/Downloads/论文/gmner/data/Twitter10000_v2.0/mrc/merged-test-mrc-updated-with-bndbox.json"
    pred_file = "result/zero-shot-no-demo/samples1500/vg/vg_results_final_clean.json"
    
    # 输出目录
    output_dir = "result/zero-shot-no-demo/samples1500/vg"
    
    # 计算指标
    metrics = evaluate_visual_grounding(gold_file, pred_file)
    
    # 打印结果
    print(f"TP: {metrics['TP']}")
    print(f"FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}")
    print(f"TP + FP: {metrics['TP'] + metrics['FP']} (应等于所有预测的实体数)")
    print(f"精确率(P): {metrics['Precision']:.4f}")
    print(f"召回率(R): {metrics['Recall']:.4f}")
    print(f"F1值: {metrics['F1']:.4f}")
    
    # 保存详细结果
    with open(os.path.join(output_dir, 'tp.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics['TP_list'], f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_dir, 'fp.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics['FP_list'], f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_dir, 'fn.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics['FN_list'], f, ensure_ascii=False, indent=4)
    
    print(f"详细结果已保存到 {output_dir} 目录下")

if __name__ == "__main__":
    main()