import json
import random

def calculate_gmner_metrics(n, m):
    # 读取MNER和视觉接地的TP文件
    with open('result/zero-shot-no-demo/samples1500/eeg(MNER)/tp.json', 'r', encoding='utf-8') as f:
        mner_tp = json.load(f)
    
    with open('result/zero-shot-no-demo/samples1500/vg/tp.json', 'r', encoding='utf-8') as f:
        vg_tp = json.load(f)
    
    # 创建MNER TP的索引，以(context, Entity_span)为键
    mner_tp_index = {(item['context'], item['Entity_span']): item for item in mner_tp}
    
    # 创建视觉接地TP的索引，以(content, entity)为键
    vg_tp_index = {(item['content'], item['entity']): item for item in vg_tp}
    
    # 找出同时在两个TP列表中的实体
    gmner_tp_list = []
    
    for mner_key, mner_item in mner_tp_index.items():
        mner_context, mner_entity = mner_key
        
        # 在视觉接地TP中查找对应项
        vg_key = (mner_context, mner_entity)
        if vg_key in vg_tp_index:
            # 找到匹配项，添加到GMNER TP列表
            gmner_tp_list.append({
                "context": mner_context,
                "entity": mner_entity,
                "type": mner_item.get("Type", ""),
                "mner_details": mner_item,
                "vg_details": vg_tp_index[vg_key]
            })
    
    # 计算指标    
    tp = len(gmner_tp_list)
    fp = pred_entities_num - tp
    fn = true_entities_num - tp
    
    precision = tp / m if m > 0 else 0
    recall = tp / n if n > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 输出结果
    print(f"GMNER评估结果:")
    print(f"真实集实体总数(n): {n}")
    print(f"预测集实体总数(m): {m}")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"精确率(P): {precision:.4f}")
    print(f"召回率(R): {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    
    # 保存GMNER TP列表
    with open('result/zero-shot-no-demo/samples1500/gmner/tp.json', 'w', encoding='utf-8') as f:
        json.dump(gmner_tp_list, f, ensure_ascii=False, indent=4)
    
    return {
        "n": n,
        "m": m,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP_list": gmner_tp_list
    }

# 添加计算真实集和预测集实体总数的功能
def calculate_entity_counts():
    # 从已有文件中计算实体总数
    # 这里您可以根据实际情况修改文件路径和计算逻辑
    
    # 计算真实集实体总数
    true_entities_num = 0
    with open('/Users/xiaomingcheng/Downloads/论文/gmner/data/Twitter10000_v2.0/mrc/merged-test-mrc-updated-with-bndbox.json', 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    # 100个样本
    # random.seed(42)
    # random.shuffle(gold_data)
    # gold_data = gold_data[:100]
        
    # 遍历每个样本
    for item in gold_data:
        # 检查是否有entities字段
        if "entities" in item:
            # 遍历实体列表
            for entity in item["entities"]:
                # 如果实体包含Entity_span字段，增加计数
                if "entity_names" in entity:
                    true_entities_num += 1
    
    # 计算预测集实体总数
    pred_entities_num = 0
    with open('result/zero-shot-base/samples1500/gmner-no-vg-base/gmner_no_vg_results.json', 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
        
    # 遍历每个样本
    for item in pred_data:
        # 检查是否有entities字段
        if "entities" in item:
            # 遍历实体列表
            for entity in item["entities"]:
                # 如果实体包含Entity_span字段，增加计数
                if "Entity_span" in entity:
                    pred_entities_num += 1
    
    return true_entities_num, pred_entities_num

if __name__ == "__main__":
    # 计算真实集和预测集实体总数
    true_entities_num, pred_entities_num = calculate_entity_counts()
    
    # 计算GMNER指标
    metrics = calculate_gmner_metrics(true_entities_num, pred_entities_num)