import json
from tqdm import tqdm
import os
import random

def extract_entities(merged_mrc_path, results_mrc_path):
    # 读取真实集
    with open(merged_mrc_path, 'r', encoding='utf-8') as f:
        merged_mrc = json.load(f)

    # 确保每次从数据集中随机选择的是100个相同的数据
    random.seed(42)
    random.shuffle(merged_mrc)
    merged_mrc = merged_mrc[:100]

    # 读取预测集
    with open(results_mrc_path, 'r', encoding='utf-8') as f:
        results_mrc = json.load(f)

    # 提取真实集中的entities，跳过为空的实体
    real_entities = []
    for item in merged_mrc:
        for entity in item['entities']: # 不会添加空列表
            real_entities.append({
                "context": item['context'],
                "image_id": item['image_id'],
                "Type": entity['label'],
                "Entity_span": entity['entity_names'],
                "start": entity['start'],
                "end": entity['end']
            })

    # 提取预测集中的entities，跳过为空的实体
    predicted_entities = []
    for item in results_mrc:
        for entity in item['entities']:
            # print(item['url'])
            predicted_entities.append({
                "context": item['content'],
                # "image_id": item['url'],
                "Type": entity['Type'],
                "Entity_span": entity['Entity_span']
            })

    return real_entities, predicted_entities, merged_mrc, results_mrc

##定义指标计算方法
def calculate_metrics(real_entities, predicted_entities, merged_mrc, results_mrc):
    tp = 0
    fp = 0
    fn = 0

    tp_list = []
    fp_list = []
    fn_list = []

    # 按类型分组真实实体
    real_entities_by_type = {label: [] for label in ['PER', 'LOC', 'ORG', 'MISC']}
    for entity in real_entities:
        real_entities_by_type[entity['Type']].append(entity)

    # 逐个比较预测实体
    for pred_entity in predicted_entities:
        entity_type = pred_entity['Type']

        if entity_type in real_entities_by_type:
            matched = False
            for real_entity in real_entities_by_type[entity_type]:
                if pred_entity['context'] == real_entity['context'] and pred_entity['Entity_span'] == real_entity['Entity_span']:
                        tp += 1
                        tp_list.append(pred_entity)
                        real_entities_by_type[entity_type].remove(real_entity)  # 从真实实体中移除
                        matched = True
                        break
                    
            if not matched:
                fp += 1
                fp_list.append(pred_entity)
    # 计算 FN
    for entities in real_entities_by_type.values():
        for real_entity in entities:
            fn += 1
            fn_list.append(real_entity)


    # 计算评估指标
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (recall + precision) if (recall + precision) > 0 else 0

    return tp, fp, fn, recall, precision, f1, tp_list, fp_list, fn_list

def save_results(tp_list, fp_list, fn_list):
    # 加一个路径变量
    path = 'result/zero-shot-demo/gpt-4o-mini/ner/origin-cot/'

    with open(path + 'tp.json', 'w', encoding='utf-8') as f:
        json.dump(tp_list, f, ensure_ascii=False, indent=4)

    with open(path + 'fp.json', 'w', encoding='utf-8') as f:
        json.dump(fp_list, f, ensure_ascii=False, indent=4)

    with open(path + 'fn.json', 'w', encoding='utf-8') as f:
        json.dump(fn_list, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    real_entities, predicted_entities, merged_mrc, results_mrc = extract_entities(
        '/Users/xiaomingcheng/Downloads/论文/gmner/data/Twitter10000_v2.0/mrc/merged-test-mrc-updated.json',
        'result/zero-shot-demo/gpt-4o-mini/ner/origin-cot/converted_ner_results.json'
    )

    tp, fp, fn, recall, precision, f1, tp_list, fp_list, fn_list = calculate_metrics(real_entities, predicted_entities, merged_mrc, results_mrc)

    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"TP+FN: {tp+fn}")
    print(f"TP+FP: {tp+fp}")

    save_results(tp_list, fp_list, fn_list)