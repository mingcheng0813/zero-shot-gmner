# -*- coding: utf-8 -*-
# @Time    : 2020/12/01 11:47
# @Author  : liuwei
# @File    : format_convert.py

"""
convert data format
"""
import json
import os
import time
from tqdm import tqdm, trange

#BMES_t0_json函数
def BMES_to_json(bmes_file, json_file):
    """
    convert bmes format file to json file, json file has two key, including text and label
    Args:
        bmes_file:
        json_file:
    :return:
    """
#读取BMES文件，打开BMES文件并读取所有行，计算总行数，并初始化一些列表用于存储文本、标签和图像ID。
    texts = []
    with open(bmes_file, 'r', encoding='utf-8') as f:

        lines = f.readlines()
        total_line_num = len(lines)
        line_iter = trange(total_line_num)
        words = []
        labels = []
        image_ids = []
#处理每一行，如果是空行，则将当前行的文本、标签和图像ID存储到列表中，并将列表清空；如果是非空行，则将当前行的文本和标签分别存储到列表中。
        for idx in line_iter:
            line = lines[idx]
            line = line.strip()

            if not line: # 空行
                assert len(words) == len(labels), (len(words), len(labels))
                sample = {}
                sample['text'] = words
                sample['label'] = labels
                sample['image_id'] = image_ids.pop(0)
                texts.append(json.dumps(sample, ensure_ascii=False))

                words = []
                labels = []
                image_ids = []

            else: # 非空行
                if "IMGID" in line:
                    image_ids.append(line[6:])                       
                else:
                    items = line.split()
                    words.append(items[0])
                    labels.append(items[1])
#写入JSON文件，将处理好的文本、标签和图像ID存储到JSON文件中。
    with open(json_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write("%s\n"%(text))
#调用函数，将BMES文件转换为JSON文件。
absolute_path = r"F:\研究生文件\自己的论文\GMNER\GMNER-XMC\Code\gmner-1\data\Twitter10000_v2.0\txt"
BMES_to_json(os.path.join(absolute_path, "dev.txt"),os.path.join(absolute_path,"dev.json"))

#这段代码的主要功能是将BMES格式的文件转换为JSON格式，并保存到指定的文件中。它通过逐行读取BMES文件，解析每一行的内容，并将解析后的数据存储为JSON格式。最终，所有JSON对象被写入到一个新的文件中。
