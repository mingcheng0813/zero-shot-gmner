#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: msra2mrc.py

import os
from bio_decode import bmes_decode
import json

#convert_file函数
def convert_file(input_file, output_file, tag2query_file):
    """
    Convert bio data to MRC format
    """
    origin_count = 0
    new_count = 0
    tag2query = json.load(open(tag2query_file,encoding='utf-8'))
    mrc_samples = []
#读取输入文件，并逐行读取，跳过空行，并将每一行的JSON数据加载到data变量中
    with open(input_file, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:    # 空行处理
                continue
            data = json.loads(line)
#处理每一行数据，获取text和label
            text = data["text"]
            labels = data["label"]
            image_id = data["image_id"]

            # 构造元组列表
            char_label_list = [(char, label) for char, label in zip(text, labels)]

            # 处理标签
            tags = bmes_decode(char_label_list=char_label_list)
            # print(line)

            # src, labels= line.split("\t")
            tags = bmes_decode(char_label_list=char_label_list)
#生成MRC样本
#遍历标签列表，将每个标签转换为MRC样本，并将样本添加到mrc_samples列表中
            for label, query in tag2query.items():
                mrc_samples.append(
                    {   
                        "image_id": image_id,    # 图像id
                        "context": " ".join(text),
                        "entity_label": label,
                        "end_position": [tag.end-1 for tag in tags if tag.tag == label],
                        "start_position": [tag.begin for tag in tags if tag.tag == label],
                        "query": query
                    }
                )
                
                new_count += 1
            origin_count += 1
#写入输出文件
    json.dump(mrc_samples, open(output_file, "w", encoding='utf-8'), ensure_ascii=False, sort_keys=True, indent=2)
    print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")



# convert_file(input_file, output_file, tag2query_file)


if __name__ == '__main__':
    input_file = r"F:\研究生文件\自己的论文\GMNER\GMNER-XMC\Code\gmner-1\data\Twitter10000_v2.0\txt\dev.jsonl"
    output_file = r"F:\研究生文件\自己的论文\GMNER\GMNER-XMC\Code\gmner-1\data\Twitter10000_v2.0\mrc\dev-mrc.json"
    tag2query_file = r'F:\研究生文件\自己的论文\GMNER\GMNER-XMC\Code\gmner-1\data\Twitter10000_v2.0\queries\queries.json' 
    convert_file(input_file,output_file,tag2query_file)
#这段代码的主要功能是将BIO格式的数据转换为MRC（Machine Reading Comprehension）格式。主要处理的是带有文本和标签的JSON数据文件。
#它通过读取输入的JSON文件，解析每一行的文本和标签，并使用bmes_decode函数将标签解码为实体标签。然后，根据标签到查询的映射文件，生成MRC格式的样本，并将其保存到指定的输出文件中。最终，代码打印出转换的样本数量。