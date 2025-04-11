#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: bmes_decode.py

from typing import Tuple, List

#定义tag类
class Tag(object):
    def __init__(self, term, tag, begin, end):
        self.term = term # 实体跨度（实体的文本内容）
        self.tag = tag  # 类别
        self.begin = begin # 实体起始位置
        self.end = end # 实体结束位置

    def to_tuple(self):
        return tuple([self.term, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})

#bems_decode函数,接受一个包含字符和标签的列表char_label_list，并返回一个包含实体标签的列表tags。
#提取的实体文本和标签类型被封装为Tag对象，并添加到tags列表中。
def bmes_decode(char_label_list: List[Tuple[str, str]]) -> List[Tag]:
    """
    decode inputs to tags
    Args:
        char_label_list: list of tuple (word, bmes-tag)
    Returns:
        tags
    Examples:
        >>> x = [("Hi", "O"), ("Beijing", "S-LOC")]
        >>> bmes_decode(x)
        [{'term': 'Beijing', 'tag': 'LOC', 'begin': 1, 'end': 2}]
    """
    tags = []  # 存储提取出的实体标签

    idx = 0  # 当前处理的位置
    length = len(char_label_list)  # 列表的长度

    while idx < length:
        current_label = char_label_list[idx][1]
        
        if current_label.startswith("B"):  # 处理以 "B-" 开头的标签
            end = idx + 1
            while end < length and char_label_list[end][1].startswith("I"):
                end += 1
            
            # 提取实体
            entity = "".join(char_label_list[i][0] for i in range(idx, end))
            tag_type = current_label[2:]  # 获取实体类型
            tags.append(Tag(entity, tag_type, idx, end))
            
            idx = end
        else:
            idx += 1  # 如果当前标签不是 "B-" 开头，则跳过

    # tags 列表现在包含了所有提取出的实体标签
    return tags
#这段代码的主要功能是从BMES格式的字符标签列表中提取实体，并将这些实体封装为Tag对象。通过识别"B-"和"I-"标签，代码能够准确地提取出实体的文本内容、标签类型、起始位置和结束位置，并将这些信息存储在Tag对象中。最终，所有提取出的实体标签被返回，便于后续的处理和分析。