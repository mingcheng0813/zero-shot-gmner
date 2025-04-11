import os
import sys
import json
import re
import random
import time
from tqdm import tqdm
import requests
import pandas as pd


class NerProcessor:
    """命名实体识别处理器：负责处理NER任务，使用QwQ API进行推理"""
    
    def __init__(self, config=None):
        """初始化NER处理器"""
        self.config = config or {
            'prompts_file_path': 'prompts/zero-shot-demo/实体类型验证与接地/eeg_英文.md',
            'eeg_input_path': 'result/zero-shot-demo/eeg-demo/eeg_demo_input.json',
            'output_path': 'result/zero-shot-demo/eeg-demo/eeg_demo.json',
            'picture_folder_path': '/Users/xxxxxx/Downloads/论文/gmner/data/IJCAI2019_data',
            'model_id': 'deepseek-ai/DeepSeek-R1',  # 使用QwQ模型
            'api_url': 'https://api.siliconflow.cn/v1/chat/completions',
            'api_key': 'xxxxxxxx'
            ''
            '',  # 需要替换为实际的API密钥
            'temperature': 0.7,
            'max_tokens': 8192
        }   
    
    def load_prompts(self, file_path):
        """加载提示内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"加载提示文件时出错: {e}")
            return "请识别文本中的命名实体"
    # 找到picture_folder路径下以及所有子目录下的同名 image_id文件
    def find_dir_name(self, dir_, file_name):
        for root, dirs, files in os.walk(dir_):
            if file_name in files:
                return os.path.join(root, file_name)

            # 递归子目录
            for dir_name in dirs:
                file_path = self.find_dir_name(os.path.join(root, dir_name), file_name)
                if file_path is not None:
                    return dir_name

        return None

    ##获取图片的url
    def image2prompt(self, picture_folder, image_id):
        dir_name = self.find_dir_name(picture_folder, f"{image_id}.jpg")
        return f'http://feign1.weisanju.fun/dataset/{dir_name}/{image_id}.jpg'
   
    def load_cluster_samples(self, eeg_input_path):
        """
        加载聚类样本数据作为模型输入
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            处理后的样本列表，每个样本包含图像URL和文本内容
        """
        print(f"正在加载聚类样本数据: {eeg_input_path}")
        with open(eeg_input_path, 'r', encoding='utf-8') as f:
            inputs = json.load(f)

        for id, input in enumerate(inputs):
            input = json.dumps(input) if isinstance(input, dict) else str(input)
            
            # 初始化结果列表
            results = []
            
            # 构建与mrc2prompt相同格式的输出
            image_id = input['image_id']
            text = input['text']
                            
            item_info = {
                "context": text,
                "image_id": image_id,
                "url": self.image2prompt('/Users/xxxxxx/Downloads/论文/gmner/data/IJCAI2019_data', image_id)
            }
            
            results.append((item_info))
            
            print(f"已加载 {len(results)} 条聚类样本数据")
            return results
            
    
    def qwq_api_call(self, system_prompt, user_text, max_retries=5):
        """使用QwQ API进行推理"""
        url = self.config['api_url']
        
        payload = {
            "model": self.config['model_id'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            "stream": False,
            "max_tokens": self.config['max_tokens'],
            "stop": None,
            "temperature": self.config['temperature'],
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }
        
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        # 实现简单的重试机制
        for attempt in range(max_retries):
            try:
                response = requests.request("POST", url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()  # 检查HTTP错误
                result = response.json()
                print(result)
                
                # 简化日志输出
                print(f"API响应状态码: {response.status_code}, 响应ID: {result.get('id', 'unknown')}")
                
                # 根据QwQ API的响应格式提取内容
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                reasoning_content = result.get('choices', [{}])[0].get('message', {}).get('reasoning_content', '')
                
                return content, reasoning_content
                
            except requests.exceptions.RequestException as e:
                print(f"API请求失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"等待5秒后重试...")
                    time.sleep(5)
                else:
                    return f"API请求错误: {str(e)}", None
                    
            except (KeyError, IndexError, ValueError) as e:
                print(f"解析API响应失败: {e}")
                return f"响应解析错误: {str(e)}", None
    
    def process_single_sample(self, item, prompts):
        """处理单个样本"""
        
        # 使用QwQ API进行推理
        try:
            content, reasoning_content = self.qwq_api_call(prompts, item["context"])
            # print(content)
            # print(reasoning_content)
            
            entities = self.extract_entities(content)
            if not entities:
                entities = content
                print(f"{item['image_id']}未能通过```json```格式提取实体，尝试直接解析")
            
            return {
                "content": item['context'],
                "entities": entities,
                "reasoning_content": reasoning_content,
                "image_id": item['image_id']
            }
        
        except Exception as e:
            print(f"处理样本 {item['image_id']} 时出错: {e}")
            return {
                "content": item['context'],
                "entities": "处理失败",
                "image_id": item['image_id'],
            }
    
    def extract_entities(self, output):
        """从模型输出中提取实体"""
        entities = None
        
        # 提取 ```json ... ``` 格式
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', output)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                # 首先尝试完整解析
                entities = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON解析失败，错误: {e}")               
        
        # 如果仍未提取成功，使用原始输出
        if entities is None:
            entities = output
            
        return entities
    
    def save_intermediate_result(self, result, index):
        """保存中间结果"""
        try:
            intermediate_path = os.path.join(
                self.config['output_dir'], 
                f"ner_result_{index}.json"
            )
            with open(intermediate_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"保存中间结果时出错: {e}")
    
    def run(self):
        """执行完整的处理流程"""
        # 1. 加载提示
        prompts = self.load_prompts(self.config['prompts_file_path'])
        
        # 2. 加载数据
        text = self.load_cluster_samples(self.config['csv_path'])
        
        # 3. 实时处理每个样本
        result = []
        
        print(f"开始使用模型 {self.config['model_id']} 处理 {len(text)} 条数据...")
        for i, item in enumerate(tqdm(text)):
            # 处理单个样本
            sample_result = self.process_single_sample(item, prompts)
            
            result.append(sample_result)
            break

        # 4. 保存最终结果
        final_path = self.config['output_path']
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print(f"处理完成! 总样本: {len(result)}",
              f"结果已保存至: {final_path}")
        
        return result


if __name__ == "__main__":
    # 创建NER处理器实例
    ner_processor = NerProcessor()
    
    # 执行处理流程
    ner_processor.run()

    # 处理完成后，打印提示信息
    print("所有样本处理完成！")