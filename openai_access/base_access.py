# encoding: utf-8
import os
import random
import time
from math import ceil  # 用于计算大于或等于指定数的最小整数值
from typing import List
import base64
import requests
import re
import json
import openai  # 这个库通常用于与OpenAI的API进行交互，可能是用于调用GPT模型或其他相关的API服务
from openai import OpenAI
from pydantic import BaseModel
import configparser

from pathlib import Path
from tqdm import tqdm

from logger import get_logger
print("所有导入成功!")

logger = get_logger(__name__)

# 常量定义
INIT_DELAY = 10  # 初始化的延迟时间，处理API请求失败后的重试间隔
EXPONENTIAL_BASE = 1.5  # 指数基数，可能用于计算指数退避的延迟时间（每次重试的等待时间指数增加）
MAX_RETRIES = 3  # 最大重试次数，指的是在发生错误或失败时允许的最大重试次数


# 包含了一些用于与 OpenAI API 交互的属性和方法
class AccessBase(object):
    delay = INIT_DELAY

    # 初始化方法，设置与 OpenAI API 交互的参数
    def __init__(self, model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, best_of,n):
        self.model = model  # OpenAI API 中使用的模型引擎名称（例如 davinci）
        self.temperature = temperature  # 控制生成文本的多样性，值越高生成的文本越随机
        self.max_tokens = max_tokens  # 设置生成文本的最大长度（单位：标记/词）
        self.top_p = top_p  # 用于基于核采样策略的采样方式(对比贪婪搜索)
        self.frequency_penalty = frequency_penalty  # 控制生成文本中的重复词汇频率
        self.presence_penalty = presence_penalty  # 控制生成的文本与输入的相关性
        self.best_of = best_of  # 指定在生成多个候选文本时返回最佳的一个
        self.client = self._get_client()  # 获取 OpenAI API 客户端对象
        self.n = n
    # 非batch方法，用于单个请求
    # def completion(self, prompt):
    #     response = self.client.chat.completions.create(
    #         model= self.model,
    #         # messages=[{"role": "user", "content": f"{prompt}"}],
    #         messages= prompt,
    #         stream=False
    #     )
    #     # print(f"input:{response.usage.prompt_tokens},output:{response.usage.completion_tokens}")
    #     return response
    def completion(self, prompt):
        """
        非batch方法，用于单个请求，增加了网络异常处理和重试机制
        """
        max_retries = MAX_RETRIES  # 使用类常量定义的最大重试次数
        retry_count = 0
        base_delay = INIT_DELAY  # 初始延迟时间
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    stream=False
                )
                # 请求成功，重置延迟并返回
                AccessBase.delay = INIT_DELAY
                return response
                
            except (openai.APIConnectionError, openai.APITimeoutError, 
                    requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                # 处理网络相关异常
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"网络连接异常，已达到最大重试次数: {e}")
                    raise  # 超过最大重试次数，重新抛出异常
                
                # 计算退避延迟时间
                jitter = random.random() * 0.5  # 添加0-0.5之间的随机抖动
                wait_time = base_delay * (EXPONENTIAL_BASE ** (retry_count - 1)) * (1 + jitter)
                wait_time = min(wait_time, 60)  # 最大等待60秒
                
                logger.warning(f"网络异常: {type(e).__name__}: {str(e)}. {wait_time:.2f}秒后第{retry_count}/{max_retries}次重试")
                time.sleep(wait_time)
                
            except openai.RateLimitError as e:
                # 处理API速率限制
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"API速率限制，已达到最大重试次数: {e}")
                    raise
                    
                # 速率限制通常需要更长的等待时间
                wait_time = base_delay * (EXPONENTIAL_BASE ** retry_count) * (1 + random.random())
                wait_time = min(wait_time, 120)  # 最大等待120秒
                
                logger.warning(f"速率限制: {str(e)}. {wait_time:.2f}秒后第{retry_count}/{max_retries}次重试")
                time.sleep(wait_time)
                
            except Exception as e:
                # 记录详细错误信息
                logger.error(f"API调用异常: {type(e).__name__}: {str(e)}")
                # 其他类型的异常可能不适合重试，直接抛出
                raise
    
        # 如果达到这里，说明已经超过了最大重试次数
        raise Exception(f"达到最大重试次数({max_retries})，completion请求失败")

    # 定义batch_completion方法，用于批量    处理多个请求
    def batch_completion(self, batch_request:list[dict], model=None, batch_id=None):
        # 设置请求项的method和url
        for item in batch_request:
            item['method'] = 'POST'
            item['url'] = '/v1/chat/completions'
            if model is not None and item['body'].get('model') is None:
                item['body']['model'] = model

        # 将请求项转换为JSON字符串并编码为字节
        request_bytes = "\n".join([json.dumps(item, ensure_ascii=False) for item in batch_request]).encode('utf-8')

        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 上传字节数据创建文件
                if batch_id is None:
                    # 上传批量请求以创建batch文件(获取ID)
                    input_file_id_response = self.client.files.create(file=request_bytes, purpose='batch')

                    if input_file_id_response.id is None:
                        print(input_file_id_response.model_dump_json())
                        raise ValueError
                    
                    # 使用文件ID创建批处理任务
                    batch_response = self.client.batches.create(
                        completion_window='24h',
                        endpoint='/v1/chat/completions',
                        input_file_id=input_file_id_response.id
                    )
                    if batch_response.id is None:
                        print(batch_response.model_dump_json())
                        raise ValueError
                    
                    batch_id = batch_response.id
                    print(f"batch_id:{batch_id}")

                # 轮询批处理任务状态
                poll_count = 10
                max_polls = 500  # 最多轮询60次
                poll_count = min(10 + poll_count // 10, 60)  # 间隔从10秒开始，最大60秒

                while poll_count < max_polls:
                    try:
                        task_obj = self.client.batches.retrieve(batch_id)
                        print()
                        print(task_obj.model_dump_json(), flush=True)
                        print(flush=True)

                        if task_obj.status in ["validating", "in_progress", "finalizing"]:
                            time.sleep(5)
                            poll_count += 1
                            continue
                        
                        break
                    except Exception as e:
                        print(f"轮询异常: {str(e)}")
                        time.sleep(10)  # 出错后等待10秒
                        poll_count += 1
                        
                if poll_count >= max_polls:
                    print("轮询次数过多，尝试重新开始批处理")
                    batch_id = None  # 重置batch_id，重新创建批处理
                    retry_count += 1
                    continue

                # 处理错误文件
                if task_obj.error_file_id is not None:
                    print(self.client.files.content(task_obj.error_file_id).text)

                result = []

                # 读取输出文件内容并解析为JSON对象
                if task_obj.output_file_id is not None:
                    for item in self.client.files.content(task_obj.output_file_id).iter_lines():
                        response = json.loads(item)
                        result.append(response)

                return result
                
            except Exception as e:
                print(f"批处理异常: {type(e).__name__}: {str(e)}")
                retry_count += 1
                if retry_count >= max_retries:
                    raise  # 超过最大重试次数，重新抛出异常
                print(f"5秒后进行第{retry_count}次重试...")
                time.sleep(5)
                batch_id = None  # 重置batch_id，重新创建批处理
        
        raise Exception("达到最大重试次数，批处理失败")
    ##调用大模型api
    def _get_client(self):
        config_file = os.path.join('/Users/xxxxxx/Downloads/论文/gmner/openai_access/env.ini')
        parser = configparser.ConfigParser()
        # 显式指定编码为 utf-8
        with open(config_file, 'r', encoding='utf-8') as file:
            parser.read_file(file)

        section = parser.get('common', 'section', fallback='openai')

        return OpenAI(
            api_key=parser.get(section, 'api_key'),
            base_url=parser.get(section, 'base_url'),
            timeout=60.0,  # 增加超时时间到60秒
            max_retries=3  # 添加自动重试次数
        )

    def _get_multiple_sample(self, prompt_list: List[str]):
        response = self.client.chat.completions.create(
            model=self.model,
            prompt=prompt_list,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            best_of=self.best_of,
            n=self.n,
            batch=self.batch
        )
        results = []
        for choice in response.choices:
            text = choice.text
            json_match = re.search(r'\[\s*{.*?}\s*\]', text, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
                results.append(json_content)
            else:
                print("未能找到匹配的JSON内容")

        logger.info(msg="prompt_and_result", extra={"prompt_list": prompt_list, "results": results})
        return results
    ##没用到
    def get_multiple_sample(
            self,
            prompt_list: List[str],
            jitter: bool = True,
    ):
        errors: tuple = (openai.RateLimitError,)
        num_retries = 0

        while True:
            used_delay = AccessBase.delay
            try:
                logger.info(f"Delay={used_delay - 1}")
                for _ in tqdm(range(ceil(used_delay)), desc=f"sleep{used_delay}"):
                    time.sleep(1)

                results = self._get_multiple_sample(prompt_list)
                AccessBase.delay = INIT_DELAY
                return results
            except errors as e:
                logger.info("retry ...")
                num_retries += 1
                if num_retries > MAX_RETRIES:
                    logger.error("failed")
                    raise Exception(f"Maximum number of retries ({MAX_RETRIES}) exceeded.")
                AccessBase.delay = max(AccessBase.delay, used_delay * EXPONENTIAL_BASE * (1 + jitter * random.random()))
            except Exception:
                logger.info("retry ...")
                num_retries += 1
                if num_retries > MAX_RETRIES:
                    logger.error("failed")
                    raise Exception(f"Maximum number of retries ({MAX_RETRIES}) exceeded.")
                AccessBase.delay = max(AccessBase.delay, used_delay * EXPONENTIAL_BASE * (1 + jitter * random.random()))


CommonClient = AccessBase(
    model= "gpt-4o-mini",  # "gpt-4o-2024-08-06","gpt-4o-mini","gpt-4o-2024-11-20"，“Qwen/Qwen2.5-VL-72B-Instruct”
    temperature=0, # 0.4
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    best_of=1,
    n=1
)