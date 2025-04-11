import os
import sys
import base64
import json
import re
import argparse
import random
import time
from tqdm import tqdm


from logger import get_logger
from base_access import AccessBase, CommonClient

logger = get_logger(__name__)

##没用
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, help="directory for the input")
    parser.add_argument("--source-name", type=str, help="file name for the input")
    parser.add_argument("--train-name", type=str, default="None", help="file name for the training set")
    parser.add_argument("--data-name", type=str, help="dataset name for the input")
    parser.add_argument("--last-results", type=str, default="None", help="unfinished file")
    parser.add_argument("--write-dir", type=str, default=r"C:\Users\xiao\Desktop\Code\CoT_Decomposition",
                        help="directory for the output")
    parser.add_argument("--write-name", type=str, default="Entity_Definition", help="file name for the output")
    parser.add_argument("--picture_folder", type=str, help="picture folder path for the input")
    parser.add_argument("batch_size", type=int, default=32, help="batch size for the input")
    return parser


def read_mrc_data(dir_, prefix="test"):
    file_name = os.path.join(dir_, f"{prefix}-mrc.json")
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def read_results(dir_):
    with open(dir_, "r") as file:
        results = file.readlines()
    return results


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# 找到picture_folder路径下以及所有子目录下的同名 image_id文件
def find_dir_name(dir_, file_name):
    for root, dirs, files in os.walk(dir_):
        if file_name in files:
            return os.path.join(root, file_name)

        # 递归子目录
        for dir_name in dirs:
            file_path = find_dir_name(os.path.join(root, dir_name), file_name)
            if file_path is not None:
                return dir_name

    return None

##获取图片的url
def image2prompt(picture_folder, image_id):
    dir_name = find_dir_name(picture_folder, f"{image_id}.jpg")
    return f'http://feign1.weisanju.fun/dataset/{dir_name}/{image_id}.jpg'

##获取数据
def mrc2prompt(mrc_data_path, picture_folder_path):
    print("mrc2prompt ...")

    # 加载路径mrc_data_path 地址下的json 数组文件
    with open(mrc_data_path, 'r', encoding='utf-8') as f:
        mrc_data = json.load(f)

    # 确保每次从数据集中随机选择的是100个相同的数据
    random.seed(42)
    random.shuffle(mrc_data)
    mrc_data = mrc_data[:100]

    results = []

    for item_idx in tqdm(range(len(mrc_data))):
        
        item_ = mrc_data[item_idx]  #文本内容
        image_id = item_["image_id"]    
      #  item_ = transform(item_)
        results.append(({
                            "type": "image_url",
                            "image_url": {
                                "url": image2prompt(picture_folder_path, image_id)
                            }
                        }, item_))
    return results
# 解析步骤一的结果为步骤二的输入
def parse_output_results(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for item in data:
        if 'entities' in item and len(item['entities']) > 1:
            output_text = item['entities'][1].get('output', '')
            
            # 使用正则表达式删除实体标记中的实体类型，但保留{}和实体名称
            # 将 {实体名称,实体类型} 格式替换为 {实体名称}
            clean_text = re.sub(r'\{([^,]+),\w+\}', r'{\1}', output_text)
            
            results.append({
                "original_output": output_text,
                "cleaned_text": clean_text,
                "url": item.get('url', '')
            })
    
    return results

# 这段代码定义了一个名为 transform 的函数，其目的是将输入的数据转换成特定格式。
# 该函数的作用是将输入的json数据中，实体的位置信息提取出来，并转换成特定格式。
def transform(data):
    data_copy = data.copy()
    # data_copy['entities'] = []
    # # range i
    # for i in range(len(data['entity_label'])):
    #     start = data['start_position'][i]
    #     end = data['end_position'][i]
    #     entity_label = data['entity_label'][i]
    #     data_copy['entities'].append({start, end, entity_label})

    return data_copy


def load_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts = file.read().strip()
    return prompts

def remove_entity_types(input_file):
    """保留原始结构但删除Type字段"""
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理数据
    for item in data:
        # 删除url字段
        if "url" in item:
            del item["url"]
        if "entities" in item and isinstance(item["entities"], list):
            for entity in item["entities"]:
                if "Type" in entity:
                    del entity["Type"]
    
    
    print(f"处理完成。处理了 {len(data)}")
    return data
# 单个测试
def test_identity():
    # 读取 all_prompts.md
    prompts_file_path = r'F:\研究生文件\自己的论文\GMNER\GMNER-XMC\Code\gmner-1\openai_access\prompts\第四次实验prompt\system.md'
    prompts = load_prompts(prompts_file_path)
    client = AccessBase(
        model="gpt-4o-2024-08-06",
        temperature=0.0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )

    context = "Iggy got it RT @ iHitModelsRaw : Final MVP too RT @ NO_TATS_B : NBA CHAMPION"
    url = "http://feign1.weisanju.fun/dataset/twitter2017_images/17_06_4466.jpg"

    # 构建多模态prompt
    prompt = [
                {"role": "system", "content": prompts},
                {
                    "role": "user",
                    "content": [{
                            "type": "text",
                            "text": context
                        }, 
                        {
                            "type": "image_url",
                            "image_url": {"url": url}
                        }]
                }
            ]

    response = client.completion(prompt, 'gpt-4o-2024-08-06')

    print(response)


##少量测试
def test_image_identity():
    # 读取 all_prompts.md
    prompts_file_path = r'C:\Users\xiao\PycharmProjects\gmner\CoT_Decomposition\all_prompts.md'
    prompts = load_prompts(prompts_file_path)
    client = AccessBase(
        model="gpt-4o-2024-08-06",
        temperature=0.0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        n=1,  ###不动
    )

    samples = [
        {
            "url":'http://feign1.weisanju.fun/dataset/twitter2017_images/17_06_707.jpg',
            "context": "# LionelMessi ' s bride # antonellaRoccuzzo ' first lady of football ' |",
        },
        {
            "url":'http://feign1.weisanju.fun/dataset/twitter2017_images/16_05_10_983.jpg',
            "context": "# Baseball games are great friend makers",
        },
        {
            "url": 'http://feign1.weisanju.fun/dataset/twitter2015_images/72560.jpg',
            "context": "A favorite last night : Bruce and Diana Rauner see their reflections in drawing by a supporter",
        }
    ]

    results = []
    for sample in samples:
        contents = [{
            "type": "text",
            "text": prompts.replace('${context}', sample["context"])
        }, {
            "type": "image_url",
            "image_url": {
                "url": sample["url"]
            }
        }]
        # 构建多模态prompt
        prompt = [
            {
                "role": "user",
                "content": contents,
            }
        ]

        response = client.completion(prompt, 'gpt-4o-2024-08-06')
        print("url")
        print("context")
        print(response)
        ##输出结果至新的文件中
        results.append({
            "content": sample["context"],
            "entities": response,
            "url": sample["url"]
        })

    with open('results1（修改prompt前）.mrc', 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
        f.flush()


# base_access中已经定义好了batch_completion方法，用于批量处理多个请求，这里我们只需要调用即可。
def test_batch_mode():

    # 读取 all_prompts.md
    prompts_file_path = r'/Users/xxxxxx/Downloads/论文/gmner/prompts/实体跨度验证/valition_英文-2.md'
    prompts = load_prompts(prompts_file_path)

    ner_results_path = r'/Users/xxxxxx/Downloads/论文/gmner/result/NER/第五次实验/converted_ner_results.json'
    ner_results = remove_entity_types(ner_results_path)

    images_text = mrc2prompt(r'/Users/xxxxxx/Downloads/论文/gmner/data/Twitter10000_v2.0/mrc/merged-test-mrc-updated.json',
                        r'/Users/xxxxxx/Downloads/论文/gmner/data/IJCAI2019_data')

    ##输入的batch请求列表
    batch_request = []

    # random_images to dict image_id -> image
    image_dict = {obj[1]["image_id"]: obj for obj in images_text}

    for id, ner_result in enumerate(ner_results):
        image_info, item_info = images_text[id]  # 解包元组
        messages = {
            "custom_id": item_info["image_id"],
            "body": {
                "messages": [
                    {"role": "system", "content": prompts},
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text":f"```json\n{ner_result}\n```"
                        }]
                    }
                ],
            }
        }
        batch_request.append(messages)
        # break # 单个测试

    # 写出batch_request为一个json文件
    with open('/Users/xxxxxx/Downloads/论文/gmner/result/span_val/第三次实验/batch_request.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(batch_request, ensure_ascii=False, indent=4))
        f.flush()

    # 响应结果
    # responses = CommonClient.batch_completion(batch_request, "gpt-4o-2024-08-06") # "gpt-4o-2024-08-06","gpt-4o-mini",gpt-4o-2024-11-20
    # print(responses)
    result = []

    # 加载jsonl文件
    responses = []
    with open('/Users/xxxxxx/Downloads/论文/gmner/result/span_val/第三次实验/batch_67dfb9b9c57c8190a816712422881961_output.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 忽略空行
                responses.append(json.loads(line))

    # 解析json结果
    erro_count = 0
    for response in responses:
        output = response['response']['body']['choices'][0]['message']['content']

        entities = None       
        # 提取 ```json ... ``` 格式
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', output)
        if json_match:
            json_str = json_match.group(1).strip()
            json_str = re.sub(r'("Entity_valid"\s*:\s*(?:"True"|"False"|true|false))\s*#[^}]*?(\s*})', r'\1\2', json_str)
            json_str = json_str.replace('True', 'true').replace('False', 'false')
            try:
                entities = json.loads(json_str)
                # print(f"成功通过```json```格式提取实体")
            except json.JSONDecodeError as e:
                print(f"JSON解析失败，错误: {e}")
        
        else:
            entities = output
            # 如果entities中存在换行符，需要从该处打断，只提取换行符之前的内容
            erro_id = response['custom_id']
            erro_count += 1
            print(f"{erro_id}未能通过```json```格式提取实体，尝试直接解析")  

        image = image_dict[response["custom_id"]]
        result.append(
            {
                "content": image[1]['context'], 
                "entities": entities,
                "url": image[0]['image_url']['url']
            }
        )

    print(f"共有{erro_count}条数据未能通过```json```格式提取实体")

    with open('/Users/xxxxxx/Downloads/论文/gmner/result/span_val/第三次实验/val_results.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
        f.flush()


##非批量测试
def test():
    # 获取时间戳
    random.seed(time.time())
    # 读取 all_prompts.md
    prompts_file_path = '../CoT_Decomposition/all_prompts.md'
    prompts = load_prompts(prompts_file_path)

    images = mrc2prompt(r'C:\Users\xiao\PycharmProjects\gmner\data\Twitter10000_v2.0\mrc\merger-mrc_filtered.json',
                        r'C:\Users\xiao\PycharmProjects\gmner\data\IJCAI2019_data', )

    result = []

    random_images = random.sample(images, min(3, len(images)))

    client = AccessBase(
        model="gpt-4o-2024-08-06",
        temperature=0,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=3,
        n=1,
    )

    for (image, item) in random_images:
        contents = [{
            "type": "text",
            "content": prompts.replace('${context}', item['context'])
        }]
        contents.append(image)
        # 构建多模态prompt
        prompt = [
            {
                "role": "user",
                "content": contents
            }
        ]

        response = client.completion(prompt, 'gpt-4o-2024-08-06')

        print('url:' + image['image_url']['url'])
        print('context:' + item['context'])

        print(response, flush=True)

        try:
            json_start = response.index('```json') + len('```json')
            json_end = response.index('```', json_start)
            json_response = response[json_start:json_end].strip()
            entities = json.loads(json_response)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error parsing JSON response: {e}")
            continue

        result.append(
            {
                "content": item['context'],
                "entities": entities,
                "url": image['image_url']['url']
            }
        )

    with open('results1（修改prompt前）.mrc', 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
        f.flush()
    #  print(json.dumps(result, ensure_ascii=False))


def ner_access(openai_access, ner_pairs):
    print("tagging ...")
    results = []
    start_ = 0
    pbar = tqdm(total=len(ner_pairs))
    while start_ < len(ner_pairs):
        end_ = min(start_ + 1, len(ner_pairs))  # 从ner_pairs中逐个处理
        results += openai_access.get_multiple_sample(ner_pairs[start_:end_])
        pbar.update(end_ - start_)
        start_ = end_
    pbar.close()
    return results


def write_file(labels, dir_, last_name):
    print("writing ...")
    file_name = os.path.join(dir_, f"{last_name}.txt")
    with open(file_name, "w", encoding="utf-8") as file:
        for line in labels:
            file.write(line.strip() + '\n')


if __name__ == '__main__':

    # batch
    test_batch_mode()

    # 非batch，多个测试
    # test()    

    # 单个测试，非batch形式
    # test_identity()      
    
