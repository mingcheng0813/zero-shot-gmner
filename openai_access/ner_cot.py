import os
import sys
import base64
import json
import argparse
import random
import time
from tqdm import tqdm
import os
import re


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
        matched_image_ids = item_["matched_image_ids"]
        results.append(({
                            "type": "image_url",
                            "image_url": {
                                "url": image2prompt(picture_folder_path, image_id)
                            }
                        }, item_, matched_image_ids))
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


##base_access中已经定义好了batch_completion方法，用于批量处理多个请求，这里我们只需要调用即可。
def test_batch_mode():

    # 读取 all_prompts.md
    prompts_file_path = r'prompts/zero-shot-demo/NER/NER_system_cot5.md'
    prompts = load_prompts(prompts_file_path)


    images_text_matched_image_id = mrc2prompt(r'data/Twitter10000_v2.0/mrc/merged-test-mrc-updated-with-matches.json',
                        r'/Users/xxxxxx/Downloads/论文/gmner/data/IJCAI2019_data')
    # 加载一个json文件
    with open(r'result/zero-shot-demo/ner-demo/cot_correct.json', 'r', encoding='utf-8') as f:
        demo = json.load(f)

    ##输入的batch请求列表
    batch_request = []
    image_dict = {obj[1]["image_id"]: obj for obj in images_text_matched_image_id}

    for (image, item, matched_image_ids) in images_text_matched_image_id:
        matched_image_id = matched_image_ids[:6]
        examples = []
        for cot in demo:
            if cot.get('image_id') in matched_image_id:
                # 找到匹配项，返回entities字段
                example = cot.get('entities')
                examples.append(example)

        examples_array = []
        for i, example in enumerate(examples):
                examples_array.append({
                    "input": example[0]['input'],
                    "reasoning_content": example[1]['reasoning_content'],
                    "output": example[2]['output']
                })
        
        # 使用json.dumps确保正确处理所有引号和特殊字符
        examples_json_content = json.dumps(examples_array, ensure_ascii=False, indent=4)
        
        # 包装在Markdown代码块中
        examples_json = f"```json\n{examples_json_content}\n```"
        
        # 替换模板中的占位符
        few_shot_prompt = re.sub('{{EXAMPLES}}', examples_json, prompts)

        messages = {
            "custom_id": item["image_id"],
            "body": {
                "messages": [
                    {"role": "system", "content": few_shot_prompt},
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": item["context"]
                        }]
                    }
                ],
            }
        }
        batch_request.append(messages)
        # break # 单个测试

    # 写出batch_request为一个json文件
    with open('result/zero-shot-demo/gpt-4o-mini/ner/origin-cot/batch_request.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(batch_request, ensure_ascii=False, indent=4))
        f.flush()

    # 响应结果
    responses = CommonClient.batch_completion(batch_request, "gpt-4o-mini") # "gpt-4o-2024-08-06","gpt-4o-mini",gpt-4o-2024-11-20
    print(responses)
    
    result = []

    # 加载jsonl文件
    # responses = []
    # with open('result/zero-shot-demo/ner/origin-cot/sota/batch_67f0e9d0560881908806a22af01f5e3f_output.jsonl', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         if line.strip():  # 忽略空行
    #             responses.append(json.loads(line))

    # 解析json结果
    erro_count = 0
    for response in responses:
        output = response['response']['body']['choices'][0]['message']['content']
    
        entities = None
        erro_id = response['custom_id']
               
        # 提取 ```json ... ``` 格式
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', output)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                entities = json.loads(json_str)
            except json.JSONDecodeError as e:
                erro_count += 1
                entities = output  # 使用原始输出作为fallback
                print(f"{erro_id}未能解析JSON，使用原始输出")

        else:
            erro_count += 1
            entities = output  # 使用原始输出作为fallback
            print(f"{erro_id}未能解析JSON，使用原始输出")
            # 处理未找到JSON的情况...
               
        image = image_dict[response["custom_id"]]
        result.append(
            {
                "content": image[1]['context'],
                "entities": entities,
                "url": image[0]['image_url']['url']
            }
        )

    with open('result/zero-shot-demo/gpt-4o-mini/ner/origin-cot/ner_results.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))
        f.flush()
        print("写入成功")
    print(f"解析json失败的数量: {erro_count}")


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
    
