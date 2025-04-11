import os
import sys
import base64
import json
import argparse
import random
import time
from tqdm import tqdm


from logger import get_logger
from base_access import AccessBase, CommonClient
# from silicon_flow_access import SiliconFlowAccess, SiliconFlowClient

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
    # random.seed(42)
    # random.shuffle(mrc_data)
    # mrc_data = mrc_data[:100]

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

def test_batch_mode():
    # 读取 all_prompts.md
    prompts_file_path = r'prompts/zero-shot-base/VG.md'
    prompts = load_prompts(prompts_file_path)

    images = mrc2prompt('/Users/xxxxxx/Downloads/论文/gmner/data/Twitter10000_v2.0/mrc/merged-test-mrc-updated.json',
                      '/Users/xxxxxx/Downloads/论文/gmner/data/IJCAI2019_data')

    with open('result/zero-shot-no-demo/samples1500/vg/vg_input.json', 'r', encoding='utf-8') as f:
        inputs = json.load(f)

    # 依次输入请求
    image_dict = {obj[1]["image_id"]: obj for obj in images}
    result = []
    erro_count = 0
    
    # 检查是否存在中间结果文件，以便断点续传
    last_checkpoint = None
    checkpoint_files = [f for f in os.listdir('result/zero-shot-no-demo/samples1500/vg/') 
                        if f.startswith('vg_results_checkpoint_') and f.endswith('.json')]
    
    if checkpoint_files:
        # 找到最新的检查点文件
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        last_checkpoint = checkpoint_files[-1]
        checkpoint_id = int(last_checkpoint.split('_')[-1].split('.')[0])
        
        print(f"找到检查点文件: {last_checkpoint}，从样本 {checkpoint_id} 继续处理")
        
        # 加载现有结果
        with open(f'result/zero-shot-no-demo/samples1500/vg/{last_checkpoint}', 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        # 设置起始样本ID
        start_id = checkpoint_id
    else:
        start_id = 0
    
    # 确保保存目录存在
    os.makedirs('result/zero-shot-no-demo/samples1500/vg', exist_ok=True)
    
    all_outputs = []
    error_id = []
    # 使用tqdm添加进度条，显示总数和当前进度
    for id in tqdm(range(start_id, len(inputs)), initial=start_id, total=len(inputs), desc="处理样本"):
        input_data = inputs[id]
        image_info, item_info = images[id]  # 解包元组
        input_str = json.dumps(input_data) if isinstance(input_data, dict) else str(input_data)
        
        messages = [
            {"role": "system", "content": prompts},
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"```json\n{input_str}\n```"
                }, image_info]
            }
        ]

        # 显示当前正在处理的样本ID
        tqdm.write(f"处理样本 {id+1}/{len(inputs)}: {item_info.get('context', '')[:30]}...")

        # 记录开始时间，用于性能监控
        start_time = time.time()
        
        try:
            response = CommonClient.completion(messages) # "gpt-4o-2024-08-06","gpt-4o-mini",gpt-4o-2024-11-20
            print(response)
            process_time = time.time() - start_time
            tqdm.write(f"样本 {id+1} 处理耗时: {process_time:.2f}秒")
            
            # 处理响应
            output = response.json()  
            output = json.loads(output)      
            output = output["choices"][0]["message"]["content"]

            all_outputs.append({
                "image_id": item_info.get('image_id', ''),
                "context": item_info.get('context', ''),
                "url": image_info['image_url']['url'],
                "raw_output": output
            })

            # 只提取最后一个json结果
            if output.startswith('```json'):
                json_starts = [i for i in range(len(output)) if output.startswith('```json', i)]
                if len(json_starts) > 0:
                    try:
                        json_start = json_starts[-1] + len('```json')
                        json_end = output.index('```', json_start)
                        entities = json.loads(output[json_start:json_end].strip())
                        
                        # 如果返回的是错误对象，记录错误
                        if isinstance(entities, dict) and "error" in entities:
                            erro_count += 1
                            print(f"JSON解析错误: {entities.get('error')}")
                            # 保存原始文本以便后续分析
                            raw_text = entities.get('raw_text', '')
                            print(f"原始文本: {raw_text[:100]}...")
                            
                            # 尝试手动构建一个简单的实体对象
                            entities = {"parse_error": True, "raw_text": raw_text}
                    except Exception as e:
                        entities = output[json_start:json_end]
                        erro_count += 1
                        print(f"提取 {item_info['image_id']}JSON时发生异常: {type(e).__name__}: {str(e)}")
                        entities = {"extraction_error": str(e), "raw_output": output[:200] + "..."}
            else:
                erro_count += 1
                entities = output
                error_id.append(item_info['image_id'])
                # print(f"没有找到JSON格式的输出:{item_info['image_id']}")
                
            # 添加当前结果
            result.append({
                "content": item_info['context'],
                "entities": entities,
                "url": image_info['image_url']['url'],
            })
            
            # 每处理100个样本保存一次中间结果
            if (id + 1) % 100 == 0:
                checkpoint_file = f'result/zero-shot-no-demo/samples1500/vg/vg_results_checkpoint_{id + 1}.json'
                tqdm.write(f"保存检查点: {checkpoint_file}")
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(result, ensure_ascii=False, indent=4, fp=f)
                    f.flush()
                
                # 可选：删除旧的检查点文件以节省空间
                if last_checkpoint and last_checkpoint != checkpoint_file:
                    try:
                        os.remove(f'result/zero-shot-no-demo/samples1500/vg/{last_checkpoint}')
                        print(f"已删除旧检查点: {last_checkpoint}")
                    except:
                        pass
                
                last_checkpoint = f'vg_results_checkpoint_{id + 1}.json'
                
        except Exception as e:
            # 处理请求过程中的任何异常
            erro_count += 1
            print(f"处理样本 {id+1} 时发生异常: {type(e).__name__}: {str(e)}")
            
            # 记录失败的样本但继续处理
            result.append({
                "content": item_info['context'],
                "entities": {"processing_error": str(e)},
                "url": image_info['image_url']['url'],
                "error": True
            })
            
            # 即使发生错误也保存中间结果（如果恰好是100的倍数或发生严重错误）
            if (id + 1) % 100 == 0 or isinstance(e, (KeyboardInterrupt, SystemExit)):
                checkpoint_file = f'result/zero-shot-no-demo/samples1500/vg/vg_results_checkpoint_{id + 1}.json'
                print(f"错误后保存检查点: {checkpoint_file}")
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(result, ensure_ascii=False, indent=4, fp=f)
                    f.flush()
        # break

    print(f"共有{erro_count}条数据未能通过```json```格式提取实体")
    print(f"共有{len(error_id)}条数据未能通过```json```格式提取实体，具体为：{error_id}")

    # 保存最终结果
    with open('result/zero-shot-no-demo/samples1500/vg/vg_results_final.json', 'w', encoding='utf-8') as f:
        json.dump(result, ensure_ascii=False, indent=4, fp=f)
        f.flush()
        print("最终结果已写出")
    
    # 保存所有原始输出
    with open('result/zero-shot-no-demo/samples1500/vg/vg_outputs_all.json', 'w', encoding='utf-8') as f:
        json.dump(all_outputs, ensure_ascii=False, indent=4, fp=f)
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
    
