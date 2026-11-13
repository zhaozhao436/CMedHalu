import re
import csv
import sys
import json
import jsonlines
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from random import randint, seed, sample, shuffle

def entity_statistics(kg_path, output):
    csv.field_size_limit(sys.maxsize)       # 文件太大，会超限制
    data_path = kg_path
    entity_map = {}
    with open(data_path, 'r', encoding='utf-8') as file:
        for i, row in tqdm(enumerate(file)):
            if i == 0:
                continue
            for entity_str in row.strip().split('\t'):
                if '@@' not in entity_str:
                    continue
                try:
                    entity, entity_type = entity_str.split('@@')
                    if entity_map.get(entity_type):
                        entity_map[entity_type].append(entity)
                    else:
                        entity_map[entity_type] = [entity]
                except:
                    continue
    
    entity_length_list = []
    for key, value in entity_map.items():
        # 统计每个实体类别中实体的数量
        entity_map[key] = list(set(value))
        print(f'{key}: {len(entity_map[key])}')
        # 统计实体的长度
        for entity in entity_map[key]:
            entity_length_list.append(len(entity))
    print('max entity length: ', max(entity_length_list))
    print('min entity length: ', min(entity_length_list))
    print('avg entity length: ', np.mean(entity_length_list))
    # save
    with open(output, 'w', encoding='utf-8') as file:
        json.dump(entity_map, file, indent=4, ensure_ascii=False)  # indent参数使输出更易读


def filter_entity(input_path, output, max_entity_length=16):
    data = []
    entity_list_num = []
    with jsonlines.open(input_path, 'r') as f:
        for item in tqdm(f):
            new_entity_list_in_answer = []
            for entity_and_type in item['entity_list_in_answer']:
                # 格式错误的跳过
                if len(entity_and_type.split('@@')) != 2:
                    continue
                entity = entity_and_type.split('@@')[0]
                type = entity_and_type.split('@@')[1]
                # 逗号、句号、实体类别错误、
                if ',' in entity or '。' in entity or (type not in ['疾病', '检查', '其他治疗', '症状', '手术治疗', '药物', '部位']):
                    continue
                new_entity_list_in_answer.append(entity_and_type)
            # 筛选后实体列表为空的丢掉
            if len(new_entity_list_in_answer) == 0:
                continue
            # 筛选知识片段中大片英文的
            total_chars = len(item['chunk'].replace(' ', ''))
            if total_chars == 0:
                continue
            letter_count = len(re.findall('[a-zA-Z]', item['chunk'].replace(' ', '')))
            if letter_count / total_chars > 0.6:
                continue

            item['entity_list_in_answer'] = new_entity_list_in_answer
            data.append(item)
            entity_list_num.append(len(new_entity_list_in_answer))
    print('filtered data num: ', len(data))
    print(f'length of new entity list in answer, max: {max(entity_list_num)}, avg: {sum(entity_list_num) / len(entity_list_num)}')
    # save
    with jsonlines.open(output, 'w') as f:
        f.write_all(data)


def run_exchange_entity_and_mismatch_answer(input_path, entity_map_path, output):
    seed(2025)
    # 读取原始数据
    data = []
    with jsonlines.open(input_path, 'r') as f:
        for item in f:
            data.append(item)
    num = int(len(data) * 0.4)
    # 读取实体库
    with open(entity_map_path, 'r', encoding='utf-8') as f:
        entity_map = json.load(f)
    
    results = []
    # ------------------
    # 答案错配
    # ------------------
    total_items = len(data)
    # 1. 计算需要交换的元素总数，并确保为偶数
    num_to_swap = int(total_items * 0.4)
    if num_to_swap % 2 != 0:
        num_to_swap -= 1
    # 2. 按 question_type 对项目进行分组
    # 使用 .copy() 是一个好习惯，避免直接修改原始输入列表中的字典对象
    grouped_by_type = defaultdict(list)
    for item in data:
        grouped_by_type[item['question_type']].append(item.copy())
    swapped_list = []
    unswapped_list = []
    swapped_count = 0
    # 为了增加随机性，将分组的键（question_type）也随机打乱
    group_keys = list(grouped_by_type.keys())
    shuffle(group_keys)
    # 3. 遍历分组进行交换
    for q_type in group_keys:
        # 如果已经达到目标交换数量，则停止
        if swapped_count >= num_to_swap:
            break
        items_in_group = grouped_by_type[q_type]
        # 在分组内部也进行随机打乱
        shuffle(items_in_group)
        # 在当前分组内进行配对交换
        while len(items_in_group) >= 2 and swapped_count < num_to_swap:
            # 从列表末尾弹出两个元素进行配对
            item1 = items_in_group.pop()
            item2 = items_in_group.pop()
            # 确保 file_name 不同才能交换
            if item1['file_name'] != item2['file_name']:
                # 执行交换
                item1['maybe_wrong_answer'], item2['maybe_wrong_answer'] = item2['answer'], item1['answer']
                item1['hallucination'] = True
                item1['hallucination_type'] = 'logic'
                item1['hallucination_entity_list'] = []
                item1['hallucination_logic'] = f'这是《{item2["file_name"]}》的{item2["question_type"]}'
                item1['right_entity_list'] = []
                item1['difficulty'] = ''
                item2['hallucination'] = True
                item2['hallucination_type'] = 'logic'
                item2['hallucination_entity_list'] = []
                item2['hallucination_logic'] = f'这是《{item1["file_name"]}》的{item1["question_type"]}'
                item2['right_entity_list'] = []
                item2['difficulty'] = ''
                # 将交换后的元素存入 swapped_list
                results.extend([item1, item2])
                swapped_count += 2
            else:
                # 如果 file_name 相同，无法交换，将它们放入 unswapped_list
                unswapped_list.extend([item1, item2])
    # 4. 收集所有剩余的、未被处理的元素
    # 这些元素包括：
    # - 位于处理过的分组中，但因数量奇数而落单的元素
    # - 位于因达到交换上限而未被处理的分组中的所有元素
    for key in group_keys:
        unswapped_list.extend(grouped_by_type[key])
    print('答案错配数量：', len(results))

    # --------------
    # 实体交换
    # --------------
    for item in tqdm(unswapped_list[:num]):
        entity_list = item['entity_list_in_answer']     # 原始实体列表
        exchange_num = randint(1, min(5, len(entity_list)))     # 交换的数量
        to_exchange_entity_list = sample(entity_list, exchange_num)     # 采样将要交换的实体
        # 交换实体
        exchanged_entity_list = []
        for to_exchange_entity in to_exchange_entity_list:
            entity, entity_type = to_exchange_entity.split('@@')[0], to_exchange_entity.split('@@')[1]
            exchanged_entity = sample(entity_map[entity_type], 1)[0]
            exchanged_entity_list.append(exchanged_entity + f'@@{entity_type}')
        # 组装交换实体后的错误答案
        maybe_wrong_answer = deepcopy(item['answer'])
        for right_entity, wrong_entity in zip(to_exchange_entity_list, exchanged_entity_list):
            maybe_wrong_answer = maybe_wrong_answer.replace(right_entity.split('@@')[0], wrong_entity.split('@@')[0])
        # 组装数据
        item['hallucination'] = True
        item['hallucination_type'] = 'entity'
        item['hallucination_entity_list'] = exchanged_entity_list
        item['hallucination_logic'] = None
        item['right_entity_list'] = to_exchange_entity_list
        item['maybe_wrong_answer'] = maybe_wrong_answer
        item['difficulty'] = ''
        results.append(item)
    print('实体交换数量：', num)

    # # -------------
    # # 正确样本
    # # -------------
    for item in tqdm(unswapped_list[num:]):
        item['hallucination'] = False
        item['hallucination_type'] = ''
        item['hallucination_entity_list'] = []
        item['hallucination_logic'] = None
        item['right_entity_list'] = []
        item['maybe_wrong_answer'] = ''
        item['difficulty'] = ''
        results.append(item)
    print('正确样本数：', len(unswapped_list[num:]))

    # 保存
    with jsonlines.open(output, 'w') as f:
        f.write_all(results)


def get_train_data_for_llamafactory(input_path, output):
    entity_prompt = '''下边是一个问题和答案，答案可能正确，也可能存在实体级事实幻觉。这里的实体级事实幻觉指的是，尽管答案是语义是通顺的，但其中存在某些实体错误，与客观事实或医疗知识相违背。要求：只输出以下内容,不输出其他内容，且输出的修正实体和幻觉实体要一一对应，顺序不可以错：
幻觉标签label1：[yes, no]；
幻觉类型label2：[实体幻觉, /];
幻觉实体label3：[实体1@实体类型1, 实体2@实体类型2.., /]
修正实体label4：[实体3@实体类型3, 实体4@实体类型4.., /]

现在开始！

问题：{question}
回答：{answer}
输出：
'''
    logic_prompt = '''下边是一个问题和答案，答案可能正确，也可能存在逻辑级事实幻觉。这里的逻辑级事实幻觉指的是，尽管答案是语义是通顺的，但其中存在一整段逻辑错误，比如提问的是A病的治疗方法，但回答了B病的治疗方法，这与客观事实或医疗知识相违背。要求：只输出以下内容,不输出其他内容：
幻觉标签label1：[yes, no]
幻觉类型label2：[逻辑幻觉，/]
幻觉逻辑label3：[“XXX”,/]
正确答案label4：[“XXXXX”,/]

现在开始！

问题：{question}
答案：{answer}
输出：
'''
    data = []
    with jsonlines.open(input_path, 'r') as f:
        for item in f:
            data.append(item)
    results = []
    for item in data:
        prompt1 = entity_prompt.format_map({'question': item['question'], 'answer': item['maybe_wrong_answer']})
        prompt2 = logic_prompt.format_map({'question': item['question'], 'answer': item['maybe_wrong_answer']})
        output1 = '幻觉标签label1：{label1}\n幻觉类型label2：{label2}\n幻觉实体label3：{label3}\n修正实体label4：{label4}'
        output2 = '幻觉标签label1：{label1}\n幻觉类型label2：{label2}\n幻觉逻辑label3：{label3}\n正确答案label4：{label4}'
        if item['hallucination'] and item['hallucination_type'] == 'entity':
            results.append(
                {
                    'instruction': prompt1,
                    'input': '',
                    'output': output1.format_map({'label1': 'yes', 'label2': '实体幻觉', 'label3': item['hallucination_entity_list'], 'label4': item['right_entity_list']})
                }
            )
        elif item['hallucination'] and item['hallucination_type'] == 'logic':
            results.append(
                {
                    'instruction': prompt2,
                    'input': '',
                    'output': output2.format_map({'label1': 'yes', 'label2': '逻辑幻觉', 'label3': item['hallucination_logic'], 'label4': item['answer']})
                }
            )
        else:
            results.append(
                {
                    'instruction': prompt1,
                    'input': '',
                    'output': output1.format_map({'label1': 'no', 'label2': '/', 'label3': '/', 'label4': '/'})
                }
            )
            results.append(
                {
                    'instruction': prompt2,
                    'input': '',
                    'output': output2.format_map({'label1': 'no', 'label2': '/', 'label3': '/', 'label4': '/'})
                }
            )
    with jsonlines.open(output, 'w') as f:
        f.write_all(results)
    print(len(results))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Original construction helpers; see README limitations.')
    p.add_argument('stage', choices=['entities', 'filter', 'construct', 'sft'])
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--entity-map')
    a = p.parse_args()
    if a.stage == 'entities': entity_statistics(a.input, a.output)
    elif a.stage == 'filter': filter_entity(a.input, a.output)
    elif a.stage == 'sft': get_train_data_for_llamafactory(a.input, a.output)
    else:
        if not a.entity_map: p.error('--entity-map is required for construct')
        run_exchange_entity_and_mismatch_answer(a.input, a.entity_map, a.output)
