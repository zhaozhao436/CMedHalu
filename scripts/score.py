import os
import re
import pandas as pd
import jsonlines
import ast
from sacrebleu.metrics import BLEU
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entity_metrics import obtain_entity_extraction_p_r_f







def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, mode='r') as reader:
        for item in reader:
            data.append(item)
    return data



def main(data_path, entity_negative_count=750):
    print('=' * 50 + f'\n{data_path.split("/")[-1]}\n' + '=' * 50)
    data = read_jsonl(data_path)
    hallucination_entity, hallucination_logic, hallucination_no = [], [], []
    for item in data:
        if item['hallucination_type'] == 'entity':
            hallucination_entity.append(item)
        elif item['hallucination_type'] == 'logic':
            hallucination_logic.append(item)
        elif item['hallucination_type'] == '':
            hallucination_no.append(item)
    print(len(hallucination_entity), len(hallucination_logic), len(hallucination_no))

    results = {'entity': {}, 'logic': {}}
    for type in ['entity', 'logic']:
        print('*' * 20)
        data_to_score = {
            'entity': hallucination_entity + hallucination_no[:entity_negative_count],    # [:750]: for v2
            'logic': hallucination_logic + hallucination_no[entity_negative_count:]       # [750:]: for v2
        }[type]

        label1, pred1 = [], []
        label2, pred2 = [], []
        label3, pred3 = [], []
        label4, pred4 = [], []
        for item in data_to_score:
            # ---------------------
            # 提取标签
            # ---------------------
            # label 1
            if item['hallucination']:
                label1.append('yes')
            else:
                label1.append('no')
            # label 2
            if item['hallucination_type'] == '':
                label2.append('/')
            elif item['hallucination_type'] == 'entity':
                label2.append('实体幻觉')
            else:
                label2.append('逻辑幻觉')
            # label 3 && label 4
            if type == 'entity':
                label3.append(f"({','.join(['(' + t + ')' for t in item['hallucination_entity_list']])})")
                label4.append(f"({','.join(['(' + t + ')' for t in item['entity_list_in_answer']])})")
            if type == 'logic':
                match = re.match(r'这是[“"](.*?)[”"]的(.*)', item['hallucination_logic'] if item['hallucination_logic'] else '')
                label3.append((match.group(1), match.group(2)) if match else (None, None))
                # label3.append(item['hallucination_logic'])
                label4.append(item['answer'])
            
            # ---------------------
            # 提取预测值
            # ---------------------
            prediction = item['prediction']
            if isinstance(prediction, list):
                prediction = prediction[0] if type == 'entity' else prediction[1]
            # prediction += '\nlabel2：xxx\nlabel3：xxx\nlabel4：xxx'     # NOTE: 只有 with chunk 才需要添加这行
            prediction_labels = prediction.strip().split('\n')
            if len(prediction_labels) != 4:
                pred1.append('invalid')
                pred2.append('invalid')
                pred3.append('invalid' if type == 'entity' else (None, None))
                pred4.append('invalid')
                continue
            pred1_str = prediction_labels[0].split('label1：')[-1].strip()
            pred1.append(pred1_str if pred1_str in ['yes', 'no'] else 'invalid')
            pred2_str = prediction_labels[1].split('label2：')[-1].strip()
            pred2.append(pred2_str if pred2_str in ['实体幻觉', '逻辑幻觉', '/'] else 'invalid')
            if type == 'entity':
                # pred 3
                prediction_label3 = prediction_labels[2].split('label3：')[-1].strip()
                try:
                    prediction_label3 = ','.join(['(' + t + ')' for t in ast.literal_eval(prediction_label3)])
                except:
                    prediction_label3 = 'invalid'
                pred3.append(prediction_label3)
                # pred 4
                prediction_label4 = prediction_labels[3].split('label4：')[-1].strip()
                try:
                    prediction_label4 = ','.join(['(' + t + ')' for t in ast.literal_eval(prediction_label4)])
                except:
                    prediction_label4 = 'invalid'
                pred4.append(prediction_label4)
            else:
                match = re.match(r'这是[“"](.*?)[”"]的(.*)', prediction_labels[2].split('label3：')[-1].strip())
                pred3.append((match.group(1), match.group(2)) if match else (None, None))
                # pred3.append(prediction_labels[2].split('label3：')[-1].strip())
                pred4.append(prediction_labels[3].split('label4：')[-1].strip())
                # print(item['hallucination_logic'], label3)
                # print(prediction_labels[2].split('label3：')[-1].strip(), pred3)

        # label 1
        acc = accuracy_score(label1, pred1)
        recall = recall_score(label1, pred1, average='macro', zero_division=0, labels=['yes', 'no'])
        f1 = f1_score(label1, pred1, average='macro', zero_division=0, labels=['yes', 'no'])
        print(f'{type} - label 1:')
        print(f'acc: {acc}\nrecall: {recall}\nf1: {f1}\n')
        results[type]['label1'] = f'acc: {acc}\nrecall: {recall}\nf1: {f1}'
        print(classification_report(label1, pred1))

        # label 2
        acc = accuracy_score(label2, pred2)
        labels = ['实体幻觉', '/'] if type == 'entity' else ['逻辑幻觉', '/']
        recall = recall_score(label2, pred2, average='macro', zero_division=0, labels=labels)
        f1 = f1_score(label2, pred2, average='macro', zero_division=0, labels=labels)
        print(f'{type} - label 2:')
        print(f'acc: {acc}\nrecall: {recall}\nf1: {f1}\n')
        results[type]['label2'] = f'acc: {acc}\nrecall: {recall}\nf1: {f1}'

        if type == 'entity':
            # label 3
            print(f'{type} - label 3:')
            exact_p_r_f, inexact_p_r_f = obtain_entity_extraction_p_r_f(pred3, label3, ['其他治疗','手术治疗','流行病学','社会学','部位','预后','疾病','检查','症状','药物','其他'])
            print(f'f1: {exact_p_r_f["全局"]["f1"]}\n')
            results[type]['label3'] = f'f1: {exact_p_r_f["全局"]["f1"]}'

            # label 4
            print(f'{type} - label 4:')
            exact_p_r_f, inexact_p_r_f = obtain_entity_extraction_p_r_f(pred4, label4, ['其他治疗','手术治疗','流行病学','社会学','部位','预后','疾病','检查','症状','药物','其他'])
            print(f'f1: {exact_p_r_f["全局"]["f1"]}\n')
            results[type]['label4'] = f'f1: {exact_p_r_f["全局"]["f1"]}'
        else:
            # label 3
            print(f'{type} - label 3:')
            acc = sum([t[0] == p[0] and t[1] == p[1] for t, p in zip(label3, pred3)]) / len(pred3)
            print(f'acc: {acc}\n')
            results[type]['label3'] = f'acc: {acc}'

            # label 4
            print(f'{type} - label 4:')
            bleu = BLEU(tokenize='zh').corpus_score(pred4, [label4]).score
            print(f'bleu: {bleu}\n')
            results[type]['label4'] = f'bleu: {bleu}'
    return results



if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--entity-negative-count', type=int, default=750)
    a = p.parse_args()
    main(a.input, a.entity_negative_count)
