"""Minimal local-model inference entry point using the original prompts."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Hugging Face ID or your local checkpoint')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--task', choices=['entity', 'logic'], required=True)
    p.add_argument('--with-chunk', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=4096)
    a = p.parse_args()
    if Path(a.input).resolve() == Path(a.output).resolve():
        p.error('input and output must differ')
    template = getattr(prompts, a.task + '_prompt' + ('_with_chunk' if a.with_chunk else ''))
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype='auto', device_map='auto')
    model.eval()
    with open(a.input, encoding='utf-8') as source, open(a.output, 'x', encoding='utf-8') as target:
        for line in source:
            item = json.loads(line)
            text = template.format_map(dict(question=item['question'], answer=item['maybe_wrong_answer'], chunk=item.get('chunk', '')))
            inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            item['prediction'] = tokenizer.decode(output[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            target.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
