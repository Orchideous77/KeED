import time
import torch
import pandas
import jsonlines
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


if __name__ == '__main__':
    model_name = 'Baichuan-13B-Chat'
    df = pandas.read_csv('turing.csv', encoding='GBK')
    query_names = [f'{df[df.columns[0]][i]}（{df[df.columns[1]][i]}）' for i in range(df.shape[0])]

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        f".\\models\\{model_name}",
        trust_remote_code=True
    ).half().to('cuda:0')
    model.generation_config = GenerationConfig.from_pretrained(
        f".\\models\\{model_name}"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        f".\\models\\{model_name}",
        use_fast=False,
        trust_remote_code=True
    )
    mid = time.time()
    print(f"Load model for {mid - start}s.")
    model = model.eval()

    meta_instruction = "欢迎使用百川大模型，输入内容即可进行对话。现在我需要你扮演一个中文维基百科计算机领域编辑者，为所给图灵奖获得者编写生平简介，主要包含生卒年、出生地、国籍、毕业学院、主要事迹、获得奖项等信息。字数不限，但内容要完整。"
    with jsonlines.open(f'outputs/turing_{model_name}.jsonl', 'w') as fout:
        for query_name in tqdm(query_names):
            query = meta_instruction + query_name
            if "7" in model_name:
                inputs = tokenizer(query, return_tensors='pt')
                inputs = inputs.to('cuda:0')
                pred = model.generate(**inputs, max_new_tokens=500, repetition_penalty=1.1)
                fout.write({'query': query, 'response': tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)})
            elif "13" in model_name:
                message = [{"role": "user", "content": query}]
                response = model.chat(tokenizer, message)
                fout.write({'query': query, 'response': response})