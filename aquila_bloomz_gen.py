import time
import torch
import pandas
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == '__main__':
    model_name = 'bloomz-7b1-mt'
    df = pandas.read_csv('turing.csv', encoding='GBK')
    query_names = [f'{df[df.columns[0]][i]}（{df[df.columns[1]][i]}）' for i in range(df.shape[0])]

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(f".\\models\\{model_name}").to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(f".\\models\\{model_name}")
    mid = time.time()
    print(f"Load model for {mid - start}s.")
    model = model.eval()

    meta_instruction = "现在我需要你扮演一个中文维基百科计算机领域编辑者，为所给图灵奖获得者编写生平简介，主要包含生卒年、出生地、国籍、毕业学院、主要事迹、获得奖项等信息。字数不限，但内容要完整。"

    with jsonlines.open(f'outputs/turing_{model_name}.jsonl', 'w') as fout:
        for query_name in tqdm(query_names):
            query = meta_instruction + query_name
            tokens = tokenizer.encode_plus(query)['input_ids'][:-1]
            tokens = torch.tensor(tokens)[None,].to("cuda:0")
            response = model.generate(tokens, do_sample=True, max_length=512, eos_token_id=100007)[0]
            response = tokenizer.decode(response.cpu().numpy().tolist())
            fout.write({'query': query, 'response': response})