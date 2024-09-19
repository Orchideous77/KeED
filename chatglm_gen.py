import time
import pandas
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


if __name__ == '__main__':
    model_name = 'chatglm2-6b'
    df = pandas.read_csv('turing.csv', encoding='GBK')
    query_names = [f'{df[df.columns[0]][i]}（{df[df.columns[1]][i]}）' for i in range(df.shape[0])]

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(f".\\models\\{model_name}", trust_remote_code=True)
    model = AutoModel.from_pretrained(f".\\models\\{model_name}", trust_remote_code=True).half().to('cuda:0')
    mid = time.time()
    print(f"Load model for {mid - start}s.")
    model = model.eval()
    meta_instruction = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话。现在我需要你扮演一个中文维基百科计算机领域编辑者，为所给图灵奖获得者编写生平简介，主要包含生卒年、出生地、国籍、毕业学院、主要事迹、获得奖项等信息。字数不限，但内容要完整。"

    query_names = ['理查德·斯特恩斯（Richard E. Stearns）']
    for query_name in query_names:
        query = meta_instruction + f"\n\n用户：{query_name}\n\nChatGLM-6B："
        response, history = model.chat(tokenizer, query)
        print(response)

    with jsonlines.open(f'outputs/turing_{model_name}.jsonl', 'w') as fout:
        for query_name in tqdm(query_names):
            query = meta_instruction + f"\n\n用户：{query_name}\n\nChatGLM-6B："
            response, history = model.chat(tokenizer, query)
            fout.write({'query': query, 'response': response})