import os
import json
import time
import openai
import jsonlines
from tenacity import (
    retry,
    RetryError,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def get_api_keys(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        api_lists = []
        for line in lines:
            api = line.strip().split('----')[-1]
            api_lists.append(api)
    return api_lists


if __name__ == '__main__':
    api_list = get_api_keys(r'/root/lmy/turing/api_key_for_turing.txt')
    api_index = 0

    in_path = r'/root/lmy/turing/turing_texts'
    out_path = r'/root/lmy/turing/extract_out'
    files = os.listdir(in_path)
    instruct = '请抽取出下文中的人物，外文名称，出生时间，出生地，逝世时间，逝世地，享年，国籍，毕业于，知名于，荣誉，研究领域，所属机构或公司等信息：'
    for file in files:
        with jsonlines.open(f'{out_path}/{file}', 'w') as fout:
            with open(f'{in_path}/{file}', 'r', encoding='utf8') as fin:
                start = time.time()
                for line in fin.readlines():
                    line = json.loads(line.strip())

                    openai.api_key = api_list[api_index]
                    print(openai.api_key)
                    # import pdb;pdb.set_trace()
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"{instruct}\n{line['response'][:500]}"}],
                    )
                    # time.sleep(2)

                    reply = response.choices[0]['message']['content'].strip()
                    fout.write({'query': line['query'], 'response': reply})
                    api_index += 1
                    if api_index >= len(api_list):
                        api_index -= len(api_list)
                print(f'{file}: {time.time() - start}s')
