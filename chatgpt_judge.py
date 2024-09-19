import os
import json
import time
# import openai
import jsonlines


def get_api_keys(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        api_lists = []
        for line in lines:
            api = line.strip().split('----')[-1]
            api_lists.append(api)
    return api_lists


def get_knowledge(path):
    res = []
    with open(path, 'r', encoding='utf8') as f:
        now = ''
        for line in f.readlines():
            line = line.strip()
            if line == '':
                res.append(now)
                now = ''
                continue
            if now == '':
                now += f'人物：{line}'
            else:
                line = line.replace("\t", "：")
                now += ('\n' + line)
    return res


if __name__ == '__main__':
    # api_list = get_api_keys(r'/root/lmy/turing/api_key_for_turing.txt')
    api_index = 0

    knowledge_path = 'cnwiki_turing.txt'
    in_path = r'/root/lmy/turing/turing_texts'
    out_path = r'/root/lmy/turing/extract_out'
    # in_path = 'extract_processed_gpt'
    # out_path = 'test'
    # files = os.listdir(in_path)
    knowledge = get_knowledge(knowledge_path)
    for file in files:
        with jsonlines.open(f'{out_path}/{file}', 'w') as fout:
            with open(f'{in_path}/{file}', 'r', encoding='utf8') as fin:
                start = time.time()
                for i, line in enumerate(fin.readlines()):
                    line = json.loads(line.strip())

                    openai.api_key = api_list[api_index]
                    print(openai.api_key)
                    # import pdb;pdb.set_trace()
                    content = f'根据所给信息：\n{knowledge[i]}\n\n判断并标注下列信息中每一项是否存在不符，无不符标记为T（True），存在不符标记为F（False），不用修改：\n{line["response"]}\n\n判断原则：\n1）语义相同标记为T，语义不同标记为F；\n2）信息不完整，但被所给信息所包含也为T；\n3）包含了所有所给信息，但仍包含多余信息，标记为F；\n4）信息可由所给信息推理得到，标记为T\n\n'
                    print(content)
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": content}],
                    )
                    # time.sleep(2)

                    reply = response.choices[0]['message']['content'].strip()
                    fout.write({'query': line['query'], 'response': reply})
                    api_index += 1
                    if api_index >= len(api_list):
                        api_index -= len(api_list)
                print(f'{file}: {time.time() - start}s')
