from src import utils, dataset_loader
import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm

import openai
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


dataset_name_list = [
'gaokao-mathcloze',
'gaokao-geography',
'gaokao-physics',
'sat-en',
'sat-en-without-passage',
'gaokao-chemistry',
'gaokao-biology',
'sat-math',
'lsat-ar',
'gaokao-history',
'gaokao-chinese',
'aqua-rat',
'lsat-rc',
'gaokao-english',
'gaokao-mathqa',
'lsat-lr',
'logiqa-en',
'logiqa-zh',
'jec-qa-kd',
'jec-qa-ca',
'math'
]


setting_name_list = [
        #'few-shot', 
        #'few-shot-CoT',
        # 'zero-shot',
         'zero-shot-CoT'
    ]

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt4(query, engine="gpt-4-32k"):
    
    engine = 'gpt-4-32k'
    try:
        messages = [
                       {"role": "system", "content": "You are a helpful AI assistant."},
                   ]
        if isinstance(query, str):
            messages.append(
                {"role": "user", "content": query},
            )
        elif isinstance(query, list):
            messages += query
        else:
            raise ValueError("Unsupported query: {0}".format(query))
        response = openai.ChatCompletion.create(
            engine=engine,  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
            messages=messages,
            temperature=0,
            stop=["<|im_end|>"]
        )
    except TypeError as e:
        print("type error:", e)
        return {'choices': [{'message': {'content': ""}}]}
    
    return response # ['choices'][0]['message']['content']



def func(input_path, first_stage_output_path):
    
    js_list = utils.read_jsonl(input_path)
    content_list = [item["context"] for item in js_list]
    Res = []
    for i in tqdm(content_list):
        try:
            res = gpt4(i)
            #print(res)
            Res.append(res)
        except :
            Res.append({'choices': [{'message': {'content': ""}}]})
        
        utils.save_jsonl(Res, first_stage_output_path)
    
    

chat_mode = True
dataset_dir = "data/v1"
raw_prompt_path = "./data/few_shot_prompts.csv"
output_dir = "./outputs-zero-shot-cot/gpt-4-32k"
gpt_model = "gpt-4-32k"
#openai_api.default_engine = gpt_model
os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

"""
for dataset_name in dataset_name_list:
    for setting_name in setting_name_list:
        dataset = dataset_loader.load_dataset(
            dataset_name, setting_name, dataset_dir,
            prompt_path=raw_prompt_path, max_tokens=2048,
            end_of_example="<END>\n", verbose=True, chat_mode=chat_mode)
        
        # dataset = dataset[:10]
        input_path = os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.jsonl")
        utils.save_jsonl(dataset, input_path)
        # dataset = dataset[:10]
        # print(dataset[0]['context'])
        output_path = os.path.join(
            output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.jsonl')
        first_stage_output_path = os.path.join(
            output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.first_stage.jsonl')
        
        func(input_path, first_stage_output_path)
        print(dataset_name, " Done.")
"""

for dataset_name in dataset_name_list:
    for setting_name in setting_name_list:
        dataset = dataset_loader.load_dataset(
            dataset_name, setting_name, dataset_dir,
            prompt_path=raw_prompt_path, max_tokens=2048,
            end_of_example="<END>\n", verbose=True, chat_mode=chat_mode)
        
        input_path = os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.jsonl")
        # dataset = dataset[:10]
        # print(dataset[0]['context'])
        output_path = os.path.join(
            output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.jsonl')
        first_stage_output_path = os.path.join(
            output_dir, "outputs", f'predict.{gpt_model}.{dataset_name}.{setting_name}.first_stage.jsonl')

        first_stage_results = utils.read_jsonl(first_stage_output_path)
        second_stage_input = dataset_loader.generate_second_stage_input(
                                                                    dataset_name, dataset, first_stage_results)
        
        second_stage_input_path = os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.second_stage.jsonl")
        utils.save_jsonl(second_stage_input, second_stage_input_path)
        
        func(second_stage_input_path, output_path)
        print(dataset_name, " Done.")