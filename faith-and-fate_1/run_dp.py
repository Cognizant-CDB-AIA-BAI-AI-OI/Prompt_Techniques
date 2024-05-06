import json
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import threading
import argparse

import openai
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


def get_prompt(item, setting_name):
    
    prompt = """Given a sequence of integers, find a subsequence with the highest sum, such that
no two numbers in the subsequence are adjacent in the original sequence.
Output a list with "1" for chosen numbers and "2" for unchosen ones. If multiple
solutions exist, select the lexicographically smallest.

Finally give the final answer in the last line as "Final_answer": [..]\n"""
    
    if setting_name == "Zero-shot":
        prompt += f"\nDo not explain. just give final answer.\n\nQuestion = {item['raw_input']}\n"
        return prompt
        
    if setting_name == "Zero-shot-CoT":
        prompt += f"\nQuestion = {item['raw_input']}\n\nLet's solve step by step:"
            
    if setting_name == "Few-shot":
        prompt += f"""
Question:  [3, 2, 1, 5, 2]
Final_answer: [1, 2, 2, 1, 2].

Question: [3, -2, -3, -4]
Final_answer: [1, 2, 2, 2]

[Examples End]

Question: {item['raw_input']}
        """
        
    if setting_name == "Few-shot-CoT":
        
        prompt += f"""
Question: Let's solve input = [3, 2, 1, 5, 2].
Scratchpad: dp[4] = max(input[4], 0) = max(2, 0) = 2
dp[3] = max(input[3], input[4], 0) = max(5, 2, 0) = 5
dp[2] = max(dp[3], input[2] + dp[4], 0) = max(5, 1 + 2, 0) = 5
dp[1] = max(dp[2], input[1] + dp[3], 0) = max(5, 2 + 5, 0) = 7
dp[0] = max(dp[1], input[0] + dp[2], 0) = max(7, 3 + 5, 0) = 8
Finally, we reconstruct the lexicographically smallest subsequence that fulfills
the task objective by selecting numbers as follows. We store the result on a list
named "output".
Let can_use_next_item = True.
Since dp[0] == input[0] + dp[2] (8 == 3 + 5) and can_use_next_item == True, we
store output[0] = 1. We update can_use_next_item = False.
Since dp[1] != input[1] + dp[3] (7 != 2 + 5) or can_use_next_item == False, we
store output[1] = 2. We update can_use_next_item = True.
Since dp[2] != input[2] + dp[4] (5 != 1 + 2) or can_use_next_item == False, we
store output[2] = 2. We update can_use_next_item = True.
Since dp[3] == input[3] (5 == 5) and can_use_next_item == True, we store
output[3] = 1. We update can_use_next_item = False.
Since dp[4] != input[4] (2 != 2) or can_use_next_item == False, we store
output[4] = 2.
Reconstructing all together, output=[1, 2, 2, 1, 2].

Final_answer: [1, 2, 2, 1, 2]

[Example End]

Questions: Let's solve input = {item['raw_input']}
        """
        
    return prompt


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
            temperature=0
        )
    except TypeError as e:
        print("type error:", e)
        return ''
    
    return response['choices'][0]['message']['content']



def get_response(a):
    item, setting_name = a
    prompt = get_prompt(item, setting_name)
    response = gpt4(prompt)
    item['pred'] = response
    return item

        
# setting_names = 'Zero-shot', 'Zero-shot-CoT', 'Few-shot', 'Few-shot-CoT'
parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--setting', type=str, default="Default",
                    help="Name of config, which is used to load configuration under CompanyConfig/")

args = parser.parse_args()
        
# setting_names = 'Zero-shot', 'Zero-shot-CoT', 'Few-shot', 'Few-shot-CoT'
setting_name = args.setting


data = "dp/scratchpad"
output_dir = f"outputs_{setting_name}/{data}/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

batch_size = 6
file_names = [i for i in os.listdir(f"data/{data}/") if i.endswith('.json') or i.endswith('.jsonl')]
for name in file_names:
    
    #name = name.split(".")
    print("Dataset name:", name)
    # read the input file.
    with open(f"data/{data}/{name}", 'r') as f:
        if '.jsonl' in name:
            l= [json.loads(i) for i in f.read().split("\n") if i!='']
        else:
            
            l = json.loads(f.read())
    
    name = name.split(".")[0]
    #l = l[250:]
    final_json = []
    
    batch = []
    for item in tqdm(l):
        
        batch.append(item)
        if len(batch)<batch_size:
            continue
        
        pool = ThreadPool(batch_size)
        # Prepare your parameters for each item in the batch as tuples
        prepared_params = [(item, setting_name) for item in batch]
        
        item_responses = pool.map(get_response, prepared_params)
        
        #prompt = get_prompt(item, setting_name)
        #response = gpt4(prompt)
        #item['pred'] = response
        
        final_json.extend(item_responses)
        
        with open(f"{output_dir}/{name}_final.json", 'w') as f:
            json.dump(final_json, f)
            
        batch = []
        
    if len(batch)>0:
        
        pool = ThreadPool(batch_size)
        # Prepare your parameters for each item in the batch as tuples
        prepared_params = [(item, setting_name) for item in batch]
        
        item_responses = pool.map(get_response, prepared_params)
        
        #prompt = get_prompt(item, setting_name)
        #response = gpt4(prompt)
        #item['pred'] = response
        
        final_json.extend(item_responses)
        
        with open(f"{output_dir}/{name}_final.json", 'w') as f:
            json.dump(final_json, f)
            
        batch = []
        
