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
    
    if setting_name == "Zero-shot":
        prompt = item['prompt'].replace("Let's think step by step. Please first briefly talk about your reasoning and", "")
        return prompt
        
    if setting_name == "Zero-shot-CoT":
        prompt = f"{item['prompt']}"
            
    if setting_name == "Few-shot":
        prompt = f"""
This is a logic puzzle. There are 2 houses (numbered 1 on the left, 2 on the right), from the perspective of someone standing across the street from them. Each has a different person in them. They have different characteristics:
 - Each person has a unique name: arnold, eric
 - Everyone has a favorite smoothie: cherry, desert

1. Arnold is in the first house.
2. Arnold and the person who likes Cherry smoothies are next to each other.

show your final solution by filling the blanks in the below table.

$ House: ___ | Name: ___ | Smoothie: ___ |
$ House: ___ | Name: ___ | Smoothie: ___ | 

###

Final solution:
$ House: 1 | Name: arnold | Smoothie: desert |
$ House: 2 | Name: eric | Smoothie: cherry |

[Examples End]

"""
        prompt += item['prompt'].replace("Let's think step by step. Please first briefly talk about your reasoning and", "")
        
    if setting_name == "Few-shot-CoT":
        
        prompt = f"""
This is a logic puzzle. There are 3 houses (numbered 1 on the left, 3 on the
right). Each has a different person in them. They have different characteristics:
- Each person has a unique name: peter, eric, arnold
- People have different favorite sports: soccer, tennis, basketball
- People own different car models: tesla model 3, ford f150, toyota camry
1. The person who owns a Ford F-150 is the person who loves tennis.
2. Arnold is in the third house.
3. The person who owns a Toyota Camry is directly left of the person who owns a
Ford F-150.
4. Eric is the person who owns a Toyota Camry.
5. The person who loves basketball is Eric.
6. The person who loves tennis and the person who loves soccer are next to each
other.
Let's think step by step. Please first briefly talk about your reasoning and show
your final solution by filling the blanks in the below table.
$ House: ___ $ Name: ___ $ Sports: ___ $ Car: ___
$ House: ___ $ Name: ___ $ Sports: ___ $ Car: ___
$ House: ___ $ Name: ___ $ Sports: ___ $ Car: ___
Reasoning:
Step 1: First apply clue <Arnold is in the third house.> We know that The Name in
house 3 is arnold.
Step 2: Then combine clues: <The person who loves tennis and the person who loves
soccer are next to each other.> <The person who loves basketball is Eric.>
Unique Values Rules and the fixed table structure. We know that The Name in house
1 is eric. The FavoriteSport in house 1 is basketball. The Name in house 2 is
peter.
Step 3: Then apply clue <Eric is the person who owns a Toyota Camry.> We know
that The CarModel in house 1 is toyota camry.
Step 4: Then apply clue <The person who owns a Toyota Camry is directly left of
the person who owns a Ford F-150.> and Unique Values We know that The CarModel in
house 2 is ford f150. The CarModel in house 3 is tesla model 3.
Step 5: Then apply clue <The person who owns a Ford F-150 is the person who loves
tennis.> and Unique Values We know that The FavoriteSport in house 2 is tennis.
The FavoriteSport in house 3 is soccer.
The puzzle is solved.
Final solution:
$ House: 1 $ Name: Eric $ Sports: Basketball $ Car: Camry
$ House: 2 $ Name: Peter $ Sports: Tennis $ Car: Ford
$ House: 3 $ Name: Arnold $ Sports: Soccer $ Car: Tesla

[Example End]

{item['prompt']}
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


data = "puzzle/scratchpad"
output_dir = f"outputs_{setting_name}/{data}/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

batch_size = 5
file_names = [i for i in os.listdir(f"data/{data}/") if i.endswith('.json') or i.endswith('.jsonl')]
for name in file_names:
    
    if 'train' not in name:
        #name = name.split(".")
        print("Dataset name:", name)
        # read the input file.
        with open(f"data/{data}/{name}", 'r') as f:
            l= [json.loads(i) for i in f.read().split("\n") if i!='']

        #name = name.split(".")[0]
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

