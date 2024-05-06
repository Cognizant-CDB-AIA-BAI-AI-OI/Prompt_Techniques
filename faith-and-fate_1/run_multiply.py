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
    
    prompt = """To multiply two numbers, start by multiplying the rightmost digit of the
multiplicand by each digit of the multiplier, writing down the products and
carrying over any remainders. Repeat this process for each digit of the
multiplicand, and then add up all the partial products to obtain the final
result.

Finally give the final answer in the last line as "Final_answer": final answer\n"""
    
    if setting_name == "Zero-shot":
        prompt += f"Do not explain. just give final answer.\n\nQuestion: {item['question']}\n"
        return prompt
        
    if setting_name == "Zero-shot-CoT":
        prompt += f"\n\nQuestion: {item['question']}\n\nLet's perform the multiplication step by step:"
        return prompt
            
    if setting_name == "Few-shot":
        prompt = f"""
Given a two numbers you need to tell the product of these two numbers.

Finally give the final answer in the last line as "Final_answer": final answer.

Question: what's 22 times 2? 
Final_answer: 44.

Question: what's 78 times 8?
Final_answer: 624.

[Examples End]

Question: {item['question']}
        """
        
        return prompt
        
    if setting_name == "Few-shot-CoT":
        
        prompt += f"""
Questions: What is 28277 times 788?
Answer: Let's perform the multiplication step by step:

Let's multiply 28277 by the digit in the ones place of 788, which is 8.

1. Multiply 8 by the digit in the ones place of 28277, which is 7. This gives 7 x 8 = 56. Write down the result 6 and carry over the 5 to the next step.
2. Multiply 8 by the digit in the tens place of 28277, which is 7. Add the carryover from the previous step to account for this. This gives (7 x 8) + 5 = 61. Write down the result 1 and carry over the 6 to the next step.
3. Multiply 8 by the digit in the hundreds place of 28277, which is 2. Add the carryover from the previous step to account for this. This gives (2 x 8) + 6 = 22. Write down the result 2 and carry over the 2 to the next step.
4. Multiply 8 by the digit in the thousands place of 28277, which is 8. Add the carryover from the previous step to account for this. This gives (8 x 8) + 2 = 66. Write down the result 6 and carry over the 6 to the next step.
5. Multiply 8 by the digit in the ten-thousands place of 28277, which is 2. Add the carryover from the previous step to account for this. This gives (2 x 8) + 6 = 22. Write down the result 22.
6. The partial product for this step is A=226216 which is the concatenation of the digits we found in each step.

Now, let's multiply 28277 by the digit in the tens place of 788, which is 8.

7. Multiply 8 by the digit in the ones place of 28277, which is 7. This gives 7 x 8 = 56. Write down the result 6 and carry over the 5 to the next step.
8. Multiply 8 by the digit in the tens place of 28277, which is 7. Add the carryover from the previous step to account for this. This gives (7 x 8) + 5 = 61. Write down the result 1 and carry over the 6 to the next step.
9. Multiply 8 by the digit in the hundreds place of 28277, which is 2. Add the carryover from the previous step to account for this. This gives (2 x 8) + 6 = 22. Write down the result 2 and carry over the 2 to the next step.
10. Multiply 8 by the digit in the thousands place of 28277, which is 8. Add the carryover from the previous step to account for this. This gives (8 x 8) + 2 = 66. Write down the result 6 and carry over the 6 to the next step.
11. Multiply 8 by the digit in the ten-thousands place of 28277, which is 2. Add the carryover from the previous step to account for this. This gives (2 x 8) + 6 = 22. Write down the result 22.
12. The partial product for this step is B=226216 which is the concatenation of the digits we found in each step.

Now, let's multiply 28277 by the digit in the hundreds place of 788, which is 7.

13. Multiply 7 by the digit in the ones place of 28277, which is 7. This gives 7 x 7 = 49. Write down the result 9 and carry over the 4 to the next step.
14. Multiply 7 by the digit in the tens place of 28277, which is 7. Add the carryover from the previous step to account for this. This gives (7 x 7) + 4 = 53. Write down the result 3 and carry over the 5 to the next step.
15. Multiply 7 by the digit in the hundreds place of 28277, which is 2. Add the carryover from the previous step to account for this. This gives (2 x 7) + 5 = 19. Write down the result 9 and carry over the 1 to the next step.
16. Multiply 7 by the digit in the thousands place of 28277, which is 8. Add the carryover from the previous step to account for this. This gives (8 x 7) + 1 = 57. Write down the result 7 and carry over the 5 to the next step.
17. Multiply 7 by the digit in the ten-thousands place of 28277, which is 2. Add the carryover from the previous step to account for this. This gives (2 x 7) + 5 = 19. Write down the result 19.
18. The partial product for this step is C=197939 which is the concatenation of the digits we found in each step.

Now, let's sum the 3 partial products A, B and C, and take into account the position of each digit: A=226216 (from multiplication by 8), B=226216 (from multiplication by 8 but shifted one place to the left, so it becomes 2262160) and C=197939 (from multiplication by 7 but shifted two places to the left, so it becomes 19793900). The final answer is 226216 x 1 + 226216 x 10 + 197939 x 100 = 226216 + 2262160 + 19793900 = 22282276.

Final_answer: 22282276

[Example End]

Questions: {item['question']}
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



parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--setting', type=str, default="Default",
                    help="Name of config, which is used to load configuration under CompanyConfig/")

args = parser.parse_args()
        
# setting_names = 'Zero-shot', 'Zero-shot-CoT', 'Few-shot', 'Few-shot-CoT'
setting_name = "Few-shot-CoT"#args.setting


print(setting_name)

data = "multiplication/scratchpad"
output_dir = f"outputs_{setting_name}_1/{data}/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

batch_size = 6
file_names = [i for i in os.listdir(f"data/{data}/") if i.endswith('.json') or i.endswith('.jsonl')]
for name in ["scratchpad_4_by_3_1000prompts.json",
"scratchpad_1_by_1_prompts.json",
"scratchpad_5_by_5_1000prompts.json",
"scratchpad_2_by_1_prompts.json",
"scratchpad_3_by_1_1000prompts.json",
"scratchpad_5_by_3_1000prompts.json"]: #file_names:
        
    #name = name.split(".")
    print("Dataset name:", name, setting_name)
    # read the input file.
    with open(f"data/{data}/{name}", 'r') as f:
        if '.json' in name:
            l = json.loads(f.read())
        else:
            l= [json.loads(i) for i in f.read().split("\n") if i!='']
    
    name = name.split(".")[0]
    # l = l[:250]
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
        
