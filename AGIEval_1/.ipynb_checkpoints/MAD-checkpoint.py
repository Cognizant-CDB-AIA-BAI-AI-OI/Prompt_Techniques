import os
import json
import openai
from tqdm import trange
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from multiprocessing.pool import ThreadPool
import threading


openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


    
@retry(wait=wait_random_exponential(min=1, max=80), stop=stop_after_attempt(10))
def gpt4(query, engine="gpt-4-32k"):
    
    engine = 'gpt-4-32k'
    
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
        temperature=0.7,
          max_tokens=6000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
        stop=["<|im_end|>"]
    )
    
    return response['choices'][0]['message']['content']



def get_res_from_other_agents(gpt4_res, get_prompt_for_agent):
    
    final_prompt = "These are responses from other agents."
    agent_c = 1
    for i in range(len(gpt4_res)):
        if i != get_prompt_for_agent:
            final_prompt += f"\n Agent{agent_c} :" + gpt4_res[i] 
            agent_c += 1
    
    final_prompt += "\n Based on the opinion of other agents , give an updated response"
    return final_prompt


def output(init_prompt, no_of_agents=4, no_of_rounds=4):
    
    pool = ThreadPool(no_of_agents)
    gpt4_res = pool.map(gpt4, [init_prompt]*no_of_agents)
    #gpt4_res = [gpt4(init_prompt) for i in range(no_of_agents)]
    
    
    folder_name = "4Model_multi_agent_output"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    gpt4_1_file = os.path.join(folder_name,"gpt4_1.txt")
    
    ## Initial prompt
    with open(gpt4_1_file, 'w') as file:
        file.write("Initial prompt >>>>>>>> "+ init_prompt)
        file.write('*'*500)
        file.write("\n\n")
    
    for i in trange(1, no_of_rounds):
        gpt4_input = [get_res_from_other_agents(gpt4_res, i) for i in range(no_of_agents)]

        pool = ThreadPool(no_of_agents)
        gpt4_res = pool.map(gpt4, gpt4_input)
        #gpt4_res = [gpt4(inp) for inp in gpt4_input]
        
        with open(gpt4_1_file, 'a') as file:
            file.write("\nAfter iteration >>>>>>>>>>>>>>>> "+ str(i))
            file.write("\n")
            file.write(gpt4_res[0])
            file.write("\n")
            file.write('~'*500)
            file.write("\n")

    with open(gpt4_1_file, 'r') as file:
        return_text = file.read()
    
    return gpt4_res[0]


def get_res_from_other_agents_chinese(gpt4_res, get_prompt_for_agent):
    
    final_prompt = "这些是其他代理的回应."
    agent_c = 1
    for i in range(len(gpt4_res)):
        if i != get_prompt_for_agent:
            final_prompt += f"\n Agent{agent_c} :" + gpt4_res[i] 
            agent_c += 1
    
    final_prompt += "\n 根据其他代理的意见，给出更新后的回应"
    return final_prompt


def output_chinese(init_prompt, no_of_agents=4, no_of_rounds=4):
    
    pool = ThreadPool(no_of_agents)
    gpt4_res = pool.map(gpt4, [init_prompt]*no_of_agents)
    #gpt4_res = [gpt4(init_prompt) for i in range(no_of_agents)]
    
    
    folder_name = "4Model_multi_agent_output"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    gpt4_1_file = os.path.join(folder_name,"gpt4_1.txt")
    
    ## Initial prompt
    with open(gpt4_1_file, 'w') as file:
        file.write("Initial prompt >>>>>>>> "+ init_prompt)
        file.write('*'*500)
        file.write("\n\n")
    
    for i in trange(1, no_of_rounds):
        gpt4_input = [get_res_from_other_agents_chinese(gpt4_res, i) for i in range(no_of_agents)]

        pool = ThreadPool(no_of_agents)
        gpt4_res = pool.map(gpt4, gpt4_input)
        #gpt4_res = [gpt4(inp) for inp in gpt4_input]
        
        with open(gpt4_1_file, 'a') as file:
            file.write("\nAfter iteration >>>>>>>>>>>>>>>> "+ str(i))
            file.write("\n")
            file.write(gpt4_res[0])
            file.write("\n")
            file.write('~'*500)
            file.write("\n")

    with open(gpt4_1_file, 'r') as file:
        return_text = file.read()
    
    return gpt4_res[0]