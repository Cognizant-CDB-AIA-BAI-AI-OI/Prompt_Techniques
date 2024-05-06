from tqdm import tqdm, trange
import json
import pandas as pd
import time
import openai

openai.api_type = "azure"
openai.api_base = "https://mrityunjoypanday-gpt4.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "d020880e0119447fbde26d3dcfb09bd7"


def gpt4(query, i, counter=0):
    
    try:
        messages = [
                       {"role": "system", "content": "You are a helpful AI assistant."},
                       {"role": "user", "content": query}
                   ]
        
        response = openai.ChatCompletion.create(
            engine="gpt-4-32k",  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
            messages=messages,
            temperature=0
        )
        return response['choices'][0]['message']['content'], response['usage']
    
    except:
        if counter < 5:
            time.sleep(5)
            return gpt4(prompt, i, counter + 1)
        else:
            print("exception at:" + str(i))
            return "", {
    "completion_tokens": 0,
    "prompt_tokens": 0,
    "total_tokens": 0
}
    

        
def get_zeroshot(item):
    
    context = item['context']
    questions = "\n".join([" ".join(i.split(" ")[1:]) for i in item['question']])
    prompt = context + "\n\nQuestions:\n" + questions + "\n\nGive answer in json format where question is key and answer is value(only specific value).\n"
    
    return prompt, [" ".join(i.split(" ")[:-1]) for i in item['label']]

def get_Fewshot(item):
    
    preprompt = """
[Context]:
1 Abigail entered the closet.
2 Abigail hates the peach
3 Mila entered the closet.
4 The cap is in the container.
5 Abigail exited the closet.
6 Mila moved the cap to the cupboard.

[Questions]:
Where was the cap at the beginning?
Where will Mila look for the cap?
Where does Mila think that Abigail searches for the cap?
Where is the cap really?
Where will Abigail look for the cap?
Where does Abigail think that Mila searches for the cap?

[Answer]:
{
  "Where was the cap at the beginning?": "container",
  "Where will Mila look for the cap?": "cupboard",
  "Where does Mila think that Abigail searches for the cap?": "container",
  "Where is the cap really?": "cupboard",
  "Where will Abigail look for the cap?": "container",
  "Where does Abigail think that Mila searches for the cap?": "container"
}

"""
    
    context = item['context']
    questions = "\n".join([" ".join(i.split(" ")[1:]) for i in item['question']])
    prompt = "['Context']:" + context + "\n\n[Questions]:\n" + questions + "\n\n[Answer]:\n"
    
    return preprompt + prompt, [" ".join(i.split(" ")[:-1]) for i in item['label']]
        
        
with open("test.json", 'r') as f:
    l = json.loads(f.read())
    
    
final = []
for i, item in tqdm(enumerate(l), total=len(l)):
    prompt, label = get_Fewshot(item)
    res, usage  = gpt4(prompt, i)
    item['pred'] = res
    item['usage'] = usage
    final.append(item)
    
    with open("gpt4_pred_fewshot.json", 'w') as f:
        json.dump(final, f)
