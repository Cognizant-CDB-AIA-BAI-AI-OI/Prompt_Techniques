from tqdm import tqdm, trange
import json
import pandas as pd
import time
import openai

openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


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
    

        
with open("CoT_Test.json", 'r') as f:
    dataset = json.loads(f.read())
    
def get_zeroshot(item):
    
    context = item['context']
    prompt = context + "\n\nQuestion:\n" + item['question'] + "\n\nLet's think step by step\nFinally give answer in new line as Final_answer: your final answer(only specific key word asked in question)"
    
    return prompt, item['label']

def get_fewshot(item):
    
    prompt = """
[Context]:
1 Evelyn entered the cellar.
2 Charlotte entered the cellar.
3 Lucas entered the cellar.
4 Evelyn exited the cellar.
5 The shirt is in the bottle.
6 Lucas moved the shirt to the cupboard.
7 Charlotte exited the cellar.
8 Lucas exited the cellar.
9 Charlotte entered the cellar.

[Question]:
10 Where does Charlotte think that Lucas searches for the shirt?

Let's think step by step
Finally give answer in new line as Final_answer: your final answer(only specific key word asked in question)

[Answer]:
Step 1: Evelyn, Charlotte, and Lucas entered the cellar (points 1-3).
Step 2: Evelyn exited the cellar (point 4). At this moment, Charlotte and Lucas are still in the cellar.
Step 3: The shirt is in the bottle (point 5). It is not mentioned if Charlotte and Lucas saw the shirt being placed in the bottle, so we cannot assume they know its location.
Step 4: Lucas moved the shirt to the cupboard (point 6). Since Charlotte is in the cellar with Lucas, she witnesses him move the shirt to the cupboard.
Step 5: Charlotte exited the cellar (point 7). Now, Charlotte is no longer in the cellar, but she knows that Lucas moved the shirt to the cupboard.
Step 6: Lucas exited the cellar (point 8).
Step 7: Charlotte entered the cellar again (point 9).

Now, to answer the question "Where does Charlotte think that Lucas searches for the shirt?"

Based on the information provided, Charlotte last saw Lucas move the shirt to the cupboard. Therefore, she would believe that Lucas searches for the shirt in the cupboard.

Final_answer: cupboard.

[End of Examples]

"""
    
    context = item['context']
    prompt += "[Context]:\n" + context + "\n\n[Question]:\n" + item['question'] + """\n\nLet's think step by step
Finally give answer in new line as Final_answer: your final answer(only specific key word asked in question)\n\n[Answer]:"""
    
    return prompt, item['label']

final = []
for i, item in tqdm(enumerate(dataset), total=len(dataset)):
    
    prompt, label = get_fewshot(item)
    res, usage  = gpt4(prompt, i)
    item['pred'] = res
    item['usage'] = usage
    final.append(item)
    
    with open("gpt4_pred_fewshotCoT.json", 'w') as f:
        json.dump(final, f)
