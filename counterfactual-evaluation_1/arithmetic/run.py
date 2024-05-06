import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from arithmetic.eval import get_label
import pandas as pd
from tqdm import tqdm
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
    

        


def load_data(data_file):
    return [line.strip() for line in open(data_file)]


def answer(expr, base):
    lhs, rhs = expr.split("+")
    lt, lo = lhs  # tens, ones
    rt, ro = rhs
    ones_sum = get_label(f"{lo}+{ro}", base)
    carry_over = len(ones_sum) > 1
    tens_sum_wo_carry = get_label(f"{lt}+{rt}", base)
    if carry_over:
        assert ones_sum[0] == "1"
        tens_sum_w_carry = get_label(f"{tens_sum_wo_carry}+1", base)
    else:
        tens_sum_w_carry = tens_sum_wo_carry
    assert get_label(expr, base) == tens_sum_w_carry + ones_sum[-1:]

    ret = f"We add the ones digits first. In base-{base}, {lo}+{ro}={ones_sum}. So the ones digit of the final sum is {ones_sum[-1:]}. "
    if carry_over:
        ret += f"We need to carry over the 1 to the tens place. "
    else:
        ret += f"We do not need to carry any digits over. "
    ret += f"Then we add the tens digits. In base-{base}, {lt}+{rt}={tens_sum_wo_carry}. "
    if carry_over:
        ret += f"Since we carried over the 1, {tens_sum_wo_carry}+1={tens_sum_w_carry}. "
    if len(tens_sum_w_carry) == 1:
        ret += f"So the tens digit of the final sum is {tens_sum_w_carry}. "
    else:
        ret += f"So the hundreds and tens digits of the final sum are {tens_sum_w_carry}. "
    ret += f"Putting the digits of the final sum together, we get \\boxed{{{tens_sum_w_carry}{ones_sum[-1:]}}}."
    return ret


def add_numbers_in_base(expr,  base):
    num1, num2 = expr.split("+")
    if base < 2 or base > 16:
        raise ValueError("Base must be between 2 and 16")

    # Convert the input numbers to base 10
    num1_base10 = int(str(num1), base)
    num2_base10 = int(str(num2), base)

    # Add the numbers in base 10
    result_base10 = num1_base10 + num2_base10

    # Convert the result back to the given base
    result = ""
    base_digits = "0123456789ABCDEF"
    while result_base10 > 0:
        remainder = result_base10 % base
        result = base_digits[remainder] + result
        result_base10 //= base

    return "\nAnswer:" + "\\" + "boxed{" + str(result) + "}"


def templatize(expr, base, cot=True, n_shots=0):
    if n_shots > 0:
        if cot:
            expr, demos = expr.split("\t")
            shots = demos.split(",")[:n_shots]
            assert len(shots) == n_shots
            context = "\n".join(f"{templatize(shot, base, cot)} {answer(shot, base)}" for shot in shots)
            return context + "\n" + templatize(expr, base, cot)
        else:
            expr, demos = expr.split("\t")
            shots = demos.split(",")[:n_shots]
            assert len(shots) == n_shots
            #print(shots)
            context = "\n".join(f"{templatize(shot, base, cot)} {add_numbers_in_base(shot, base)}" for shot in shots)
            return context + "\n" + templatize(expr, base, cot)
   
    
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if cot:
        return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? Let's think step by step, and end the response with the result in \"\\boxed{{result}}\"."
    else:
        return f"You are a mathematician. Assuming that all numbers are in base-{base} where the digits are \"{digits[:base]}\", what is {expr}? End the response with the result in \"\\boxed{{result}}\"."

    
def escape(str):
    assert "\t" not in str and "\\\\n" not in str and "\\\\r" not in str
    return str.replace("\\n", "\\\\n").replace("\n", "\\n").replace("\\r", "\\\\r").replace("\r", "\\r")


count = 0
for setting in ['zeroshot', 'zeroshot-CoT', 'fewshot', 'fewshot-CoT']:
    
    print(f"Setting {setting}", True if "CoT" in setting else False)
    for base in [8, 9, 10, 11, 16]:
        
        if setting == 'zeroshot' and base == 8:
            continue
        
        df = pd.DataFrame({
            "prompt": [],
            "response": [],
            "usage": []
            })
        
        if 'zero' in setting:
            data_file = f"arithmetic/data/0shot/base{base}.txt"
        else:
            data_file = f"arithmetic/data/icl/base{base}.txt"
         
        data = load_data(data_file)
        
        
        cot = True if "CoT" in setting else False
        n_shots = 1 if "few" in setting else 0
        
        templatized = [templatize(expr, base, cot=cot, n_shots=n_shots) for expr in data[:500]]
        
        if setting == 'zeroshot':
            templatized =  [temp + "\nDon't explain just give answer." for temp in templatized]

        for i, temp in tqdm(enumerate(templatized), total=len(templatized), desc=f"{setting}_{base}"):
            res, usage = gpt4(temp, i)
        
            df.loc[len(df)] = [temp, res, usage]
            
            if i%30==0:
                df.to_csv(f"arithmetic/output/gpt4_{setting}_{base}.csv", index=False)
                #exit()
                
        df.to_csv(f"arithmetic/output/gpt4_{setting}_{base}.csv", index=False)