import pandas as pd
import json
import time
from mcts import run_mcts
from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from tqdm import trange

df = pd.read_csv("../Zeroshot/predictions.csv")
data = json.loads(df.to_json(orient='records'))


for i in trange(8, len(data)):
    
    try:
        start_time = time.time()

        res = run_mcts(
            dataset = data[i:i+1],
            model_name = 'gpt-4-turbo',
            max_iters = 2,
            log_path = f'./Outputs/{data[i]["problem_id"]}.jsonl',
            verbose = True,
            n = 2)
        end_time = time.time()
        write_jsonl(f'./Outputs/res/{data[i]["problem_id"]}.jsonl', res, append=False)
    except:
        print(i)
        
    
    
    