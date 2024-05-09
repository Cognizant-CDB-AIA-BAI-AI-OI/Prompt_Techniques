import ast
import signal
import astunparse
from check_correctness import check_correctness
from typing import List, NamedTuple, Tuple
import os


class ExecuteResult(NamedTuple):
    is_passing: bool
    feedback: str
    state: Tuple[bool]


class PyExecutor:
    
    def execute(self, 
                program: str, 
                program_id: str,
                num_tests: int,
                timeout: int = 5) -> ExecuteResult:

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        state = []
        for ntests in range(1, num_tests+1):
            
            path = f"/home/jupyter/Satya/All_prompt_tech/data/{program_id}/"
            
            input_file = path + f"/{ntests}.in" if "1.in" in os.listdir(path) else  path + f"/I.{ntests}"
            output_file = path + f"/{ntests}.out" if "1.in" in os.listdir(path) else  path + f"/O.{ntests}"
            pred_file = "/home/jupyter/Satya/All_prompt_tech/Outputs/pred"
            
            res = check_correctness(program = program,
                                      test_num = ntests,
                                      timeout = 5,
                                      memory_limit = 200,
                                      input_file = input_file,
                                      pred_file = pred_file, 
                                      output_file = output_file
                                      )
            
            with open(input_file, 'r') as f:
                input_str = f.read()
            with open(output_file, 'r') as f:
                output_str = f.read()
            
            if res['result_type'].value == 1:  # success.
                if len(input_str)>500 or len(output_str)>500:
                    success_tests += [f"Trimmed Test Input:\n{input_str[:500]}\nTrimmed Test Output:\n{output_str[:500]}\n\n"]
                else:
                    success_tests += [f"Test Input:\n{input_str}\nTest Output:\n{output_str}\n\n"]
                state.append(True)
            else:
                if len(input_str)>500 or len(res['status'])>500:
                    failed_tests += [f"Trimmed Test Input:\n{input_str[:500]}\n# output:\n{res['status'][:500]}\n\n"]
                else:
                    failed_tests += [f"Test Input:\n{input_str}\n# output:\n{res['status']}\n\n"]
                is_passing = False
                state.append(False)
        

        state = tuple(state)

        feedback = "Tests passed:"
        for test in success_tests:
            feedback += f"\n{test}\n"
        feedback += "\n\nTests failed:"
        for test in failed_tests:
            feedback += f"\n{test}\n"
            
        return ExecuteResult(is_passing, feedback, state)

#     def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
#         """
#         Evaluates the implementation on Human-Eval Python.

#         probably should be written in a dataset-agnostic way but not now
#         """
#         code = f"""{func}

# {test}

# check({name})
#     """
#         try:

#             function_with_timeout(exec, (code, globals()), timeout)

#             return True
#         except Exception:
#             return False


if __name__ == "__main__":
    pass
    # Test the function
    func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
    print(PyExecutor().execute(func, tests, timeout=1))