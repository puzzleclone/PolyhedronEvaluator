import json
import os

from main import compute_score, evaluation
import time


def load_test_cases(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_test_cases(group: str = None, case_idx: int = None):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cases_path = os.path.join(cur_dir, "test_cases.json")
    all_cases = load_test_cases(cases_path) 
    inconsistent_num = 0

    if group is not None and group in all_cases:
        if case_idx is not None:
            cases = all_cases[group]
            if 0 <= case_idx < len(cases):
                cases = [cases[case_idx]]
            else:
                print(f"Invalid case_idx (out of range): {case_idx}")
                return
            all_cases = {group: cases}
        else:
            all_cases = {group: all_cases[group]}

    all_cases_len = 0
    for group_name, cases in all_cases.items():
        all_cases_len += len(cases)
        print(f"=== Group: {group_name} ({len(cases)} cases) ===")
        for idx, case in enumerate(cases):
            pre = case.get("pre", "")
            ref = case.get("ref", "")
            eval_type = case.get("eval_type", None)
            expected_score = float(case.get("expected_score", 0.0))
            expected_eval_res = case.get("expected_eval_res", 0.0)

            score, debug_info = compute_score(pre, ref, eval_type, score_type="soft", debug=True)
            eval_res = evaluation(pre, ref, eval_type)
            
            tolerance = 1e-7 
            if abs(score - expected_score) >= tolerance or eval_res != expected_eval_res:
                inconsistent_num += 1
                print(f"===================  Case {idx} (👇) Failed  ===================")
                print(f"eval_type: {eval_type}")
                print(f"ref: {ref}")
                print(f"expected_score: {expected_score}, expected_eval_res: {expected_eval_res}")
                print(f"computed_score: {score}, computed_eval_res: {eval_res}")

                print("Debug Info:")
                for info in debug_info:
                    print("   ", *info)
                print()
                # case["expected_score"] = round(score, 8)
                # case["expected_eval_res"] = eval_res
    
    if inconsistent_num == 0:
        print(f"\nAll {all_cases_len} test cases passed! 😊")
    else:
        print(f"\n{inconsistent_num} inconsistent case(s) found out of {all_cases_len} test cases. 😞")

    return all_cases
                

if __name__ == "__main__":
    start_time = time.time()
    all_cases = run_test_cases(group=None, case_idx=None)
    # run_test_cases(group="combination", case_idx=3)
    execution_time = time.time() - start_time
    print(f"Code execution time: {execution_time:.6f} seconds")

    # with open("test_cases.json", "w", encoding="utf-8") as f:
    #     json.dump(all_cases, f, ensure_ascii=False, indent=4)

''' Output after running the test cases:
=== Group: numeral (30 cases) ===
=== Group: nominal (28 cases) ===
=== Group: multiple_choice (16 cases) ===
=== Group: array (53 cases) ===
=== Group: combination (9 cases) ===

All 136 test cases passed! 😊
Code execution time: 0.941148 seconds
'''
