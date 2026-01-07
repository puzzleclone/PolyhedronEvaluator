"""
LLM Evaluation Framework Main Module

This module provides the main evaluation logic and routing for assessing
Large Language Model performance on diverse question types and answer formats.
"""

import re
import json
from typing import List, Optional, Union, Any

import json_repair
from functools import lru_cache

try:
    from extract_answer import extract_answer
    from eval_utils import parse_latex_table
    from eval_multiple_choice import option_equal, multi_answers_MCQ
    from eval_nominal import nominal_equal , strict_nominal_equal
    from eval_numeral import numeral_equal
    from eval_array import (order_array_equal, unorder_array_equal,
                           order_number_array_equal, unorder_number_array_equal,
                           subset_array_equal)
except ImportError:
    from .extract_answer import extract_answer
    from .eval_utils import parse_latex_table
    from .eval_multiple_choice import option_equal, multi_answers_MCQ
    from .eval_nominal import nominal_equal , strict_nominal_equal
    from .eval_numeral import numeral_equal
    from .eval_array import (order_array_equal, unorder_array_equal,
                           order_number_array_equal, unorder_number_array_equal,
                           subset_array_equal)


DEBUG = False
DEBUG_INFO = []
from fractions import Fraction  # Convert float to fraction for debugging
from itertools import permutations


def is_option(ref: str) -> bool:
    """
    Check if string matches option format patterns.
    
    Supports formats:
    1. (A), (B), etc.
    2. A), B), etc.
    3. A., B., etc.
    4. A:, B:, etc.
    5. Single letter A-Z
    
    Args:
        ref: String to check
        
    Returns:
        True if string matches option format, False otherwise
    """
    if len(ref) == 1 and ref.isalpha():
        return True
    
    if len(ref) > 1:
        if ref[0] == '(' and ref[1].isalpha() and len(ref) > 2 and ref[2] == ")":
            return True
        if ref[0].isalpha() and ref[1] in '.:)':
            return True
    
    return False


def is_valid_answer_string(s: str) -> bool:
    """
    Check if string contains only valid answer characters.
    
    Args:
        s: String to validate
        
    Returns:
        True if string contains only numbers, letters, or Chinese characters
    """
    # Pattern matches numbers, English letters, or Chinese characters
    pattern = r'^[0-9a-zA-Z\u4e00-\u9fa5]+$'
    return bool(re.match(pattern, s))


def is_multiple_query(ref: str) -> Optional[List[str]]:
    """
    Check if string represents multiple queries and split them.
    
    Supports formats:
    1. A\nB\nC
    2. A B C
    3. A, B, C
    4. 1. A, 2. B, 3. C
    
    Args:
        ref: Reference string to check
        
    Returns:
        List of split queries if multiple query format detected, None otherwise
    """
    separator = ['\n', ' ', ',', ';']

    for si in separator:
        if si in ref:
            split_res = [si.strip() for si in ref.split(si)]
            if all([is_valid_answer_string(si) for si in split_res]):
                return split_res
    return None

@lru_cache(maxsize=10000)
def split_and_strip(text: str) -> List[str]:
    """
    Split and strip text based on common separators.
    
    Args:
        text: Input text to split and strip
        
    Returns:
        List of stripped text parts
    """
    if text.startswith('[') and text.endswith(']'):
        try:
            return json_repair.loads(text)
        except:
            pass

    if "\\begin{" in text:
        return parse_latex_table(text)

    # Split using regex pattern for common separators
    parts = re.split(r'[,;、，；和或]', text)
    # If no splitting occurred, try backslash
    if len(parts) <= 1:
        parts = text.split("\\")
    result = [part.strip() for part in parts if part.strip()]
    return result


def parseArray(ref: str, pre: str) -> tuple[List[Any], List[Any]]:
    """
    Parse array strings into structured data.
    
    Args:
        ref: Reference array string
        pre: Prediction array string
        
    Returns:
        Tuple of (reference_data, prediction_data) as lists
    """
    ref_clean = ref.replace("(", "[").replace(")", "]").replace("\\", "").lstrip("#")
    pre_initial = pre.replace("(", "[").replace(")", "]").replace("“", '"').replace("”", '"').lstrip("#")
    pre_clean = pre_initial.replace("\\", "")
    if pre_clean.startswith('{[') and pre_clean.endswith(']}'):
        pre_clean = '[' + pre_clean[1:-1] + ']'
    try:
        ref_t = json.loads(ref_clean)
        pre_t = json.loads(pre_clean)
    except (SyntaxError, ValueError):
        try:
            ref_t = json_repair.loads(ref_clean)
            # if pre.startswith("\\begin{"):
            if "\\begin{" in pre:
                pre_t = parse_latex_table(pre_initial)
            else:
                pre_t = json_repair.loads(pre_clean)
                if len(re.split(r'\W+', str(pre_t))) <= len(re.split(r'\W+', pre_clean))//1.1:
                    new_content = "[" + pre_clean + "]"
                    pre_t = json_repair.loads(new_content)
        except json.JSONDecodeError:
            return [], []

    # 兜底：保证都是 list，避免后面 str + list 之类的错误
    if not isinstance(ref_t, list):
        ref_t = [ref_t]
    if not isinstance(pre_t, list):
        if isinstance(pre_t, dict): # dict -> list
            pre_t = list(pre_t.values())
        else:
            pre_t = [pre_t]

    ref_len = len(ref_t)
    pre_len = len(pre_t)
    if pre_len < ref_len:
        pre_t = pre_t + [None] * (ref_len - pre_len)
    return ref_t, pre_t


def average(scores: list[int|float]):
    return sum(scores) / max(len(scores), 1)


def comput_inner_array_score(pre_t: List[Any], ref_t: List[Any], f_score, score_type: str, array_type: str, indent=0) -> Union[bool, float]:
    
    if len(array_type) == 0:
        return f_score(pre_t, ref_t, score_type)

    if array_type[0] == 'o':  # ordered outer array
        inner_scores = []
        for index, ri in enumerate(ref_t):
            if index >= len(pre_t) or not isinstance(pre_t[index], list):
                inner_scores.append(0.0 if score_type == 'soft' else False)
                continue
            inner_scores.append(comput_inner_array_score(pre_t[index], ri, f_score, score_type, array_type[1:], indent=indent+1))

        # If soft, average the scores; if hard, all must be True
        score = average(inner_scores) if score_type == 'soft' else all(inner_scores)
        if DEBUG:
            indent_str = indent * "+" + " " if indent > 0 else ""
            DEBUG_INFO.append((indent_str + "pre_t:", pre_t, "  inner_scores:", *[Fraction(si).limit_denominator(100) for si in inner_scores], "  answer_final_score:", Fraction(score).limit_denominator(100)))
        return score

    elif array_type[0] == 'u':  # unordered outer array
        from itertools import permutations
        all_permutations = permutations(pre_t)
        possible_scores = set()
        for perm in all_permutations:
            score = comput_inner_array_score(list(perm), ref_t, f_score, score_type, 'o' + array_type[1:])
            possible_scores.add(score)
            if score == 1.0: # Perfect score found
                break  # No need to check further if perfect score found
        best_score = max(possible_scores)
        if DEBUG:
            indent_str = indent * "+" + " " if indent > 0 else ""
            DEBUG_INFO.append((indent_str + "best_score:", Fraction(best_score).limit_denominator(100)))
        return best_score


@lru_cache(maxsize=10000)
def eval_router(pre: str, ref: str, eval_type: Optional[str] = None, score_type: str = 'hard') -> bool:
    """
    Route evaluation to appropriate evaluator based on type.
    
    Args:
        pre: Predicted answer
        ref: Reference (ground truth) answer
        eval_type: Evaluation type. Supported types:
            - "nominal": Text-based answers, converting word numbers to digits and ignoring punctuation and case
            - "strict_nominal": Strict text-based answers with longest common substring scoring
            - "numeral": Numerical computations
            - "option": Multiple choice options with single correct answer
            - "multi_options": Multiple choice options with multiple correct answers
            - "ordered array": Sequence-sensitive lists (same as "oa_nominal"), eg: [1,2,3]  ; A,B,C ; 
            - "unordered array": Order-independent sets (same as "ua_nominal")
            - "subset": Subset relationship validation
            - "x*a_y": Nested array evaluation, where x indicates Ordered/Unordered array, which can be repeated for multiple levels; y indicates the type of inner array elements (e.g. nominal, numeral). For example:
                - "ooa_numeral": Ordered outer, ordered inner arrays (numerical)
                - "oua_nominal": Ordered outer, unordered inner arrays (nominal)
                - "uoa_nominal": Unordered outer, ordered inner arrays (nominal)
                - "oooa_numeral": Ordered → ordered → ordered arrays (numerical)
        score_type: The strictness level (hard or soft) of evaluation. 
            
    Returns:
        True if prediction matches reference according to eval_type, False otherwise
    """
    if not pre or not ref:
        return False

    pre = pre.strip()
    ref = ref.strip()
    
    if eval_type:
        if eval_type == "option":
            return option_equal(pre, ref)
        if eval_type == "nominal":
            return nominal_equal(pre, ref)
        if eval_type == "strict_nominal":
            return strict_nominal_equal(pre, ref, score_type)
        if eval_type == "numeral":
            return numeral_equal(pre, ref, score_type)
        if eval_type == "multi_options":
            return multi_answers_MCQ(pre, ref, score_type)
        if eval_type.startswith("order"):
            ref = split_and_strip(ref)
            pre = split_and_strip(pre)
            return order_array_equal(pre, ref, score_type)
        if eval_type.startswith("unorder"):
            ref = split_and_strip(ref)
            pre = split_and_strip(pre)
            return unorder_array_equal(pre, ref, score_type)
        if eval_type.startswith("subset"):
            ref = split_and_strip(ref)
            pre = split_and_strip(pre)
            return subset_array_equal(pre, ref)
        

        if "a_" in eval_type:  # Nested array evaluation
            f_score = None
            et_before, et_after = eval_type.split("a_")
            if et_before[-1] == "o":  # ordered inner array
                if et_after == "nominal":
                    f_score = order_array_equal
                elif et_after == "numeral":
                    f_score = order_number_array_equal
            elif et_before[-1] == "u":  # unordered inner array
                if et_after == "nominal":
                    f_score = unorder_array_equal
                elif et_after == "numeral":
                    f_score = unorder_number_array_equal

            if f_score is not None:
                if len(et_before) == 1:
                    ref_t = split_and_strip(ref)
                    pre_t = split_and_strip(pre)
                else:
                    ref_t, pre_t = parseArray(ref, pre)
                return comput_inner_array_score(pre_t, ref_t, f_score, score_type, et_before[:-1])


    # Auto-detection based on format
    if ref.startswith('[') and ref.endswith(']'):
        ref = json_repair.loads(ref)
        pre = split_and_strip(pre)
        return unorder_array_equal(pre, ref)

    if is_option(ref):
        return option_equal(pre, ref)

    ref_res = is_multiple_query(ref)
    if ref_res is not None:
        ref = ref_res
        pre = split_and_strip(pre)
        return order_array_equal(pre, ref)

    # Default to numerical evaluation
    return numeral_equal(pre, ref)


def get_n_elements(arr: List[Any], n: int, start: Optional[int] = None, fill_value: Any = None) -> List[Any]:
    """
    Get n elements from array starting from specified position, padding with fill_value if needed.
    
    Args:
        arr: Input array
        n: Number of elements to retrieve
        start: Starting index (0-based). If None, get last n elements.
               If positive, count from start; if negative, count from end.
        fill_value: Value to pad with if array has fewer than n elements
        
    Returns:
        List of n elements (may include fill_value for padding)
    
    Examples:
        arr = [1, 2, 3, 4]
        get_n_elements(arr, 3)  # [2, 3, 4]
        get_n_elements(arr, 3, start=2)  # [3, 4, None]
        get_n_elements(arr, 3, start=-2)  # [3, 4, None]
    """
    if start == None:
        res = arr[-n:]  # Get last n
    else:
        if start < 0:
            start = len(arr) + start
        res = arr[start:start+n]
    
    # Pad with fill_value if needed
    if len(res) < n:
        res = res + [fill_value] * (n - len(res))
    return res


@lru_cache(maxsize=10000)
def multi_query_answer_split(text: str, num_questions: int):
    """
    Split multi-question answer into parts.
    Args:
        text: Text containing multiple questions and answers. e.g. "1. Answer1 2. Answer2"
        num_questions: Number of questions in the text. e.g. 2
    
    Returns:
        List of answers for each question, or None if no matches found. e.g. ["Answer1", "Answer2"]
    """
    answers = []
    remaining_text = text
    for i in range(1, num_questions + 1):
        if f"{i}." not in remaining_text:
            return None
        # 构造模式：
        # 1\.   —— 当前题号
        # (.*?) —— 非贪婪捕获答案
        # (?=\s*\d+\.|$) —— 到下一个题号或字符串结尾停止
        pattern = rf"{i}\.\s*(.+?)(?=\s*{i+1}\.|$)"
        match = re.search(pattern, remaining_text, flags=re.S)
        if not match:
            answers.append(None)
            continue
        ans = match.group(1).strip()
        answers.append(ans)
        # 更新 remaining_text：从本次匹配结束位置开始
        remaining_text = remaining_text[match.end():]
    return answers


@lru_cache(maxsize=10000)
def split_pred(pred: str) -> List[str]:
    """
    Split prediction string into parts based on specified separators.
    
    Args:
        pred: Prediction string to split
    Returns:
        List of split parts
    """
    pred_split = []
    seps = ["====", ",", "，", ";", "；", "\\"]
    for sep in seps:
        if sep in pred:
            pred_split.extend([x.strip() for x in pred.split(sep) if x.strip()])
            break
    return pred_split if len(pred_split) > 0 else [pred]


def compute_score(solution_str: str, ground_truth: str, eval_type = None, score_type = "hard", debug=False) -> float:
    """
    Compute score for RL training.
    
    Args:
        solution_str: Complete model output with potential multiple answers
        ground_truth: Ground truth with potential multiple parts
        eval_type: Comma-separated evaluation types
        score_type: The strictness level (hard or soft) of evaluation. We recommend "soft" as the reward for RL training
        
    Returns:
        Float score between 0.0 and 1.0 (used as RL reward)
    """
    global DEBUG, DEBUG_INFO
    DEBUG = debug
    DEBUG_INFO = []

    if "boxed{" in ground_truth:
        ground_truth = extract_answer(ground_truth, last=True)

    # Calculate number of questions based on eval types
    if eval_type == None or eval_type == "":
        eval_type = [None]
    else:
        eval_type = re.split(r'[,，]', eval_type)
    qtype_len = len(eval_type)
    true_num, true_num2 = 0, 0
    try:
        ref_list = [ground_truth]
        if qtype_len > 1:
            separators = ["====", ";", "；", "\n"]  # Used to split ground truth
            for sep in separators:
                if sep in ground_truth:
                    ref_list = ground_truth.strip().split(sep)
                    break
        ref_list = get_n_elements(ref_list, qtype_len)

        pred_res = extract_answer(solution_str)  # Model answers
        if DEBUG:
            DEBUG_INFO.append(("pred_res (extracted answers):", pred_res))
        pred_res_len = len(pred_res)
        if pred_res_len == 0:
            return False if score_type == "hard" else 0.0

        if qtype_len > 1: # Multiple questions
            # Try to split answers based on question numbers if possible
            tmp = multi_query_answer_split(pred_res[-1], qtype_len)
            if tmp != None:
                pred_res = tmp
                pred_res_len = len(pred_res)
        if len(ref_list) > pred_res_len:
            tmp = []
            for pi in pred_res:
                tmp.extend(split_pred(pi))
            pred_res = tmp

        if pred_res_len > qtype_len and score_type == "soft": # Taking different answers to evaluate
            different_attemps = pred_res_len-qtype_len+1
            attempt_str_list = {}
            for si in range(different_attemps):
                pred_list = get_n_elements(pred_res, qtype_len, start=si)  # Get qtype_len answers from the start position
                pred_list_str = str(pred_list)
                if pred_list_str in attempt_str_list:
                    tmp = attempt_str_list.pop(pred_list_str)
                    attempt_str_list[pred_list_str] = tmp
                    continue
                check_res = [eval_router(pred_list[ref_i], ref_v, eval_type[ref_i], score_type) for ref_i, ref_v in enumerate(ref_list)]
                attempt_str_list[pred_list_str] = sum(check_res) # The higher the reward score for the answers that appear later
            different_answers_scores = list(attempt_str_list.values())
            if "_" in eval_type[0]:  # Only available for nested array evaluation
                # deduplicate scores in different_answers_scores, keep the last occurrence
                different_answers_scores = list(dict.fromkeys(reversed(different_answers_scores)))[::-1]
            ds_len = len(different_answers_scores)
            # The higher the reward score for the answers that appear later
            for di, ds in enumerate(different_answers_scores):
                different_answers_scores[di] = ds * (di + 1) / ds_len
            if DEBUG:
                DEBUG_INFO.append((f"pred_res: {pred_res}", "different_answers_scores (consider position):", *[Fraction(ds).limit_denominator(100) for ds in different_answers_scores]))
            true_num = max(different_answers_scores)*qtype_len/max(qtype_len, ds_len) # The more different answers there are, the lower the reward score
        else:
            pred_list = get_n_elements(pred_res, qtype_len)  # Get last qtype_len answers
            check_res = [eval_router(pred_list[ref_i], ref_v, eval_type[ref_i], score_type) for ref_i, ref_v in enumerate(ref_list)]
            true_num = sum(check_res)
            different_answers_scores = [true_num]
            if DEBUG:
                DEBUG_INFO.append((f"different_answers_scores (last {qtype_len} answers: {pred_list}):", "+".join([str(Fraction(cr).limit_denominator(100)) for cr in check_res]), "=", Fraction(true_num).limit_denominator(100)))
        
        # If multiple questions but partial answers correct, try splitting last prediction
        if qtype_len > 1 and true_num < qtype_len:
            pred_last_split = split_pred(pred_res[-1]) # Split last prediction
            if str(pred_last_split) != str([pred_res[-1]]):
                pred_res = pred_res[:-1] + pred_last_split  
                pred_list = get_n_elements(pred_res, qtype_len)  # Get last qtype_len answers
                check_res = [eval_router(pred_list[ref_i], ref_v, eval_type[ref_i], score_type) for ref_i, ref_v in enumerate(ref_list)]
                true_num = max(true_num, sum(check_res))
                if DEBUG:
                    DEBUG_INFO.append(("pred_list (after splitting last prediction):", pred_list, "  true_num (max with previous):", Fraction(true_num).limit_denominator(100)))
        
        # If single question but multiple predictions, consider concatenating multiple answers from pred_des before evaluating
        if qtype_len == 1 and pred_res_len > 1 and eval_type[0] not in ["nominal", "strict_nominal", "numeral", "option"]: # Only avalilable for array evaluation
            pred_list = [";".join(pred_res)]
            if DEBUG:
                DEBUG_INFO.append(("pred_list (concatenate all predictions):", pred_list))
            check_res = eval_router(pred_list[0], ref_list[0], eval_type[0], score_type)
            if check_res not in different_answers_scores:
                true_num2 = check_res
            elif DEBUG:
                DEBUG_INFO.append(("check_res (", Fraction(check_res).limit_denominator(100), ") already considered in different_answers_scores, so true_num2 = 0."))

    except Exception as e:
        import traceback
        error_info = traceback.format_exc()
        print("Error:", error_info)
        print("Args info:\n", {"pre": solution_str, "ref": ground_truth, "eval_type": eval_type})

    res_score = max(true_num, true_num2) / qtype_len
    if DEBUG:
        DEBUG_INFO.append(("true_num (consider ds_len):", Fraction(true_num).limit_denominator(100), "true_num2 (concatenate all predictions):", Fraction(true_num2).limit_denominator(100), "qtype_len:", qtype_len, "res_score:", Fraction(res_score).limit_denominator(100), "=", res_score))
        return res_score, DEBUG_INFO
    else:
        return res_score


def evaluation(prediction: str, ground_truth: str, eval_type = None) -> bool:
    """
    Evaluate LLM prediction with strict binary result.
    
    Args:
        prediction: Model's prediction
        ground_truth: Ground truth answer
        eval_type: Evaluation type specification
        
    Returns:
        True if all parts are correct, False otherwise
    """
    return False if compute_score(prediction, ground_truth, eval_type) < 1 else True


if __name__ == "__main__":
    
    data = {"pre": "\\boxed{[\"乌龟\", \"宠物猪\"]}, 还有 \\boxed{[\"乌龟\", \"宠物猪\"]}，以及\\boxed{[\"乌龟\", \"小猫\"]}", "ref": "[\"乌龟\", \"小猫s\"]", "eval_type": "ordered array"}

    data = {"pre": "COT: The answer is: \\boxed{[[(3, 2), (3, 3)], [(4, 1), (5, 1)], [(3, 5), (4, 5)], [(5, 3), (5, 4)]]}", "ref": "[[(3, 2), (3, 3)], [(4, 1), (5, 1)], [(3, 5), (4, 5)], [(5, 3), (5, 4)]]", "eval_type": "ooa_nominal"}

    data = {"pre": "The answer is \\boxed{1. B \\quad 2. 58}", "ref": "A====B", "eval_type": "option,option"}
    # data = {
    #     "pre": "\\boxed{A====[12.5, 13]}", 
    #     "ref": "A====[12.5, 125]", 
    #     "eval_type": "option,oa_numeral"
    # }
    res, debug_info = compute_score(data['pre'], data['ref'], data['eval_type'], "soft", debug=True)
    print(f"res: {res}")
    print(f"ref: {data['ref']}")
    print("Debug Info:")
    for info in debug_info:
        print("   ", *info)
    
    '''
    import time
    start_time = time.time()

    ref = "2"
    pre_list = ["\\boxed{Two}", "\\boxed{1} or \\boxed{2}", "\\boxed{1} or \\boxed{3} or \\boxed{2}", "\\boxed{1} or \\boxed{2} or \\boxed{3}", "\\boxed{1} or \\boxed{2} or \\boxed{3} or \\boxed{4}", "\\boxed{1} or \\boxed{3} or \\boxed{2} or \\boxed{4}"]
    for pi in pre_list:
        print(compute_score(pi, ref, "numeral", "soft"), end="\t")  # 1.0     0.5     0.3333333333333333      0.2222222222222222      0.125   0.1875
    print()
    for pi in pre_list:
        print(compute_score(pi, ref, "numeral", "hard"), end="\t")  # 1.0     1.0     1.0     0.0     0.0     0.0
    print()

    print(evaluation("This is text. 最终答案\n\\[\n\\boxed{18}\\quad\\boxed{10}\n\\]\n", "18,10", eval_type="numeral"))  # True

    # print(eval_router("({2014}^{2012} + {2013}^{2013})^{2011} + {2012}^{{2014}^{2012} + {2013}^{2013} - 1} \\square 2011", "2",eval_type="nominal"))
    print(eval_router("1/3", "33.33%"))  # True
    print(eval_router("1/3", "33.33%", eval_type="nominal"))  # False
    print(eval_router("[['a','b'], ['c', 'd'], ['d', 'e']]", "[['b','a'], ['d', 'c'],['d', 'e']]", eval_type="ooa_nominal"))  # False

    solution_str = "xixihh\\boxed{[[\"6\", \"7\"], [\"8\", \"9\"], [\"7\", \"9\"], [\"6\", \"9\"]]} \\boxed{A}\n$$"
    ground_truth = "[['6', '7'], ['8', '9'], ['7', '9'], ['6', '9']]====A"
    eval_type = "ooa_numeral,option"

    print(compute_score(solution_str, ground_truth, eval_type))  # 1.0

    solution_str = "xixihh\\boxed{({2014}^{2012} + {2013}^{2013})^{2011} + {2012}^{{2014}^{2012} + {2013}^{2013} - 1} \\square 2011} \\]"
    ground_truth = "2"
    eval_type = "numeral"
    # print(compute_score(solution_str, ground_truth, eval_type))  # 0.0
    

    solution_str = "\\boxed{A, 12.5}\n$$"
    ground_truth = "A====125"
    eval_type = "option,nominal"
    print(compute_score(solution_str, ground_truth, eval_type))  # 1.0 (as the eval_type of the second answer is nominal. If it is numeral, the score will be 0.5)

    print("---laoshi test cases---")
    pre = "\\boxed{(1,2)}",
    ref = "(1,2)"
    print(compute_score(pre, ref, 'ordered array'), compute_score(pre, ref, 'ordered array', 'soft')) # 0.0    0.333
    print(compute_score(pre, ref, 'unordered array'), compute_score(pre, ref, 'unordered array', 'soft')) # 1.0    1.0

    pre = "\\boxed{老15，老二}"
    ref = "2，15"
    print(compute_score(pre, ref, 'ordered array')) # 0.0
    print(compute_score(pre, ref, 'unordered array')) # 1.0

    print("--------------------")
    pre = "\\boxed{护理学,音乐学,化学,经济学,土木工程}"
    ref = "化学,音乐学,土木工程,经济学,护理学\n"
    print(compute_score(pre, ref, 'ordered array'), compute_score(pre, ref, 'ordered array', 'soft'))  # 0.0    0.4
    print(compute_score(pre, ref, 'unordered array'), compute_score(pre, ref, 'unordered array', 'soft'))  # 1.0    1.0

    print("--------------------")
    pre = "sdfsdf\\boxed{王桂芝};\\boxed{张凯};\\boxed{张俊};\\boxed{张玲};sdfsdf"
    ref = "王桂芝,张凯,张玲,张俊\n"
    print(compute_score(pre, ref, 'ordered array'), compute_score(pre, ref, 'ordered array', 'soft'))  # 0.0    0.5
    print(compute_score(pre, ref, 'unordered array'), compute_score(pre, ref, 'unordered array', 'soft'))  # 1.0    1.0

    pre = "sdfsdf\\boxed{[[55, '刘莹', 1.602], [52, '何丽', 18], [53, '雷刚', 19], [54, '许婷婷', 17.02]]};sdfsdf"
    ref = "[[55, '刘莹', 1.60], [52, '何丽', 18], [53, '雷刚', 19], [54, '许婷婷', 17]]\n"
    print(compute_score(pre, ref, 'ooa_nominal'), compute_score(pre, ref, 'ooa_nominal', 'soft'))  # 0.0    0.8333333333333333

    pre = "sdfsdf\\boxed{[[\"仓鼠\"],[\"仓鼠\", \"蝾螈\"],[\"仓鼠\", \"蝾螈\"],[\"仓鼠\"],[\"仓鼠\", \"蝾螈\"],[\"仓鼠\", \"乌龟\"],[\"蝾螈\", \"变色龙\"],[\"仓鼠\", \"蝾螈\"],[\"变色龙\"]]};sdfsdf"
    ref = "[['仓鼠'], ['仓鼠', '蝾螈'], ['仓鼠', '蝾螈'], ['仓鼠'], ['仓鼠', '蝾螈'], ['仓鼠', '乌龟22'], ['蝾螈22', '变色龙'], ['仓鼠', '蝾螈'], ['变色龙']]\n"
    print(compute_score(pre, ref, 'oua_nominal'), compute_score(pre, ref, 'oua_nominal', 'soft'))  # 0.0    0.8888888888888888
    
    execution_time = time.time() - start_time
    print(f"Code execution time: {execution_time:.6f} seconds")
    '''

    # with open(file_path, 'r') as file, open("grpo_extracted_logs_scores_ch.txt", 'w') as output_file:
    #     for index, line in enumerate(file):
    #         try:
    #             log_entry = json.loads(line.strip())
    #             pre = log_entry.get('pre', "")
    #             ref = log_entry.get('ref', "")
    #             eval_type = log_entry.get('eval_type', "")
    #             score = compute_score(pre, ref, eval_type, score_type='soft')
    #             output_file.write(f"{index}\t{score}\n")
    #             print(index, score)
    #             # print(f"Pre: {pre}\nRef: {ref}\nEval Type: {eval_type}\nScore: {score}\n")
    #         except json.JSONDecodeError:
    #             print("Invalid JSON format in extracted log:", line)