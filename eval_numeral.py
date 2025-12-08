"""
Numerical Evaluation Module

This module provides comprehensive functionality for evaluating numerical answers,
including support for mathematical expressions, matrices, equations, and various
number formats with tolerance-based comparison.
"""

import re
import regex, math
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
import time, ctypes

from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from sympy import simplify, N
from latex2sympy2 import latex2sympy

try:
    from eval_utils import convert_word_number
except ImportError:
    from .eval_utils import convert_word_number


@lru_cache(maxsize=10000)
def strip_string(string: str) -> str:
    """
    Strip and normalize mathematical expression strings.
    
    Args:
        string: Mathematical expression string
        
    Returns:
        Normalized string ready for evaluation
    """
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    # replace \\ with \
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")
    # string = string.replace("\\\\", "\\")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # for unit_text in unit_texts:
    #     # use regex, the prefix should be either the start of the string or a non-alphanumeric character
    #     # the suffix should be either the end of the string or a non-alphanumeric character
    #     _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
    #     if _string != "":
    #         string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    # convert word number to digit
    string = convert_word_number(string)

    # replace "\\text{...}" to "..."
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    # remove percentage
    string = string.replace("\\%", "")
    # string = string.replace("\%", "")
    string = string.replace("%", "")
    
    months = r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b"
    string = re.sub(months, "", string, flags=re.IGNORECASE)

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    # string = string.replace("\\cdot", "")
    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

@lru_cache(maxsize=10000)
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

@lru_cache(maxsize=10000)
def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

@lru_cache(maxsize=10000)
def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


@lru_cache(maxsize=10000)
def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


@lru_cache(maxsize=10000)
def extract_numbers(text: str) -> str:
    """从文本中提取所有数值（包括分数、小数）并以分号分隔"""
    text = convert_word_number(text)
    # 匹配数字模式：整数、小数、分数、百分数等
    number_pattern = r'\d+\.?\d*\/?\d*\.?\d*%?'
    numbers = re.findall(number_pattern, text)
    return ';'.join(numbers)


@lru_cache(maxsize=10000)
def get_tol_max_diff(reference: float, rel_tol: float, abs_tol: float = 1e-4):
    """
    Compute tolerance and maximum difference for scoring.

    Args:
        reference: ground truth number
        rel_tol: relative tolerance 
        abs_tol: absolute tolerance
    
    Returns:
        tol: absolute tolerance
        max_diff: maximum difference for soft scoring
    """
    tol = max(abs(reference) * rel_tol, abs_tol)
    max_diff = max(abs(reference) / 100, abs_tol)  # beyond which soft reward is zero
    return tol, max_diff


@lru_cache(maxsize=10000)
def numeric_equal(
    prediction: float,
    reference: float,
    score_type: str = "hard",
    rel_tol: float = 1e-7,
    exp_k: int = 6
):
    """
    Unified scoring function for both evaluation (hard)
    and RL reward (soft).

    Args:
        prediction: model output number
        reference: ground truth number
        score_type: "hard" (True/False) or "soft" (0~1 reward)
        rel_tol: relative tolerance
        exp_k: exponent factor for soft scoring, higher means sharper decay

    Returns:
        For hard: True / False
        For soft: float reward [0,1]
    """

    diff = abs(prediction - reference)
    tol, max_diff = get_tol_max_diff(reference, rel_tol)

    # print(f"Diff: {diff}, Tol: {tol}, Max Diff: {max_diff}")
    #  HARD MODE
    if score_type == "hard":
        if diff > 0 and (prediction == 0 or reference == 0):
            return False
        return diff < tol
    #  SOFT MODE
    return 0.0 if diff > max_diff else math.exp(-exp_k * diff / max_diff)


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


@lru_cache(maxsize=10000)
def symbolic_equal(a, b, score_type="hard"):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\").replace("\\frac{", "+\\frac{"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if a == b:
            return True
    except:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    try:
        return numeric_equal(float(N(a)), float(N(b)), score_type=score_type)
    except:
        pass

    return False


def call_with_timeout(func, *args, timeout=3, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            for thread in list(executor._threads):
                while thread.is_alive():
                    for e in [SystemExit, KeyboardInterrupt]:
                        if thread.is_alive():
                            tid = ctypes.c_long(thread.ident)
                            if not ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(e)):
                                raise ValueError("Invalid thread id")
                            executor.shutdown(wait=False)
                            time.sleep(0.1)
                        else:
                            break
            return False


@lru_cache(maxsize=10000)
def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    score_type: str = "hard",
    timeout: bool = True,
    depth: int = 0,
    max_depth: int = 5
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    
    if depth > max_depth:
        return False

    reference = reference.strip()
    prediction = prediction.strip()
    if prediction == "" or reference == "":
        return False

    if prediction.lower() == reference.lower():
        return True

    max_score = set()
    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference, reference / 100, reference * 100]
            else:
                gt_result = [reference]
            for index_i, item in enumerate(gt_result):
                try:
                    max_score.add(numeric_equal(prediction, item, score_type=score_type))
                except Exception as e:
                    # print(f"Exception occurred in numeric_equal (1): {e}")
                    pass
    except Exception as e:
        # print(f"Exception occurred in numeric_equal (2): {e}")
        pass

    if max(max_score, default=0) == 1:
        return True if score_type == "hard" else 1.0

    try:  # 2. symbolic equal
        if prediction.count("=") == 1 and reference.count("=") == 1:
            pred = prediction.split("=")
            pred = f"{pred[0].strip()} - ({pred[1].strip()})"
            ref = reference.split("=")
            ref = f"{ref[0].strip()} - ({ref[1].strip()})"
            max_score.add(symbolic_equal(pred, ref, score_type=score_type)) 
            max_score.add(symbolic_equal(f"-({pred})", ref, score_type=score_type))
        elif (
            prediction.count("=") == 1
            and len(prediction.split("=")[0].strip()) <= 2
            and "=" not in reference
        ):
            max_score.add(math_equal(
                prediction.split("=")[1], reference, include_percentage, score_type=score_type, timeout=timeout, depth=depth+1, max_depth=max_depth
            ))
        elif (
            reference.count("=") == 1
            and len(reference.split("=")[0].strip()) <= 2
            and "=" not in prediction
        ):
            max_score.add(math_equal(
                prediction, reference.split("=")[1], include_percentage, score_type=score_type, timeout=timeout, depth=depth+1, max_depth=max_depth
            ))

    except Exception as e:
        # print(f"Exception occurred in symbolic_equal (1): {e}")
        pass

    if max(max_score, default=0) < 1:
        try:
            if timeout:
                max_score.add(call_with_timeout(symbolic_equal, prediction, reference, score_type))
            else:
                max_score.add(symbolic_equal(prediction, reference, score_type))
        except Exception as e:
            # print(f"Exception occurred in symbolic_equal (2): {e}")
            pass

    if score_type == "hard":
        return True if True in max_score else False
    return float(max(max_score)) if len(max_score) > 0 else 0.0


@lru_cache(maxsize=10000)
def numeral_equal(pre, ref, score_type="hard"):
    pre_t = str(parse_digits(pre)) if is_digit(pre) else strip_string(pre)
    ref_t = str(parse_digits(ref)) if is_digit(ref) else strip_string(ref)
    # print(pre_t, ref_t)
    eval_res = math_equal(pre_t, ref_t, score_type=score_type, timeout=True)
    if not eval_res:
        pre_t = extract_numbers(str(pre))
        if len(pre_t):
            ref_t = extract_numbers(str(ref))
            if len(ref_t):
                # print(pre_t, ref_t)
                eval_res = math_equal(pre_t, ref_t, score_type=score_type, timeout=True)
    return eval_res


if __name__ == "__main__":
    start_time = time.time()

    # print(numeral_equal("0.00", "0.01"))  # False
    # print(numeral_equal('({2014}^{2012} + {2013}^{2013})^{2011} + {2012}^{{2014}^{2012} + {2013}^{2013} - 1} \\square 2011', "2"))  # False
    # print(numeral_equal("25", "twenty-five"))  # True
    # print(numeral_equal("262.6401", "262.64"))  # True
    # print(numeral_equal("答案是25", "二十五"))  # True
    # print(numeral_equal(1/4, "零点二五"))  # True
    # print(numeral_equal("老二", "老2"))  # True
    # print(numeral_equal("零点二五", "1/4"))  # True
    # print(numeral_equal("22.4 m", "22.4"))  # True
    # print(numeral_equal("0.245", "24.5%"))  # True
    # print(numeral_equal("24.5", "24.5%"))  # True
    # print("F", numeral_equal("24.55", "24.5%"))  # False
    # print(numeral_equal("1/3", "33.33%"))  # True
    # print(numeral_equal("答案是零点五加仑", "4/8加仑"))  # True
    # print(numeral_equal("零点二五", "\\frac{1}{4}"))  # True
    # print(numeral_equal("\\dfrac{5}{4}", "1\\frac{1}{4}"))  # True  支持假分数 和 带分数（mixed number）
    # print(numeral_equal("\\dfrac{1}{5}", "1:5"))  # True  支持假分数 和 带分数（mixed number）
    # print(numeral_equal("18;10", "18,10"))  # True
    # print(numeral_equal("18+10", "28"))  # True
    # print(numeral_equal("a=2,\\b=2", "2,2"))  # True

    execution_time = time.time() - start_time
    print(f"代码执行时间: {execution_time:.6f} 秒")