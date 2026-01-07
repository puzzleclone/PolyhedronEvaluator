"""
Array Evaluation Module

This module provides functionality for evaluating array-type answers
with support for ordered, unordered, and subset comparisons.
"""

from typing import List , Any

from functools import lru_cache

try:
    from eval_numeral import is_digit, numeral_equal
    from eval_nominal import nominal_equal
except ImportError:
    from .eval_numeral import is_digit, numeral_equal
    from .eval_nominal import nominal_equal


@lru_cache(maxsize=10000)
def is_sub_str(r, p):
    rt = r.strip().lower()
    pt = p.strip().lower()
    if len(rt) * len(pt) == 0:
        return len(rt) == len(pt)
    return rt in pt


def array_element_equal(pi, ri):
    if isinstance(pi, list) and isinstance(ri, list):
        return order_array_equal(pi, ri, score_type='hard')

    if isinstance(pi, list) or isinstance(ri, list):
        return False

    pi_str = str(pi)
    ri_str = str(ri)

    if nominal_equal(pi_str, ri_str):
        return True
    if is_digit(ri_str):
        if numeral_equal(pi_str, ri_str):
            return True
    else:
        if is_sub_str(ri_str, pi_str):
            return True
    return False


def unorder_array_equal(pre: list[str], ref: list[str], score_type: str = 'hard'):
    if type(pre) != list or type(ref) != list:
        return False

    if score_type == 'hard':
        if len(pre) != len(ref):
            return False

        ref = ref.copy()  # 防止.remove()改变lru_cache的结果
        for pi in pre:
            find_flag = False
            for ri in ref:
                if array_element_equal(pi, ri):
                    find_flag = True
                    ref.remove(ri)
                    break
            if not find_flag:
                return False
        return True

    else:
        n_correct = 0
        n_ref = len(ref)
        ref = ref.copy()  # 防止.remove()改变lru_cache的结果
        for pi in pre:
            for ri in ref:
                if array_element_equal(pi, ri):
                    n_correct += 1
                    ref.remove(ri)
                    break
        return n_correct / max(len(pre), n_ref, 1)


def order_array_equal(pre: list[str], ref: list[str], score_type: str = 'hard'):
    if type(pre) != list or type(ref) != list:
        return False

    # len(zip(a, b)) = min(len(a), len(b))
    if score_type == 'hard':
        return len(pre) == len(ref) and all(array_element_equal(pi, ri) for pi, ri in zip(pre, ref))
    else:
        n_correct = sum(array_element_equal(pi, ri) for pi, ri in zip(pre, ref))
        return n_correct / max(len(pre), len(ref), 1)


def _to_float_safe(x: Any) -> float | None:
    """安全地把元素转成 float，list 等非法类型返回 None。"""
    # 如果是 list，尝试展开单元素列表，否则视为非法
    if isinstance(x, list):
        if len(x) >= 1:
            x = x[0]
        else:
            return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def order_number_array_equal(pre: list[Any], ref: list[Any], score_type: str = 'hard'):
    if not isinstance(pre, list) or not isinstance(ref, list):
        return False

    if score_type == 'hard':
        if len(pre) != len(ref):
            return False
        for pi, ri in zip(pre, ref):
            f_pi = _to_float_safe(pi)
            f_ri = _to_float_safe(ri)
            if f_pi is None or f_ri is None or f_ri != f_pi:
                return False
        return True
    else:
        n_correct = 0
        for pi, ri in zip(pre, ref):
            f_pi = _to_float_safe(pi)
            f_ri = _to_float_safe(ri)
            if f_pi is not None and f_ri is not None and f_ri == f_pi:
                n_correct += 1
        return n_correct / max(len(pre), len(ref), 1)


def unorder_number_array_equal(pre: list[Any], ref: list[Any], score_type: str = 'hard'):
    if not isinstance(pre, list) or not isinstance(ref, list):
        return False

    if score_type == 'hard':
        if len(pre) < len(ref):
            return False

        ref_copy = ref.copy()  # 防止.remove()改变lru_cache的结果
        for pi in pre:
            f_pi = _to_float_safe(pi)
            if f_pi is None: # 
                return False  # 这一位根本不是合法数字，直接视为不匹配

            find_flag = False
            for ri in ref_copy:
                f_ri = _to_float_safe(ri)
                if f_ri is not None and f_ri == f_pi:
                    find_flag = True
                    ref_copy.remove(ri)
                    break
            if not find_flag:
                return False

        return True

    else:  # soft
        n_correct = 0
        n_ref = len(ref)
        ref_copy = ref.copy()
        for pi in pre:
            f_pi = _to_float_safe(pi)
            if f_pi is None:
                continue  # 非数字，跳过这一个

            for ri in ref_copy:
                f_ri = _to_float_safe(ri)
                if f_ri is not None and f_ri == f_pi:
                    n_correct += 1
                    ref_copy.remove(ri)
                    break

        return n_correct / max(len(pre), n_ref, 1)



def subset_array_equal(pre: List[str], ref: List[str]) -> bool:
    """
    Check if predicted array is a subset of reference array.
    
    Args:
        pre: Predicted array
        ref: Reference array
        
    Returns:
        True if all elements in pre are contained in ref
    """
    for pi in pre:
        if not any([ri.lower() in pi.lower() for ri in ref]):
            return False

    return True


if __name__ == "__main__":
    test_cases = [
        {
            "pre": ["老15", "老三", "老5"],
            "ref": ["3", "15", "6"]
        },  # False False 0.0 0.6666666666666666
        {
            "pre": ["二", "(B)", "Seven"],
            "ref": ["B", "2", "7"]
        },   # False True 0.3333333333333333 1.0
        {
             "pre": [50, '刘鑫', 21],
             "ref": [50, '刘鑫', 22],
        }
    ]

    for case in test_cases:
        print(order_array_equal(case["pre"], case["ref"]), 
              unorder_array_equal(case["pre"], case["ref"]),
              order_array_equal(case["pre"], case["ref"], score_type='soft'), 
              unorder_array_equal(case["pre"], case["ref"], score_type='soft'))