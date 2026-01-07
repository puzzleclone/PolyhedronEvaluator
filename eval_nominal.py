"""
Nominal Evaluation Module

This module provides functionality for evaluating text/name-based answers
with normalization and word number conversion support.
"""

import re
import string
from functools import lru_cache

try:
    from eval_utils import convert_word_number
except ImportError:
    from .eval_utils import convert_word_number


@lru_cache(maxsize=10000)
def remove_punctuation(text: str) -> str:
    """
    Remove Chinese and English punctuation from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with punctuation removed
    """
    if not isinstance(text, str):
        return str(text)
    # Define Chinese and English punctuation
    chinese_punctuation = "。，、；：？！''""（）【】《》……—～·"
    all_punctuation = string.punctuation + chinese_punctuation
    
    # Use regex to remove punctuation
    pattern = f"[{re.escape(all_punctuation)}]"
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text


@lru_cache(maxsize=10000)
def nominal_equal(pre: str, ref: str) -> bool:
    """
    Compare two nominal answers with normalization.
    
    Performs the following normalization steps:
    1. Remove punctuation if reference has non-empty content after removal
    2. Convert word numbers to digits
    3. Compare case-insensitively with spaces removed
    
    Args:
        pre: Predicted answer
        ref: Reference answer
        
    Returns:
        True if answers are equivalent after normalization
    """
    ref2 = remove_punctuation(ref)
    # if ref after removing punctuation is not empty, remove punctuation from both
    if len(ref2.strip()) > 0:
        ref = ref2
        pre = remove_punctuation(pre)

    pre = convert_word_number(pre)
    ref = convert_word_number(ref)

    # Compare with normalized spaces and case
    if ''.join(pre.lower().split()) == ''.join(ref.lower().split()):
        return True

    return False


@lru_cache(maxsize=10000)
def strict_nominal_equal(pre: str, ref: str, score_type = "hard") -> float:

    # Strict nominal comparison with longest common substring scoring.
    pre = ''.join(pre.lower().split())
    ref = ''.join(ref.lower().split())

    denom = max(len(pre), len(ref))
    if denom == 0:
        return False if score_type == "hard" else 0.0

    if score_type == "hard":
        return True if pre == ref else False
    
    # compute longest common substring length
    m, n = len(pre), len(ref)
    dp = [0] * (n + 1)
    max_len = 0
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            curr = dp[j]
            if pre[i - 1] == ref[j - 1]:
                dp[j] = prev + 1
                if dp[j] > max_len:
                    max_len = dp[j]
            else:
                dp[j] = 0
            prev = curr

    return max_len / denom


if __name__ == "__main__":
    print(nominal_equal("F. 23", "F"))          # False
    print(nominal_equal("a\n ", "(A)"))         # True
    print(nominal_equal("3", "3"))          # True
    print(nominal_equal("小明妈妈", "小明"))     # False
    print(nominal_equal("小明", "小明妈妈"))     # False
    print(strict_nominal_equal("{}}}(", "{}}(]", "soft"))     # 0.6