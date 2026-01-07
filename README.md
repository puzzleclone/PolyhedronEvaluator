# LLM Evaluation Framework

This directory contains a comprehensive evaluation framework for assessing Large Language Model (LLM) performance on diverse question types and answer formats. The system supports multiple data types and comparison methods to ensure accurate evaluation across various problem domains.

## Overview

The evaluation framework is designed to handle complex answer formats including:

- Multiple choice questions with single or multiple correct answers
- Numerical computations and mathematical expressions
- Text/nominal answers with normalization (regular and strict)
- Ordered and unordered arrays
- Multi-dimensional nested data structures
- Mixed-type responses
- LaTeX mathematical notation

## Key Features

- **🔍 Flexible Answer Extraction**: Extracts answers from LaTeX `\boxed{}` notation and various text formats
- **🎯 Multi-Type Evaluation**: Supports different evaluation strategies based on answer type with automatic detection
- **🛠️ Robust Parsing**: Handles JSON, LaTeX tables, malformed data, and mixed text formats
- **📊 Score Computation**: Provides both hard (binary) and soft (partial credit) scoring for multi-part questions
- **🌍 Language Support**: Handles both English and Chinese text, including number word conversion
- **⚡ Performance Optimized**: Uses LRU caching and efficient parsing strategies

## Installation

```bash
git clone https://github.com/puzzleclone/PolyhedronEvaluator.git
cd PolyhedronEvaluator
pip install -r requirement.txt
```

## Quick Start

### Basic Evaluation

```python
from main import compute_score, evaluation

# Simple evaluation for binary correctness
prediction = "\\boxed{A}"
reference = "A"
result = compute_score(prediction, reference, "option")
print(result)  # 1.0

# Numerical evaluation
result = compute_score("\\boxed{25}", "twenty-five", "numeral")
print(result)  # 1.0

# Array evaluation
pred_array = "\\boxed{['apple', 'banana', 'cherry']}"
ref_array = "['apple', 'banana', 'cherry']"
result = compute_score(pred_array, ref_array, "ordered array")
print(result)  # 1.0
```

### Scoring modes

We support two scoring modes: **hard scoring** (i.e., without partial scores; default) and **soft scoring** (i.e., with partial scores). 

```python
# Multi-part question scoring with soft scoring
solution = "\\boxed{A}\\boxed{100}"
ground_truth = "A====125"
eval_types = "option,numeral"
score = compute_score(solution, ground_truth, eval_types, score_type="soft")
print(f"Score: {score}")  # 0.5 (1 answer out of 2 is correct)

# Hard scoring (binary)
score = compute_score(solution, ground_truth, eval_types, score_type="hard")
print(f"Score: {score}")  # 0.0 (not all answers correct)
```

### Advanced Examples

```python
# Complex nested array evaluation
solution_str = "\\boxed{[['a', 'b'], ['c', 'd']]}"
ground_truth = "[['b', 'a'], ['d', 'c']]"
score = compute_score(solution_str, ground_truth, "oua_nominal", score_type="soft")
print(f"Ordered outer, unordered inner array score: {score}") # 1.0

# Multiple choice with multiple answers
result = evaluation("\\boxed{A, B, C}", "ABC", "multi_options")
print(result)  # True

# Multi-part question with different evaluation types
solution = "\\boxed{[['6', '7'], ['8', '9']]} \\boxed{A}"
ground_truth = "[['6', '7'], ['8', '9']]====A"
eval_type = "ooa_numeral,option"
score = compute_score(solution, ground_truth, eval_type, "soft")
print(f"Score: {score}")  # 1.0

# Multiple attempts with soft scoring (rewards later correct answers)
pre = "\\boxed{1} or \\boxed{2} or \\boxed{3}"
ref = "2"
score = compute_score(pre, ref, "numeral", "soft")
print(f"Score: {score}")  # ~0.33 (found correct answer in 3rd position)

# Percentage and fraction equivalence
result = evaluation("\\boxed{1/3}", "33.33%", "numeral")
print(result)  # True

# Chinese text evaluation
result = evaluation("\\boxed{二十五}", "25", "numeral")
print(result)  # True
```

## Scoring Mechanisms

### Hard vs Soft Scoring

The framework supports two scoring modes controlled by the `score_type` parameter:

#### Hard Scoring (`score_type="hard"`)
- Binary evaluation: returns `True` (1.0) or `False` (0.0)
- All parts of a multi-part question must be correct
- Suitable for final evaluation and testing
- No partial credit

#### Soft Scoring (`score_type="soft"`)
- Partial credit scoring: returns float between 0.0 and 1.0
- **Recommended for RL training** as it provides gradient signals
- Average score across multiple parts
- Special features:
  - **Position-aware rewards**: Later correct answers receive adjusted scores
  - **Multiple attempt handling**: Considers different answer positions
  - **Deduplication**: Removes duplicate answer attempts
  - **Concatenation fallback**: For single-question array evaluations, tries concatenating multiple predictions

### Multi-Part Question Handling

For questions with multiple parts:
1. Ground truth uses separators: `====`, `;`, `；`, or `\n`
2. Evaluation types are comma-separated: `"option,numeral,nominal"`
3. Scores are computed per part and averaged
4. Automatic answer extraction from multiple `\boxed{}` notations

### Multiple Attempts Strategy (Soft Scoring)

When a model produces multiple attempts (e.g., `\boxed{1} or \boxed{2} or \boxed{3}`):
- Each possible answer window is evaluated
- Scores are position-weighted: later correct answers get higher rewards
- Formula: `score * (position + 1) / total_attempts`
- Final score considers diversity: more attempts reduce the overall score

## Evaluation Types

### Basic Types

- `"option"`: Multiple choice answers with single correct answer (A, B, C, etc.)
- `"nominal"`: Text-based answers with normalization (converts number words, ignores punctuation and case)
- `"strict_nominal"`: Strict text-based answers using longest common substring scoring
- `"numeral"`: Numerical computations and expressions (supports fractions, percentages, etc.)

### Array Types

- `"ordered array"` or `"oa_nominal"`: Sequence-sensitive lists (e.g., [1,2,3])
- `"unordered array"` or `"ua_nominal"`: Order-independent sets
- `"subset"`: Subset relationship validation

### Advanced Types

- `"multi_options"`: Multiple choice with multiple correct answers
- `"ooa_nominal"`: Ordered outer, ordered inner arrays (nominal data)
- `"ooa_numeral"`: Ordered outer, ordered inner arrays (numerical data)
- `"oua_nominal"`: Ordered outer, unordered inner arrays (nominal data)
- `"oua_numeral"`: Ordered outer, unordered inner arrays (numerical data)
- `"uoa_nominal"`: Unordered outer, ordered inner arrays (nominal data)
- `"uoa_numeral"`: Unordered outer, ordered inner arrays (numerical data)
- `"uua_nominal"`: Unordered outer, unordered inner arrays (nominal data)
- `"uua_numeral"`: Unordered outer, unordered inner arrays (numerical data)
- `"oooa_numeral"`: Three-level nested arrays (ordered → ordered → ordered)
- And other complex multi-dimensional combinations using the `x*a_y` pattern

### Nested Array Pattern: `x*a_y`

The framework supports arbitrary nesting levels using the pattern where:
- `x` = sequence of 'o' (ordered) or 'u' (unordered) for each nesting level
- `a_` = separator
- `y` = element type ('nominal' or 'numeral')

Examples:
- `"oa_numeral"`: Ordered array of numbers
- `"ooa_nominal"`: Ordered array of ordered arrays (nominal)
- `"uua_numeral"`: Unordered array of unordered arrays (numerical)

### Automatic Type Detection

When `eval_type=None` or not specified, the framework automatically detects the type based on format:

1. **Array format** (`[...]`): Treated as unordered array
2. **Option format** (A, B, (C), D., etc.): Treated as multiple choice option
3. **Multiple items** (separated by newline, space, comma): Treated as ordered array
4. **Default fallback**: Numerical evaluation

This automatic detection makes the framework easy to use without explicit type specification.

## Project Structure

```
PolyhedronEvaluator/
├── main.py                    # Main evaluation logic and routing
├── extract_answer.py          # Answer extraction from model outputs
├── eval_multiple_choice.py    # Multiple choice question evaluation
├── eval_nominal.py            # Text/name-based answer evaluation
├── eval_numeral.py            # Numerical answer evaluation with SymPy
├── eval_array.py              # Array and sequence evaluation
├── eval_utils.py              # Utility functions and text processing
├── test_cases.py              # Test cases for validation
├── test_cases.json            # JSON test data
├── requirement.txt            # Package dependencies
└── README.md                  # This documentation
```

## Core Functions

### `compute_score(solution_str, ground_truth, eval_type=None, score_type="hard", debug=False)`

Comprehensive scoring function for reinforcement learning training with partial credit support.

**Parameters:**

- `solution_str` (str): Complete model output with potential multiple answers
- `ground_truth` (str): Ground truth with potential multiple parts (use `====` or `;` or `；` or `\n` to separate multiple answers)
- `eval_type` (Optional[str]): Comma-separated evaluation types (e.g., "option,numeral" for multi-part questions). If None, auto-detects type.
- `score_type` (str): Scoring mode:
  - `"hard"`: Binary scoring (0.0 or 1.0)
  - `"soft"`: Partial credit scoring (recommended for RL training)
- `debug` (bool): If True, returns tuple of (score, debug_info) with detailed scoring breakdown

**Returns:** 
- Float score between 0.0 and 1.0 (suitable as RL reward)
- If debug=True: tuple of (score, debug_info_list)

**Features:**
- Handles multiple attempts (considers different answer positions with adjusted rewards)
- Automatically extracts answers from `\boxed{}` notation
- Supports splitting last prediction if partial answers are correct
- Concatenates multiple predictions for single-question array evaluation

### `evaluation(prediction, ground_truth, eval_type=None)`

Strict binary evaluation for model inference assessment.

**Parameters:**

- `prediction` (str): Model's prediction
- `ground_truth` (str): Ground truth answer
- `eval_type` (Optional[str]): Evaluation type specification

**Returns:** True if all parts are correct (score = 1.0), False otherwise

### `eval_router(pre, ref, eval_type=None, score_type='hard')`

Internal routing function that directs to appropriate evaluator based on type.

**Features:**
- Automatic type detection if eval_type is None
- LRU cached for performance (maxsize=10000)
- Handles nested array evaluation with `comput_inner_array_score()`

## Text Processing Features

- **📝 Number Word Conversion**: Converts English ("twenty-five") and Chinese ("二十五") number words to digits
- **🔧 Punctuation Normalization**: Removes and normalizes punctuation across languages
- **📊 LaTeX Table Parsing**: Extracts structured data from LaTeX table formats (`\begin{array}`, `\begin{bmatrix}`, etc.)
- **🛠️ JSON Repair**: Handles malformed JSON with automatic correction using `json_repair`
- **🧮 Mathematical Expression Handling**: Processes complex mathematical notation
- **🔍 Smart Array Parsing**: Intelligently parses arrays from various formats including special handling for nested structures
- **💾 LRU Caching**: Performance optimization with `@lru_cache` decorators (maxsize=10000)

## Supported Input Formats

- Plain text answers
- LaTeX mathematical expressions with `\boxed{}` notation
- JSON arrays and objects (with automatic repair for malformed JSON)
- LaTeX tables (`\begin{array}`, `\begin{bmatrix}`)
- Mixed format responses with multiple `\boxed{}` sections
- Comma/semicolon separated lists (`,`, `，`, `;`, `；`, `和`, `或`)
- Multi-line responses with `\n` separator
- Nested arrays with arbitrary depth

## Error Handling

The system includes robust error handling for:

- ✅ Malformed JSON structures (with `json_repair` fallback)
- ✅ Invalid LaTeX expressions
- ✅ Missing or incomplete answers
- ✅ Type conversion errors
- ✅ Parsing failures
- ✅ Exception logging with full traceback for debugging

All evaluation functions return appropriate defaults (False/0.0) when errors occur, ensuring the evaluation pipeline continues running. Errors are printed with full context including the problematic inputs.

## Performance Considerations

- **💾 LRU Caching**: Functions like `eval_router()`, `split_and_strip()`, and `split_pred()` use `@lru_cache(maxsize=10000)` for significant performance improvements
- **🎯 Efficient Parsing**: Optimized parsing strategies for different data formats
- **📊 Batch Processing**: Designed to handle large-scale evaluation efficiently
- **🔄 Smart Array Processing**: Efficient permutation handling with early termination when perfect score is found
- **⚡ Optimized Pattern Matching**: Uses regex patterns with strategic fallbacks

## Contributing

When adding new evaluation types:

1. Implement the evaluation logic in the appropriate `eval_*.py` file
2. Add the type routing in `main.py`'s `eval_router()` function
3. Update the documentation with the new type
4. Add comprehensive test cases to `test_cases.py` or create new test files
5. Ensure proper error handling and edge case coverage
6. Consider adding LRU caching if the function is called frequently
7. Test with both hard and soft scoring modes

## License

This evaluation framework is part of the [PuzzleClone](https://github.com/puzzleclone/PuzzleClone) project and is licensed under the Apache 2.0 License.

## Acknowledgments

Our evaluation code is partially modified from [LIMO](https://github.com/GAIR-NLP/LIMO). We thank their team for their valuable contributions to the community.
