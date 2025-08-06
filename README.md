# LLM Evaluation Framework

This directory contains a comprehensive evaluation framework for assessing Large Language Model (LLM) performance on diverse question types and answer formats. The system supports multiple data types and comparison methods to ensure accurate evaluation across various problem domains.

## Overview

The evaluation framework is designed to handle complex answer formats including:
- Multiple choice questions with single or multiple correct answers
- Numerical computations and mathematical expressions
- Text/nominal answers with normalization
- Ordered and unordered arrays
- Multi-dimensional data structures
- Mixed-type responses
- LaTeX mathematical notation

## Key Features

- **🔍 Flexible Answer Extraction**: Extracts answers from LaTeX `\boxed{}` notation and various text formats
- **🎯 Multi-Type Evaluation**: Supports different evaluation strategies based on answer type with automatic detection
- **🛠️ Robust Parsing**: Handles JSON, LaTeX tables, malformed data, and mixed text formats
- **📊 Score Computation**: Provides both binary evaluation and partial scoring for multi-part questions
- **🌍 Language Support**: Handles both English and Chinese text, including number word conversion
- **⚡ Performance Optimized**: Includes timeout handling and concurrent evaluation for large-scale assessments

## Installation

Install the required dependencies from the main project directory:

```bash
cd ..  # Go to main project directory
pip install -r requirements.txt
```

## Quick Start

### Basic Evaluation

```python
from main import eval_router, compute_score, evaluation

# Simple evaluation
prediction = "The answer is A"
reference = "A"
result = eval_router(prediction, reference, eval_type="option")
print(result)  # True/False

# Numerical evaluation
result = eval_router("25", "twenty-five", eval_type="numeral")
print(result)  # True

# Array evaluation
pred_array = "['apple', 'banana', 'cherry']"
ref_array = "['apple', 'banana', 'cherry']"
result = eval_router(pred_array, ref_array, eval_type="ordered array")
print(result)  # True
```

### Score Computation for RL Training

```python
# Multi-part question scoring
solution = "\\boxed{A}\\boxed{125}"
ground_truth = "A====125"
eval_types = "option,numeral"
score = compute_score(solution, ground_truth, eval_types)
print(f"Score: {score}")  # Returns float between 0.0 and 1.0
```

### Advanced Examples

```python
# Complex array evaluation
solution_str = "\\boxed{[['a','b'], ['c', 'd']]}"
ground_truth = "[['b','a'], ['d', 'c']]"
score = compute_score(solution_str, ground_truth, "oua_nominal")
print(f"Unordered outer, ordered inner array score: {score}")

# Multiple choice with multiple answers
result = eval_router("A, B, C", "ABC", eval_type="multi_answers_MCQ")
print(result)  # True if all correct options are present
```

## Evaluation Types

### Basic Types
- `"option"`: Multiple choice answers (A, B, C, etc.)
- `"nominal"`: Text-based answers with normalization
- `"numeral"`: Numerical computations and expressions

### Array Types
- `"ordered array"`: Sequence-sensitive lists
- `"unordered array"`: Order-independent sets  
- `"subset"`: Subset relationship validation

### Advanced Types
- `"multi_answers_MCQ"`: Multiple correct options in multiple choice
- `"ooa_nominal"`: Ordered outer, ordered inner arrays (nominal data)
- `"oua_numeral"`: Ordered outer, unordered inner arrays (numerical data)
- `"uoa_nominal"`: Unordered outer, ordered inner arrays
- And other complex multi-dimensional combinations

## Project Structure

```
evaluation_scripts/
├── main.py                    # Main evaluation logic and routing
├── extract_answer.py          # Answer extraction from model outputs  
├── eval_multiple_choice.py    # Multiple choice question evaluation
├── eval_nominal.py           # Text/name-based answer evaluation
├── eval_numeral.py           # Numerical answer evaluation with SymPy
├── eval_array.py             # Array and sequence evaluation
├── eval_utils.py             # Utility functions and text processing
└── README.md                 # This documentation
```

## Core Functions

### `eval_router(prediction, reference, eval_type=None)`
Main evaluation function that routes to appropriate evaluators based on type.

**Parameters:**
- `prediction`: Model's predicted answer
- `reference`: Ground truth answer  
- `eval_type`: Evaluation strategy (optional, auto-detected if None)

**Returns:** Boolean indicating correctness

### `compute_score(solution_str, ground_truth, eval_type_text)`
Scoring function for reinforcement learning training with partial credit support.

**Parameters:**
- `solution_str`: Complete model output with potential multiple answers
- `ground_truth`: Ground truth with potential multiple parts
- `eval_type_text`: Comma-separated evaluation types

**Returns:** Float score between 0.0 and 1.0 (suitable as RL reward)

### `evaluation(prediction, ground_truth, eval_type_text)`  
Strict binary evaluation for model inference assessment.

**Returns:** True if all parts are correct, False otherwise

## Text Processing Features

- **📝 Number Word Conversion**: Converts English ("twenty-five") and Chinese ("二十五") number words to digits
- **🔧 Punctuation Normalization**: Removes and normalizes punctuation across languages
- **📊 LaTeX Table Parsing**: Extracts structured data from LaTeX table formats
- **🛠️ JSON Repair**: Handles malformed JSON with automatic correction
- **🧮 Mathematical Expression Handling**: Processes complex mathematical notation

## Supported Input Formats

- Plain text answers
- LaTeX mathematical expressions with `\boxed{}` notation
- JSON arrays and objects  
- LaTeX tables (`\begin{array}`, `\begin{bmatrix}`)
- Mixed format responses with multiple `\boxed{}` sections
- Comma/semicolon separated lists
- Multi-line responses

## Error Handling

The system includes robust error handling for:
- ✅ Malformed JSON structures
- ✅ Invalid LaTeX expressions  
- ✅ Missing or incomplete answers
- ✅ Type conversion errors
- ✅ Parsing failures
- ✅ Timeout protection for complex symbolic computation

All evaluation functions return appropriate defaults (False/0.0) when errors occur, ensuring the evaluation pipeline continues running.

## Performance Considerations

- **⏱️ Timeout Protection**: Symbolic computations are protected by timeouts to prevent hanging
- **🔄 Concurrent Processing**: Uses ThreadPoolExecutor for performance-critical operations
- **🎯 Efficient Parsing**: Optimized parsing strategies for different data formats
- **📊 Batch Processing**: Designed to handle large-scale evaluation efficiently

## Contributing

When adding new evaluation types:

1. Implement the evaluation logic in the appropriate `eval_*.py` file
2. Add the type routing in `main.py`'s `eval_router` function  
3. Update the documentation with the new type
4. Add comprehensive test cases to verify functionality
5. Ensure proper error handling and edge case coverage

## License

This evaluation framework is part of the PuzzleClone project and is licensed under the Apache 2.0 License.

# Simple evaluation
prediction = "The answer is A"
reference = "A"
result = eval_router(prediction, reference, eval_type="option")
print(result)  # True/False

# Score computation for complex questions
solution = "\\boxed{A}\\boxed{125}"
ground_truth = "A====125"
eval_types = "option,numeral"
score = compute_score(solution, ground_truth, eval_types)
print(f"Score: {score}")  # 0.0 to 1.0
```

### Evaluation Types

The system supports the following evaluation types:

#### Basic Types
- `"option"`: Multiple choice answers (A, B, C, etc.)
- `"nominal"`: Text-based answers with normalization
- `"numeral"`: Numerical computations and expressions

#### Array Types
- `"ordered array"`: Sequence-sensitive lists
- `"unordered array"`: Order-independent sets
- `"subset"`: Subset relationship validation

#### Advanced Types
- `"multi_answers_MCQ"`: Multiple correct options
- `"ooa_nominal"`: Ordered outer, ordered inner arrays (nominal)
- `"oua_numeral"`: Ordered outer, unordered inner arrays (numerical)
- `"uoa_nominal"`: Unordered outer, ordered inner arrays

### Answer Extraction

The system automatically extracts answers from various formats:

```python
from extract_answer import extract_answer

# Extract from LaTeX boxed notation
text = "The solution is \\boxed{42} and \\boxed{A}"
answers = extract_answer(text)
print(answers)  # ['42', 'A']

# Extract last answer only
last_answer = extract_answer(text, last=True)
print(last_answer)  # 'A'
```

### Advanced Examples

#### Multi-Part Questions
```python
# Question with multiple parts
solution = "\\boxed{[['a','b'], ['c', 'd']]}\\boxed{A}"
ground_truth = "[['b','a'], ['d', 'c']]====A"
eval_types = "oua_nominal,option"

score = compute_score(solution, ground_truth, eval_types)
# Evaluates each part separately and returns average score
```

## Key Functions

### `eval_router(prediction, reference, eval_type=None)`
Main evaluation function that routes to appropriate evaluators based on type.

**Parameters:**
- `prediction`: Model's predicted answer
- `reference`: Ground truth answer
- `eval_type`: Evaluation strategy (optional, auto-detected if None)

**Returns:** Boolean indicating correctness

### `compute_score(solution_str, ground_truth, eval_type_text)`
Scoring function specifically designed for reinforcement learning training. Returns a continuous score that can be used as a reward signal during RL training phases.

**Parameters:**
- `solution_str`: Complete model output with potential multiple answers
- `ground_truth`: Ground truth with potential multiple parts
- `eval_type_text`: Comma-separated evaluation types

**Returns:** Float score between 0.0 and 1.0 (used as RL reward)

### `evaluation(prediction, ground_truth, eval_type_text)`
Evaluation function for model inference assessment. Provides binary correctness evaluation after model inference is complete.

**Returns:** True if all parts are correct, False otherwise (strict evaluation)

## Text Processing Features

- **Number Word Conversion**: Converts English and Chinese number words to digits
- **Punctuation Normalization**: Removes and normalizes punctuation
- **LaTeX Table Parsing**: Extracts data from LaTeX table structures
- **JSON Repair**: Handles malformed JSON with automatic correction

## Dependencies

- `regex`: Advanced pattern matching
- `json_repair`: JSON parsing and repair
- `word2number`: English number word conversion
- `cn2an`: Chinese number conversion
- `sympy`: Mathematical expression evaluation
- `latex2sympy2`: LaTeX to SymPy conversion

## Error Handling

The system includes robust error handling for:
- Malformed JSON structures
- Invalid LaTeX expressions
- Missing or incomplete answers
- Type conversion errors
- Parsing failures

All evaluation functions return appropriate defaults (False/0.0) when errors occur, ensuring the evaluation pipeline continues running.

## Contributing

When adding new evaluation types:

1. Implement the evaluation logic in the appropriate `eval_*.py` file
2. Add the type routing in `main.py`'s `eval_router` function
3. Update the documentation with the new type
4. Add test cases to verify functionality

## License

This project is part of the puzzleClone evaluation framework.
