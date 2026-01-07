# Evaluation Scripts Package

# Version Info
__version__ = "1.0.1"

# Package Description
__description__ = "A comprehensive evaluation framework for assessing Large Language Model (LLM) performance on diverse question types and answer formats."

# Import the main module
from .main import compute_score, evaluation


# Export the main functions
__all__ = ['compute_score', 'evaluation']