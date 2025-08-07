"""
Core functionality for bias detection and analysis.

This module contains the main classifiers, scrapers, and fact-checking components.
"""

from .classifier_simple import SimpleBiasClassifier, predict_bias
from .scrapper import extract_text_from_url
from .fact_checker import query_google_fact_check

__all__ = [
    'SimpleBiasClassifier',
    'predict_bias', 
    'extract_text_from_url',
    'query_google_fact_check'
] 