"""
Data sources and bias lookup tables.

This module contains bias rating data from AllSides and Ad Fontes Media.
"""

from .allsides_bias_lookup import allsides_bias
from .adfontes_bias_lookup import adfontes_bias

__all__ = [
    'allsides_bias',
    'adfontes_bias'
] 