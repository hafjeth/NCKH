"""
Evaluation Metrics Module
"""

# Import từ retrieval_metrics.py (dùng relative import)
from .retrieval_metrics import MetricsCalculator, CitationCounter

# Backward compatibility - import từ file cũ nếu cần
try:
    from ..metrics_old import *
except ImportError:
    pass

__all__ = ['MetricsCalculator', 'CitationCounter']