"""
Analytics module for deep statistical, semantic, and social network processing.
"""

from src.analytics.engine import DeepChatAnalyzer
from src.analytics.metrics import (
    analyze_sentiment,
    compute_percentiles,
    compute_shannon_entropy,
    count_tokens,
)
from src.analytics.network import SocialNetworkAnalyzer
from src.analytics.report_generator import ReportGenerator

__all__ = [
    "DeepChatAnalyzer",
    "SocialNetworkAnalyzer",
    "ReportGenerator",
    "compute_percentiles",
    "compute_shannon_entropy",
    "analyze_sentiment",
    "count_tokens",
]
