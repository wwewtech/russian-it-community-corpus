"""
Analytics module for deep statistical, semantic, and social network processing.
"""

from ricc.analytics.engine import DeepChatAnalyzer
from ricc.analytics.metrics import (
    analyze_sentiment,
    compute_percentiles,
    compute_shannon_entropy,
    count_tokens,
)
from ricc.analytics.network import SocialNetworkAnalyzer
from ricc.analytics.report_generator import ReportGenerator

__all__ = [
    "DeepChatAnalyzer",
    "SocialNetworkAnalyzer",
    "ReportGenerator",
    "compute_percentiles",
    "compute_shannon_entropy",
    "analyze_sentiment",
    "count_tokens",
]
