"""
Taxonomy, classification, and multi-label tagging module.
"""

from src.taxonomy.classifier import DomainClassifier
from src.taxonomy.tagger import TechnicalTagger

__all__ = ["DomainClassifier", "TechnicalTagger"]
