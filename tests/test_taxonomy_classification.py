"""
Unit tests for Domain Classification, Keyword Tagging, and Taxonomy.
"""

import pytest
from src.taxonomy.classifier import DomainClassifier
from src.taxonomy.tagger import TechnicalTagger


class TestTaxonomyClassification:
    def setup_method(self):
        self.classifier = DomainClassifier()
        self.tagger = TechnicalTagger()

    def test_database_classification(self):
        text = "PostgreSQL deadlock in transaction isolation level repeatable read with index scan"
        best_domain, confidence, counts = self.classifier.classify_text(text)
        assert best_domain in ("backend_databases", "general_tech_chat")

    def test_devops_classification(self):
        text = "Kubernetes ingress controller helm chart deployment with prometheus metrics"
        best_domain, confidence, counts = self.classifier.classify_text(text)
        assert best_domain in ("devops_infra", "sysadmin_security", "general_tech_chat")

    def test_ai_ml_classification(self):
        text = "Fine-tuning Qwen 2.5 with LoRA adapters in PyTorch with gradient checkpointing"
        best_domain, confidence, counts = self.classifier.classify_text(text)
        assert best_domain in ("ai_ml_nlp", "general_tech_chat")

    def test_tag_extraction(self):
        text = "Docker container running Redis and Nginx reverse proxy with TLS certificate"
        tags = self.tagger.extract_tags(text)
        assert isinstance(tags, list)
        tag_set = {t.lower() for t in tags}
        assert any(k in tag_set for k in ["docker", "redis", "nginx", "tls", "security", "infra", "container", "proxy"])
