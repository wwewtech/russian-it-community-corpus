"""
Unit tests for Domain Classification, Keyword Tagging, and Taxonomy.
"""

import unittest

from src.taxonomy.classifier import DomainClassifier
from src.taxonomy.tagger import TechnicalTagger


class TestTaxonomyClassification(unittest.TestCase):
    def setUp(self):
        self.classifier = DomainClassifier()
        self.tagger = TechnicalTagger()

    def test_database_classification(self):
        text = "PostgreSQL deadlock in transaction isolation level repeatable read with index scan"
        best_domain, confidence, counts = self.classifier.classify_text(text)
        self.assertIn(best_domain, ("backend_databases", "general_tech_chat"))

    def test_devops_classification(self):
        text = "Kubernetes ingress controller helm chart deployment with prometheus metrics"
        best_domain, confidence, counts = self.classifier.classify_text(text)
        self.assertIn(best_domain, ("devops_infra", "sysadmin_security", "general_tech_chat"))

    def test_ai_ml_classification(self):
        text = "Fine-tuning Qwen 2.5 with LoRA adapters in PyTorch with gradient checkpointing"
        best_domain, confidence, counts = self.classifier.classify_text(text)
        self.assertIn(best_domain, ("ai_ml_nlp", "general_tech_chat"))

    def test_tag_extraction(self):
        text = "Docker container running Redis and Nginx reverse proxy with TLS certificate"
        tags = self.tagger.extract_tags(text)
        self.assertIsInstance(tags, list)
        tag_set = {t.lower() for t in tags}
        self.assertTrue(any(k in tag_set for k in ["docker", "redis", "nginx", "tls", "security", "infra", "container", "proxy"]))


if __name__ == "__main__":
    unittest.main()
