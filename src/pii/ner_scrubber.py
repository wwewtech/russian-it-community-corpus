"""
NLP / NER PII Scrubber using Natasha for Russian Named Entity Recognition (PER, LOC).
"""

import logging

logger = logging.getLogger(__name__)

# High-priority whitelist of technical terms, libraries, tools, and tech companies
TECH_WHITELIST: set[str] = {
    # Programming Languages & Runtimes
    "python",
    "питон",
    "пайтон",
    "golang",
    "го",
    "голанг",
    "rust",
    "раст",
    "java",
    "джава",
    "жаба",
    "kotlin",
    "котлин",
    "c++",
    "плюсы",
    "c#",
    "шарп",
    "javascript",
    "typescript",
    "js",
    "ts",
    "php",
    "пыха",
    "ruby",
    "руби",
    "swift",
    "свифт",
    "scala",
    "скала",
    "lua",
    "erlang",
    "elixir",
    "r",
    # Frameworks & Libraries
    "fastapi",
    "django",
    "джанго",
    "flask",
    "фласк",
    "pydantic",
    "sqlalchemy",
    "alembic",
    "celery",
    "asyncio",
    "aiohttp",
    "tornado",
    "react",
    "реакт",
    "next",
    "nextjs",
    "vue",
    "vuejs",
    "вье",
    "nuxt",
    "angular",
    "ангуляр",
    "svelte",
    "tailwind",
    "тайлвинд",
    "shadcn",
    "шадсн",
    "bootstrap",
    "pytorch",
    "пайторч",
    "торч",
    "tensorflow",
    "тензорфлоу",
    "keras",
    "transformers",
    "huggingface",
    "хаггингфейс",
    "unsloth",
    "langchain",
    "llamaindex",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "sklearn",
    # AI Models & Vendors
    "deepseek",
    "дипсик",
    "openai",
    "опенаи",
    "anthropic",
    "антропик",
    "chatgpt",
    "чатгпт",
    "gpt",
    "gpt-4",
    "gpt-4o",
    "gpt-3.5",
    "claude",
    "клод",
    "llama",
    "лама",
    "llama-3",
    "llama3",
    "mistral",
    "мистраль",
    "qwen",
    "квен",
    "gemini",
    "гемини",
    "джемини",
    "grok",
    "грок",
    "midjourney",
    "миджорни",
    "stable diffusion",
    "diffusion",
    "comfyui",
    "ollama",
    "vllm",
    "sglang",
    "cursor",
    "курсор",
    "copilot",
    "копилот",
    "windsurf",
    "cline",
    # Databases & Storage
    "postgres",
    "postgresql",
    "постгрес",
    "постгре",
    "mysql",
    "мускул",
    "sqlite",
    "clickhouse",
    "кликхаус",
    "redis",
    "редис",
    "mongodb",
    "монго",
    "elasticsearch",
    "elastic",
    "эластик",
    "cassandra",
    "qdrant",
    "кдрант",
    "chroma",
    "chromadb",
    "pinecone",
    "weaviate",
    "milvus",
    "neo4j",
    # DevOps, Infra & Cloud
    "docker",
    "докер",
    "kubernetes",
    "k8s",
    "кубер",
    "кубернетес",
    "helm",
    "ansible",
    "ансибл",
    "terraform",
    "терраформ",
    "nginx",
    "нжинкс",
    "нгинкс",
    "caddy",
    "traefik",
    "apache",
    "апач",
    "kafka",
    "кафка",
    "rabbitmq",
    "кролик",
    "linux",
    "линукс",
    "ubuntu",
    "убунту",
    "debian",
    "дебиан",
    "centos",
    "alpine",
    "arch",
    "windows",
    "виндовс",
    "винда",
    "macos",
    "макос",
    "мак",
    "hetzner",
    "хетзнер",
    "хецнер",
    "aws",
    "amazon",
    "амазон",
    "gcp",
    "google",
    "гугл",
    "azure",
    "ажур",
    "yandex",
    "яндекс",
    "selectel",
    "селектел",
    "timeweb",
    "таймвеб",
    "digitalocean",
    "ovh",
    "vps",
    "vds",
    # Tools & Platforms
    "telegram",
    "телеграм",
    "телега",
    "тг",
    "tg",
    "github",
    "гитхаб",
    "гит",
    "git",
    "gitlab",
    "гитлаб",
    "jira",
    "джира",
    "confluence",
    "конфлюенс",
    "notion",
    "ноушен",
    "slack",
    "слак",
    "discord",
    "дискорд",
    "zoom",
    "зум",
    "habr",
    "хабр",
    "vc",
    "vc.ru",
    "pikabu",
    "пикабу",
    "reddit",
    "реддит",
    "apple",
    "эппл",
    "microsoft",
    "майкрософт",
    "meta",
    "мета",
    "nvidia",
    "нвидиа",
    # Tech Terms
    "sft",
    "dpo",
    "rlhf",
    "rag",
    "lora",
    "qlora",
    "bpe",
    "cuda",
    "vram",
    "gpu",
    "cpu",
    "tpu",
    "api",
    "rest",
    "grpc",
    "graphql",
    "crud",
    "ci",
    "cd",
    "ssl",
    "tls",
    "vpn",
    "wireguard",
    "vless",
    "shadowsocks",
    "proxy",
    "jwt",
    "oauth",
    "auth",
    "cors",
    "dns",
    "http",
    "https",
    "ssh",
    "ip",
}


class NERPIIScrubber:
    """
    Scrubber powered by Natasha NER for identifying Person and Location entities.
    """

    def __init__(self):
        self.enabled = False
        try:
            from natasha import (
                Doc,
                MorphVocab,
                NewsEmbedding,
                NewsMorphTagger,
                NewsNERTagger,
                Segmenter,
            )

            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            self.emb = NewsEmbedding()
            self.ner_tagger = NewsNERTagger(self.emb)
            self.morph_tagger = NewsMorphTagger(self.emb)
            self.Doc = Doc
            self.enabled = True
            logger.info("Natasha NER engine initialized successfully.")
        except Exception as e:
            logger.warning(f"Natasha NER could not be initialized: {e}. Falling back to regex.")

    def scrub(self, text: str) -> tuple[str, dict[str, int]]:
        """
        Identify PER and LOC entities in text and mask them.
        """
        stats = {"ner_per": 0, "ner_loc": 0}
        if not text or not self.enabled:
            return text, stats

        # Fast pre-filtering: NER is only needed if text contains Cyrillic AND uppercase letters (proper names)
        has_cyrillic = any("\u0400" <= c <= "\u04ff" for c in text)
        if not has_cyrillic or not any(c.isupper() for c in text):
            return text, stats

        try:
            doc = self.Doc(text)
            doc.segment(self.segmenter)
            doc.tag_ner(self.ner_tagger)

            spans_to_replace = []
            for span in doc.spans:
                # Normalize text of span
                span_text = span.text.strip().lower()

                # Check whitelist
                if span_text in TECH_WHITELIST:
                    continue
                words = span_text.split()
                if any(w in TECH_WHITELIST for w in words):
                    continue

                if span.type == "PER":
                    spans_to_replace.append((span.start, span.stop, "[PERSON_REDACTED]", "ner_per"))
                elif span.type == "LOC" and span_text not in {
                    "рф",
                    "россия",
                    "сша",
                    "европа",
                    "москва",
                    "спб",
                    "питер",
                }:
                    spans_to_replace.append((span.start, span.stop, "[LOCATION_REDACTED]", "ner_loc"))

            if not spans_to_replace:
                return text, stats

            # Sort spans in reverse order so replacements don't offset index positions
            spans_to_replace.sort(key=lambda x: x[0], reverse=True)

            clean_text = text
            for start, stop, replacement, stat_key in spans_to_replace:
                clean_text = clean_text[:start] + replacement + clean_text[stop:]
                stats[stat_key] += 1

            return clean_text, stats

        except Exception as e:
            logger.debug(f"NER scrubbing error on text snippet: {e}")
            return text, stats
