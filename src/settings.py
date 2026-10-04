"""
Unified configuration for RICC using pydantic-settings.

Single source of truth for all configuration — merges config.py + params.yaml.
Environment variables take precedence (e.g., RICC_MINHASH_NUM_PERM=256).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DomainTaxonomyItem(BaseSettings):
    title: str
    keywords: list[str]


class PIIAuditSettings(BaseSettings):
    sample_size: int = 50_000
    confidence: float = 0.99
    max_leak_tolerance: float = 1e-4


class DriftSettings(BaseSettings):
    length_buckets: int = 10
    vocab_top_k: int = 500
    psi_stable: float = 0.10
    psi_moderate: float = 0.25
    js_stable: float = 0.05
    js_moderate: float = 0.20
    vocab_jaccard_stable: float = 0.90
    vocab_jaccard_moderate: float = 0.75


class BenchmarkSettings(BaseSettings):
    humaneval_tasks: int = 40
    rummlu_questions: int = 50
    confidence: float = 0.95


class Settings(BaseSettings):
    """Main settings class — all config in one place."""

    model_config = SettingsConfigDict(
        env_prefix="RICC_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Base paths ---
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    output_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset_output")
    parquet_output_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset_output" / "parquet"
    )
    jsonl_output_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset_output" / "jsonl"
    )
    samples_output_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "dataset_output" / "samples"
    )
    reports_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "reports")

    # --- Raw input ---
    raw_export_dirs: list[Path] = Field(default_factory=list)

    # --- MinHash LSH ---
    minhash_num_perm: int = 128
    minhash_threshold: float = 0.80
    minhash_shingle_size: int = 3

    # --- Dialogue reconstruction ---
    max_reply_time_gap_hours: int = 48
    min_question_words: int = 3
    min_answer_words: int = 4
    max_thread_depth: int = 15
    max_sft_turns: int = 32

    # --- Sentiment lexicon ---
    sentiment_dict: dict[str, int] = Field(
        default_factory=lambda: {
            # Positive
            "хорошо": 1,
            "отлично": 2,
            "прекрасно": 2,
            "супер": 2,
            "круто": 1,
            "класс": 1,
            "спасибо": 1,
            "благодарю": 1,
            "лайк": 1,
            "годно": 1,
            "топ": 1,
            "лучший": 1,
            "кайф": 1,
            "плюс": 1,
            "согласен": 1,
            "помогло": 2,
            "решено": 2,
            "заработало": 2,
            "стабильно": 1,
            "быстро": 1,
            "удобно": 1,
            "рекомендую": 2,
            "огонь": 2,
            "полезно": 1,
            # Negative
            "плохо": -1,
            "ужасно": -2,
            "отстой": -1,
            "фигня": -1,
            "сложно": -1,
            "трудно": -1,
            "беда": -1,
            "проблема": -1,
            "баг": -1,
            "ошибка": -1,
            "сбой": -1,
            "упало": -2,
            "краш": -2,
            "сломалось": -2,
            "тормозит": -1,
            "лагает": -1,
            "дрянь": -2,
            "хлам": -2,
            "неудобно": -1,
            "минус": -1,
            "говно": -2,
            "бред": -1,
            "дичь": -1,
            # Neutral
            "ок": 0,
            "нормально": 0,
            "средне": 0,
            "нейтрально": 0,
        }
    )

    # --- Stopwords ---
    stopwords_ru: set[str] = Field(
        default_factory=lambda: {
            "и",
            "в",
            "на",
            "с",
            "по",
            "к",
            "у",
            "о",
            "от",
            "за",
            "для",
            "без",
            "из",
            "до",
            "при",
            "через",
            "об",
            "же",
            "бы",
            "ещё",
            "еще",
            "уже",
            "если",
            "что",
            "чтобы",
            "потому",
            "так",
            "как",
            "ну",
            "вот",
            "это",
            "этот",
            "эта",
            "эти",
            "а",
            "но",
            "или",
            "либо",
            "да",
            "нет",
            "не",
            "ни",
            "кто",
            "где",
            "куда",
            "когда",
            "почему",
            "зачем",
            "чей",
            "чья",
            "чьё",
            "чьи",
            "тот",
            "та",
            "то",
            "те",
            "весь",
            "вся",
            "всё",
            "все",
            "мой",
            "твой",
            "свой",
            "наш",
            "ваш",
            "их",
            "его",
            "её",
            "ее",
            "им",
            "ему",
            "ей",
            "нам",
            "вам",
            "ими",
            "мной",
            "тобой",
            "собой",
            "нами",
            "вами",
            "сам",
            "сама",
            "само",
            "сами",
            "тут",
            "там",
            "здесь",
            "откуда",
            "тоже",
            "также",
            "только",
            "лишь",
            "хоть",
            "хотя",
            "будто",
            "словно",
            "точно",
            "разве",
            "неужели",
            "даже",
            "просто",
            "очень",
            "совсем",
            "вообще",
            "тогда",
            "сейчас",
            "потом",
            "после",
            "всегда",
            "никогда",
            "иногда",
            "часто",
            "редко",
            "можно",
            "нужно",
            "надо",
            "нельзя",
            "будет",
            "было",
            "быть",
            "есть",
            "были",
            "будут",
        }
    )

    # --- Nested settings ---
    pii_audit: PIIAuditSettings = Field(default_factory=PIIAuditSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)
    benchmarks: BenchmarkSettings = Field(default_factory=BenchmarkSettings)

    # --- Domain taxonomy (loaded from params.yaml with fallback) ---
    domain_taxonomy: dict[str, DomainTaxonomyItem] = Field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._load_domain_taxonomy()
        self._auto_discover_exports()

    def _load_domain_taxonomy(self) -> None:
        """Load domain taxonomy from params.yaml, fallback to hardcoded defaults."""
        params_path = self.base_dir / "params.yaml"
        if params_path.exists():
            try:
                with params_path.open("r", encoding="utf-8") as f:
                    params = yaml.safe_load(f) or {}
                taxonomy = params.get("domain_taxonomy", {})
                if taxonomy:
                    for domain, info in taxonomy.items():
                        if isinstance(info, dict) and "title" in info and "keywords" in info:
                            self.domain_taxonomy[domain] = DomainTaxonomyItem(**info)
                    return
            except Exception:
                pass  # Fall through to hardcoded

        # Hardcoded fallback
        self.domain_taxonomy = {
            "ai_ml_nlp": DomainTaxonomyItem(
                title="AI, Machine Learning & NLP",
                keywords=[
                    "ai",
                    "ml",
                    "nlp",
                    "llm",
                    "deepseek",
                    "chatgpt",
                    "gpt-4",
                    "gpt",
                    "claude",
                    "openai",
                    "anthropic",
                    "llama",
                    "llama3",
                    "mistral",
                    "qwen",
                    "gemini",
                    "pytorch",
                    "tensorflow",
                    "transformer",
                    "huggingface",
                    "lora",
                    "qlora",
                    "finetune",
                    "fine-tuning",
                    "sft",
                    "dpo",
                    "rlhf",
                    "rag",
                    "embedding",
                    "embeddings",
                    "vector",
                    "qdrant",
                    "chroma",
                    "pinecone",
                    "weaviate",
                    "bert",
                    "rubert",
                    "tokens",
                    "token",
                    "tokenizer",
                    "cuda",
                    "gpu",
                    "vram",
                    "diffusion",
                    "comfyui",
                    "midjourney",
                    "whisper",
                    "ollama",
                    "vllm",
                ],
            ),
            "backend_databases": DomainTaxonomyItem(
                title="Backend Development & Databases",
                keywords=[
                    "python",
                    "fastapi",
                    "django",
                    "flask",
                    "pydantic",
                    "asyncio",
                    "aiohttp",
                    "golang",
                    "go",
                    "rust",
                    "java",
                    "kotlin",
                    "spring",
                    "c++",
                    "c#",
                    ".net",
                    "node",
                    "nodejs",
                    "express",
                    "nest",
                    "nestjs",
                    "php",
                    "laravel",
                    "sql",
                    "postgresql",
                    "postgres",
                    "mysql",
                    "sqlite",
                    "mongodb",
                    "redis",
                    "clickhouse",
                    "elasticsearch",
                    "kafka",
                    "rabbitmq",
                    "celery",
                    "graphql",
                    "rest",
                    "grpc",
                    "orm",
                    "sqlalchemy",
                    "alembic",
                    "database",
                    "бд",
                    "база",
                ],
            ),
            "frontend_ui": DomainTaxonomyItem(
                title="Frontend & Mobile Development",
                keywords=[
                    "javascript",
                    "typescript",
                    "js",
                    "ts",
                    "react",
                    "reactjs",
                    "nextjs",
                    "next",
                    "vue",
                    "vuejs",
                    "nuxt",
                    "angular",
                    "svelte",
                    "html",
                    "css",
                    "scss",
                    "sass",
                    "tailwind",
                    "shadcn",
                    "bootstrap",
                    "webpack",
                    "vite",
                    "frontend",
                    "фронт",
                    "flutter",
                    "react native",
                    "ios",
                    "swift",
                    "android",
                    "kotlin multiplatform",
                ],
            ),
            "devops_infra": DomainTaxonomyItem(
                title="DevOps, Cloud & Infrastructure",
                keywords=[
                    "docker",
                    "docker-compose",
                    "k8s",
                    "kubernetes",
                    "helm",
                    "linux",
                    "ubuntu",
                    "debian",
                    "centos",
                    "alpine",
                    "nginx",
                    "caddy",
                    "traefik",
                    "ci/cd",
                    "ci",
                    "cd",
                    "github actions",
                    "gitlab",
                    "jenkins",
                    "ansible",
                    "terraform",
                    "aws",
                    "gcp",
                    "azure",
                    "hetzner",
                    "ovh",
                    "selectel",
                    "yandex cloud",
                    "timeweb",
                    "vps",
                    "vds",
                    "server",
                    "сервер",
                    "хостинг",
                    "деплой",
                    "мониторинг",
                    "prometheus",
                    "grafana",
                    "sentry",
                    "loki",
                ],
            ),
            "business_legal_fintech": DomainTaxonomyItem(
                title="IT Business, Fintech & Legal",
                keywords=[
                    "бизнес",
                    "стартап",
                    "startup",
                    "саас",
                    "saas",
                    "b2b",
                    "b2c",
                    "ооо",
                    "ип",
                    "налог",
                    "налоги",
                    "бухгалтерия",
                    "бух",
                    "договор",
                    "оферта",
                    "юрист",
                    "stripe",
                    "paypal",
                    "крипта",
                    "crypto",
                    "usdt",
                    "btc",
                    "eth",
                    "ton",
                    "платежи",
                    "эквайринг",
                    "банк",
                    "счет",
                    "инвойс",
                    "релокация",
                    "грузия",
                    "армения",
                    "кипр",
                    "оаэ",
                    "дубай",
                    "казахстан",
                    "венчур",
                    "инвестор",
                    "монетизация",
                    "продажи",
                    "маркетинг",
                    "лиды",
                    "выручка",
                    "mrr",
                    "arr",
                ],
            ),
            "sysadmin_security": DomainTaxonomyItem(
                title="System Administration & Cybersecurity",
                keywords=[
                    "vpn",
                    "wireguard",
                    "vless",
                    "shadowsocks",
                    "xray",
                    "proxy",
                    "прокси",
                    "security",
                    "безопасность",
                    "ssl",
                    "tls",
                    "certbot",
                    "ddos",
                    "firewall",
                    "iptables",
                    "ssh",
                    "auth",
                    "jwt",
                    "oauth",
                    "penetration",
                    "хэк",
                    "уязвимость",
                ],
            ),
            "career_team_management": DomainTaxonomyItem(
                title="Career, HR & Engineering Management",
                keywords=[
                    "зарплата",
                    "зп",
                    "вакансия",
                    "найм",
                    "собес",
                    "собеседование",
                    "резюме",
                    "джун",
                    "мидл",
                    "сеньор",
                    "тимлид",
                    "lead",
                    "pm",
                    "cto",
                    "оффер",
                    "удаленка",
                    "офис",
                    "рейт",
                    "апворк",
                    "upwork",
                    "фриланс",
                    "аутсорс",
                ],
            ),
            "general_tech_chat": DomainTaxonomyItem(
                title="General Tech & Community Discussions",
                keywords=[
                    "cursor",
                    "vscode",
                    "ide",
                    "macbook",
                    "m1",
                    "m2",
                    "m3",
                    "thinkpad",
                    "ноут",
                    "железо",
                    "клавиатура",
                    "монитор",
                    "телеграм",
                    "tg",
                    "бот",
                    "bot",
                ],
            ),
        }

    def _auto_discover_exports(self) -> None:
        """Auto-discover ChatExport directories if not explicitly set."""
        if not self.raw_export_dirs:
            self.raw_export_dirs = sorted([p for p in self.base_dir.glob("ChatExport_*") if p.is_dir()])
            if not self.raw_export_dirs:
                self.raw_export_dirs = [
                    self.base_dir / "ChatExport_2026-08-21",
                    self.base_dir / "ChatExport_2026-08-22",
                    self.base_dir / "ChatExport_2026-08-23",
                ]

    # --- Computed properties for backward compatibility ---
    @property
    def BASE_DIR(self) -> Path:
        return self.base_dir

    @property
    def DATA_DIR(self) -> Path:
        return self.data_dir

    @property
    def OUTPUT_DIR(self) -> Path:
        return self.output_dir

    @property
    def PARQUET_OUTPUT_DIR(self) -> Path:
        return self.parquet_output_dir

    @property
    def JSONL_OUTPUT_DIR(self) -> Path:
        return self.jsonl_output_dir

    @property
    def SAMPLES_OUTPUT_DIR(self) -> Path:
        return self.samples_output_dir

    @property
    def REPORTS_DIR(self) -> Path:
        return self.reports_dir

    @property
    def RAW_EXPORT_DIRS(self) -> list[Path]:
        return self.raw_export_dirs

    @property
    def MINHASH_NUM_PERM(self) -> int:
        return self.minhash_num_perm

    @property
    def MINHASH_THRESHOLD(self) -> float:
        return self.minhash_threshold

    @property
    def MINHASH_SHINGLE_SIZE(self) -> int:
        return self.minhash_shingle_size

    @property
    def MAX_REPLY_TIME_GAP_HOURS(self) -> int:
        return self.max_reply_time_gap_hours

    @property
    def MIN_QUESTION_WORDS(self) -> int:
        return self.min_question_words

    @property
    def MIN_ANSWER_WORDS(self) -> int:
        return self.min_answer_words

    @property
    def MAX_THREAD_DEPTH(self) -> int:
        return self.max_thread_depth

    @property
    def MAX_SFT_TURNS(self) -> int:
        return self.max_sft_turns

    @property
    def SENTIMENT_DICT(self) -> dict[str, int]:
        return self.sentiment_dict

    @property
    def STOPWORDS_RU(self) -> set[str]:
        return self.stopwords_ru

    @property
    def DOMAIN_TAXONOMY(self) -> dict[str, dict[str, Any]]:
        return {k: v.model_dump() for k, v in self.domain_taxonomy.items()}

    def load_params(self) -> dict[str, Any]:
        """Load raw params.yaml for DVC/drift monitoring."""
        params_path = self.base_dir / "params.yaml"
        if params_path.exists():
            try:
                with params_path.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}


# Global instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection friendly getter."""
    return settings
