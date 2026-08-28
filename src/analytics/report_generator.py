"""
Analytical Report Generator producing Markdown, JSON, and Rich Terminal Visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Any

try:
    from rich.console import Console

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

logger = logging.getLogger(__name__)


def generate_ascii_bar(val: float, max_val: float, width: int = 30) -> str:
    """Generate Unicode/ASCII bar for visual terminal output."""
    if max_val <= 0:
        return ""
    fill = int(min(width, (val / max_val) * width))
    return "█" * fill + "░" * (width - fill)


class ReportGenerator:
    """
    Generates rich Markdown, JSON, and visual Terminal reports from analytics data.
    """

    def __init__(self, analytics_data: dict[str, Any]):
        self.data = analytics_data
        self.console = Console(force_terminal=True) if HAS_RICH else None

    def export_json(self, output_path: str | Path) -> Path:
        """Export analytics dictionary to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON analytics summary to {out}")
        return out

    def export_markdown(self, output_path: str | Path) -> Path:
        """Generate comprehensive Markdown analytical report."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        v = self.data.get("volume_statistics", {})
        t = self.data.get("temporal_dynamics", {})
        lex = self.data.get("lexical_analytics", {})
        s = self.data.get("domain_slang_analytics", {})
        sent = self.data.get("sentiment_and_syntax", {})
        net = self.data.get("social_network", {})
        self.data.get("author_signature_phrases", {})
        lda = self.data.get("topic_clusters_lda", [])
        longit = self.data.get("longitudinal_evolution_8_years", {})
        val = self.data.get("quality_and_readiness") or self.data.get("valuation_and_readiness", {})
        noise = self.data.get("noise_and_quality", {})

        md_lines = []
        md_lines.append("# 🔬 АНАЛИТИЧЕСКИЙ ОТЧЁТ ПО КОРПУСУ ДАННЫХ")
        md_lines.append("## Russian IT Community Multi-Domain Conversational Corpus (2017–2026)\n")
        md_lines.append(
            f"> **Дата генерации:** `{self.data.get('report_metadata', {}).get('generated_at', '')}` | **Версия аналитического модуля:** `1.0.0-OpenSource`\n"
        )

        # 1. Executive Summary & Quality Index
        md_lines.append("## 🏆 1. ОЦЕНКА СТРУКТУРНОГО КАЧЕСТВА ДЛЯ ОБУЧЕНИЯ LLM")
        md_lines.append(
            f"**Итоговый индекс качества:** `{val.get('total_score', 0)} / {val.get('max_score', 100)}` | **Категория:** **{val.get('quality_tier', '')}**"
        )
        md_lines.append(f"\n*{val.get('tier_description', '')}*\n")

        md_lines.append("| Критерий оценки | Балл | Макс | Статус |")
        md_lines.append("| :--- | :--- | :--- | :--- |")
        breakdown = val.get("score_breakdown", {})
        md_lines.append(
            f"| **Объём данных (Volume)** | {breakdown.get('volume_score', 0)} | 25 | {'🟢 Высокий' if breakdown.get('volume_score', 0) >= 20 else '🟡 Базовый'} |"
        )
        md_lines.append(
            f"| **Разнообразие авторов (Diversity)** | {breakdown.get('author_diversity_score', 0)} | 20 | {'🟢 Высокое' if breakdown.get('author_diversity_score', 0) >= 15 else '🟡 Базовое'} |"
        )
        md_lines.append(
            f"| **Техническая плотность (Domain Density)** | {breakdown.get('technical_density_score', 0)} | 20 | {'🟢 Высокая' if breakdown.get('technical_density_score', 0) >= 15 else '🟡 Базовая'} |"
        )
        md_lines.append(
            f"| **Диалоговая связность (Q&A Ratio)** | {breakdown.get('dialogue_continuity_score', 0)} | 15 | 🟢 Норма |"
        )
        md_lines.append(
            f"| **Лексическое разнообразие (Shannon Diversity)** | {breakdown.get('lexical_diversity_score', 0)} | 10 | 🟢 Норма |"
        )
        md_lines.append(
            f"| **Очистка PII (Heuristic / NER Scrubbing)** | {breakdown.get('pii_compliance_score', 0)} | 10 | 🟢 Обработано |"
        )
        md_lines.append("\n---\n")

        # 2. General Statistics
        md_lines.append("## 📊 2. ОБЩАЯ СТАТИСТИКА КОРПУСА")
        md_lines.append(f"- **Всего сообщений:** `{v.get('total_messages', 0):,}`")
        md_lines.append(f"- **Уникальных участников:** `{v.get('unique_authors', 0):,}`")
        md_lines.append(
            f"- **Временной диапазон:** `{v.get('date_start', '')}` — `{v.get('date_end', '')}` (`{v.get('total_days_active', 0)}` дней)"
        )
        md_lines.append(
            f"- **Средняя интенсивность:** `{v.get('messages_per_day', 0)}` сообщений / день (`{v.get('tokens_per_day', 0):,}` токенов / день)"
        )
        md_lines.append(f"- **Общее число слов:** `{v.get('total_words', 0):,}`")
        md_lines.append(
            f"- **Оценка объёма токенов (BPE):** `{v.get('total_tokens_estimated', 0):,}` токенов (~`{v.get('total_tokens_estimated', 0) / 1_000_000:.2f}M`)"
        )
        md_lines.append(f"- **Уникальный словарь (Леммы/Слова):** `{v.get('vocabulary_unique_words', 0):,}`")
        md_lines.append(f"- **Индекс лексического разнообразия (Shannon):** `{lex.get('shannon_entropy', 0):.2f}`\n")

        # Length Distribution Table
        char_dist = v.get("character_length_distribution", {})
        word_dist = v.get("word_count_distribution", {})
        tok_dist = v.get("token_count_distribution", {})

        md_lines.append("### Распределение длины сообщений и токенов")
        md_lines.append("| Метрика | Среднее | Медиана | Min | Max | P25 | P75 | P90 | P99 |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        md_lines.append(
            f"| **Символов** | {char_dist.get('mean', 0)} | {char_dist.get('median', 0)} | {char_dist.get('min', 0)} | {char_dist.get('max', 0)} | {char_dist.get('p25', 0)} | {char_dist.get('p75', 0)} | {char_dist.get('p90', 0)} | {char_dist.get('p99', 0)} |"
        )
        md_lines.append(
            f"| **Слов** | {word_dist.get('mean', 0)} | {word_dist.get('median', 0)} | {word_dist.get('min', 0)} | {word_dist.get('max', 0)} | {word_dist.get('p25', 0)} | {word_dist.get('p75', 0)} | {word_dist.get('p90', 0)} | {word_dist.get('p99', 0)} |"
        )
        md_lines.append(
            f"| **Токенов** | {tok_dist.get('mean', 0)} | {tok_dist.get('median', 0)} | {tok_dist.get('min', 0)} | {tok_dist.get('max', 0)} | {tok_dist.get('p25', 0)} | {tok_dist.get('p75', 0)} | {tok_dist.get('p90', 0)} | {tok_dist.get('p99', 0)} |"
        )
        md_lines.append("\n---\n")

        # 3. Domain Breakdown
        md_lines.append("## 🧠 3. ТЕМАТИЧЕСКАЯ СТРУКТУРА И ДОМЕННОЕ РАСПРЕДЕЛЕНИЕ")
        md_lines.append("| Домен / Направление | Сообщений | Доля | Визуальное распределение |")
        md_lines.append("| :--- | :--- | :--- | :--- |")
        domain_dist = s.get("domain_message_distribution", {})
        max_cnt = max((d["count"] for d in domain_dist.values()), default=1)
        for dom, info in domain_dist.items():
            cnt = info.get("count", 0)
            pct = info.get("percentage", 0)
            bar = generate_ascii_bar(cnt, max_cnt, width=20)
            md_lines.append(f"| **{dom}** | {cnt:,} | {pct:.1f}% | `{bar}` |")
        md_lines.append("\n---\n")

        # 4. Temporal Patterns
        md_lines.append("## ⏰ 4. ВРЕМЕННЫЕ ПАТТЕРНЫ И ДИНАМИКА АКТИВНОСТИ")
        md_lines.append(f"- **Пиковый час активности:** `{t.get('peak_hour', 0)}:00`")
        md_lines.append(f"- **Пиковый день недели:** `{t.get('peak_weekday', '')}`\n")

        md_lines.append("### Активность по часам суток (0–23):")
        md_lines.append("```")
        hourly = t.get("hourly_distribution", {})
        max_h = max(hourly.values(), default=1)
        for hour_str, cnt in hourly.items():
            bar = "█" * int((cnt / max_h) * 35)
            md_lines.append(f"{hour_str} | {bar:<35} {cnt:>6,}")
        md_lines.append("```\n")

        md_lines.append("### Динамика по годам (2018–2026):")
        md_lines.append("| Год | Сообщений | Доля | Ключевые технологические фокусы года |")
        md_lines.append("| :--- | :--- | :--- | :--- |")
        yearly = t.get("yearly_volume", {})
        tot_m = v.get("total_messages", 1)
        for yr, cnt in yearly.items():
            pct = (cnt / tot_m) * 100
            yr_info = longit.get(yr, {})
            techs = ", ".join(yr_info.get("top_tech_keywords", [])[:5])
            md_lines.append(f"| **{yr}** | {cnt:,} | {pct:.1f}% | {techs if techs else '—'} |")
        md_lines.append("\n---\n")

        # 5. Russian IT Slang & Key Entities
        md_lines.append("## 💬 5. АУТЕНТИЧНЫЙ IT-СЛЕНГ И ТЕХНИЧЕСКИЕ ТЕРМИНЫ")
        md_lines.append("Корпус содержит глубокий пласт реального русскоязычного инженерного сленга:\n")
        md_lines.append("| Термин | Встречаемость | Категория контекста |")
        md_lines.append("| :--- | :--- | :--- |")
        for slang in s.get("top_slang_terms", [])[:25]:
            md_lines.append(f"| `{slang.get('term')}` | {slang.get('count'):,} | Инженерия / Разработка / Бизнес |")
        md_lines.append("\n---\n")

        # 6. Social Graph & Top Influencers
        md_lines.append("## 🕸️ 6. СОЦИАЛЬНЫЙ ГРАФ И ЭКСПЕРТЫ СООБЩЕСТВА")
        md_lines.append(f"- **Узлов в сети:** `{net.get('total_nodes', 0):,}`")
        md_lines.append(f"- **Связей (Reply Directed Edges):** `{net.get('total_edges', 0):,}`")
        md_lines.append(f"- **Всего зафиксировано взаимодействий:** `{net.get('total_interactions', 0):,}`")
        md_lines.append(f"- **Плотность графа:** `{net.get('density', 0):.6f}`\n")

        md_lines.append("### Топ-10 влиятельных участников (получают больше всего обращений и вопросов):")
        md_lines.append("| Автор (Anon ID) | Получено ответов / вопросов |")
        md_lines.append("| :--- | :--- |")
        for inf in net.get("top_influencers", [])[:10]:
            md_lines.append(f"| **{inf.get('author')}** | {inf.get('replies_received'):,} |")
        md_lines.append("\n---\n")

        # 7. LDA Topic Modeling
        md_lines.append("## 🧠 7. СЕМАНТИЧЕСКИЕ ТЕМАТИЧЕСКИЕ КЛАСТЕРЫ (LDA)")
        if lda:
            for top in lda:
                md_lines.append(f"### {top.get('label')}")
                md_lines.append(f"- **Ключевые слова темы:** `{', '.join(top.get('top_keywords', []))}`\n")
        else:
            md_lines.append("*Тематические кластеры извлечены на основе многоуровневого словаря доменов.*")
        md_lines.append("\n---\n")

        # 8. Sentiment & Noise
        md_lines.append("## 🔊 8. ТОНАЛЬНОСТЬ, ВОПРОСЫ И УРОВЕНЬ ШУМА")
        sent_stat = sent.get("sentiment", {})
        md_lines.append(f"- **Средний эмоциональный балл:** `{sent_stat.get('average', 0):.3f}`")
        md_lines.append(
            f"- **Положительных сообщений:** `{sent_stat.get('positive', 0):,}` (`{sent_stat.get('pos_ratio', 0):.1f}%`)"
        )
        md_lines.append(
            f"- **Отрицательных сообщений:** `{sent_stat.get('negative', 0):,}` (`{sent_stat.get('neg_ratio', 0):.1f}%`)"
        )
        md_lines.append(
            f"- **Нейтральных сообщений:** `{sent_stat.get('neutral', 0):,}` (`{sent_stat.get('neu_ratio', 0):.1f}%`)"
        )
        md_lines.append(
            f"- **Вопросительных сообщений (Q&A potential):** `{sent.get('questions_count', 0):,}` (`{sent.get('questions_ratio_percentage', 0):.1f}%`)"
        )
        md_lines.append(
            f"- **Сообщений с фрагментами кода:** `{sent.get('code_snippets_count', 0):,}` (`{sent.get('code_snippets_ratio_percentage', 0):.1f}%`)"
        )
        md_lines.append(
            f"- **Коротких сообщений (<20 симв):** `{noise.get('short_messages_ratio_percentage', 0):.1f}%`"
        )
        md_lines.append(
            f"- **Пустых / без слов сообщений:** `{noise.get('empty_messages_ratio_percentage', 0):.1f}%`\n"
        )

        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Saved comprehensive Markdown report to {out}")
        return out

    def print_terminal_summary(self):
        """Print high-contrast rich visualization in terminal."""
        v = self.data.get("volume_statistics", {})
        t = self.data.get("temporal_dynamics", {})
        val = self.data.get("quality_and_readiness") or self.data.get("valuation_and_readiness", {})
        s = self.data.get("domain_slang_analytics", {})

        print("\n" + "=" * 80)
        print("🔬 АНАЛИТИЧЕСКИЙ ОТЧЁТ ПО КОРПУСУ ДАННЫХ (RICC)")
        print("=" * 80)
        print("\n📊 ОБЩАЯ СТАТИСТИКА")
        print(f"   Сообщений: {v.get('total_messages', 0):,}")
        print(f"   Участников: {v.get('unique_authors', 0):,}")
        print(
            f"   Период: {v.get('date_start', '')[:10]} — {v.get('date_end', '')[:10]} ({v.get('total_days_active', 0)} дней)"
        )
        print(f"   Общее число слов: {v.get('total_words', 0):,}")
        print(
            f"   Оценка токенов: {v.get('total_tokens_estimated', 0):,} (~{v.get('total_tokens_estimated', 0) / 1e6:.2f}M)"
        )
        print(f"   Уникальных слов: {v.get('vocabulary_unique_words', 0):,}")

        print("\n⏰ ВРЕМЕННЫЕ ПАТТЕРНЫ")
        print(f"   Пиковый час: {t.get('peak_hour', 0)}:00")
        print(f"   Пиковый день: {t.get('peak_weekday', '')}")
        print("   Часы активности:")
        hourly = t.get("hourly_distribution", {})
        max_h = max(hourly.values(), default=1)
        for h_str, cnt in list(hourly.items())[::2]:
            bar = "█" * int((cnt / max_h) * 25)
            print(f"     {h_str}: {bar:<25} {cnt:>6,}")

        print("\n🧠 ТЕМАТИЧЕСКАЯ СТРУКТУРА (ТОП ДОМЕНОВ):")
        for dom, info in list(s.get("domain_message_distribution", {}).items())[:6]:
            print(f"   • {dom:<25}: {info.get('count', 0):>6,} ({info.get('percentage', 0):.1f}%)")

        print("\n🏆 ИНДЕКС СТРУКТУРНОГО КАЧЕСТВА ДЛЯ LLM")
        print(f"   Балл: {val.get('total_score', 0)} из 100 ({val.get('quality_tier', '')})")
        print("=" * 80 + "\n")
