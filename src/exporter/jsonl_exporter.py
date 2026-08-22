"""
JSONL Exporters for standard LLM fine-tuning formats (ShareGPT, Alpaca, OpenAI ChatML).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

from src.ingestion.schema import SFTDialogue

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_DEFAULT = (
    "Ты — опытный русскоязычный IT-консультант, Senior разработчик и архитектор. "
    "Отвечай на технические вопросы точно, структурированно, опираясь на реальные практики разработки, "
    "DevOps, архитектуры и IT-бизнеса."
)


class JSONLExporter:
    """
    Exports SFT dialogues to standard JSONL formats for various training frameworks (Unsloth, Axolotl, TRL).
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_sharegpt(self, dialogues: List[SFTDialogue], file_name: str = "sft_sharegpt_format.jsonl") -> Path:
        """
        Export dialogues to ShareGPT format:
        {"id": "...", "topic": "...", "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        """
        out_path = self.output_dir / file_name
        logger.info(f"Exporting {len(dialogues)} dialogues to ShareGPT JSONL at {out_path}...")

        with open(out_path, "w", encoding="utf-8") as f:
            for d in dialogues:
                convs = []
                for m in d.messages:
                    from_role = "human" if m.role == "user" else "gpt"
                    convs.append({
                        "from": from_role,
                        "value": m.content,
                    })

                record = {
                    "id": f"dialogue_{d.thread_id}",
                    "chat": d.chat_name,
                    "domain": d.topic_domain,
                    "tags": d.topic_tags,
                    "quality_score": d.quality_score,
                    "conversations": convs,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved ShareGPT JSONL at {out_path}")
        return out_path

    def export_alpaca(self, dialogues: List[SFTDialogue], file_name: str = "sft_alpaca_format.jsonl") -> Path:
        """
        Export dialogues to Alpaca/Instruction format:
        {"instruction": "...", "input": "...", "output": "...", "domain": "..."}
        """
        out_path = self.output_dir / file_name
        logger.info(f"Exporting {len(dialogues)} instruction pairs to Alpaca JSONL at {out_path}...")

        pair_count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for d in dialogues:
                # Pair consecutive user questions and assistant answers
                msgs = d.messages
                for i in range(len(msgs) - 1):
                    if msgs[i].role == "user" and msgs[i+1].role == "assistant":
                        record = {
                            "instruction": msgs[i].content,
                            "input": "",
                            "output": msgs[i+1].content,
                            "domain": d.topic_domain,
                            "tags": d.topic_tags,
                            "quality_score": d.quality_score,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        pair_count += 1

        logger.info(f"Saved {pair_count} Alpaca instruction pairs at {out_path}")
        return out_path

    def export_openai_chatml(
        self,
        dialogues: List[SFTDialogue],
        file_name: str = "sft_openai_messages.jsonl",
        system_prompt: str = SYSTEM_PROMPT_DEFAULT,
    ) -> Path:
        """
        Export dialogues to OpenAI ChatML / TRL format:
        {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        """
        out_path = self.output_dir / file_name
        logger.info(f"Exporting {len(dialogues)} dialogues to OpenAI ChatML JSONL at {out_path}...")

        with open(out_path, "w", encoding="utf-8") as f:
            for d in dialogues:
                chat_messages = [{"role": "system", "content": system_prompt}]
                for m in d.messages:
                    chat_messages.append({
                        "role": m.role,
                        "content": m.content,
                    })

                record = {
                    "id": f"chat_{d.thread_id}",
                    "domain": d.topic_domain,
                    "tags": d.topic_tags,
                    "quality": d.quality_score,
                    "messages": chat_messages,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved OpenAI ChatML JSONL at {out_path}")
        return out_path
