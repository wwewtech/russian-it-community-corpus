"""One-off: regenerate the full SFT JSONL from the real HF parquet.

The local dataset_output/jsonl fixture (100 rows) is a smoke artifact;
the canonical training corpus is the parquet published at the pinned HF
revision. This script re-exports it into the OpenAI ChatML JSONL the
trainer consumes, deterministically.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = "Ты — опытный инженер русскоязычного IT-сообщества. Отвечай точно и по делу."


def main() -> None:
    df = pd.read_parquet("dataset_output/parquet/sft_dialogues.parquet")
    out = Path("dataset_output/jsonl/sft_openai_messages.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
            ok = True
            for m in row["messages"]:
                role, content = m.get("role"), m.get("content")
                if role not in ("user", "assistant") or not content:
                    ok = False
                    break
                msgs.append({"role": role, "content": content})
            if ok and len(msgs) >= 3:
                record = {
                    "id": f"chat_{row['thread_id']}",
                    "domain": row["topic_domain"],
                    "tags": list(row["topic_tags"]),
                    "quality": row["quality_score"],
                    "messages": msgs,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"written: {n}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
