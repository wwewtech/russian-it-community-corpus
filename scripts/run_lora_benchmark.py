"""run_lora_benchmark.py — honest held-out benchmarks for LoRA adapters.

Compares a base model against its trained LoRA adapter on the **held-out tail**
of the SFT corpus (the last ``--tail`` dialogues, reserved and excluded from
training by ``batch_train_zoo.py --holdout-tail``):

* mean NLL / perplexity over held-out dialogues (base vs LoRA);
* greedy generation snapshots for a fixed set of Russian IT prompts;
* optional ``lm_eval`` tasks when the package is installed
  (``pip install lm-eval``) via ``--lm-eval-tasks``.

Perplexity here is full-sequence (prompt + answer) so the base and LoRA runs are
measured identically; the training objective itself masks prompt tokens.

Usage::

    python scripts/run_lora_benchmark.py --adapter lora_adapters/qwen2.5_1.5b_instruct
    python scripts/run_lora_benchmark.py --adapter a --adapter b --lm-eval-tasks humaneval

Reports are written to ``reports/benchmark_<adapter>.json`` and
``reports/benchmark_<adapter>.md``. Exit code 1 if any adapter fails.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from collections import deque
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bootstrap import setup_runtime_env  # noqa: E402

setup_runtime_env()

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LoRABenchmark")

DEFAULT_DATASET = "dataset_output/jsonl/sft_openai_messages.jsonl"

BENCH_PROMPTS = [
    "Как настроить приём платежей для SaaS из РФ?",
    "Объясни разницу между монолитом и микросервисами.",
    "Почему Docker-контейнер падает с ошибкой OOMKilled?",
    "Как организовать CI/CD для Python-проекта?",
    "Что такое индексы в PostgreSQL и когда они не помогают?",
    "Как защитить публичный API от злоупотреблений?",
    "Стоит ли использовать Kafka в небольшом проекте?",
    "Как мигрировать с MySQL на PostgreSQL без простоя?",
]


def read_holdout_tail(dataset_path: Path, tail: int) -> list[dict[str, Any]]:
    """Read the last ``tail`` valid JSONL dialogues (the training holdout)."""
    rows: deque[dict[str, Any]] = deque(maxlen=tail)
    with dataset_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(cast(dict[str, Any], json.loads(line)))
            except json.JSONDecodeError:
                continue
    return list(rows)


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dialogue_text(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return cast(str, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    except Exception:
        return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def _prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return cast(str, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception:
        return prompt


def load_base(model_name: str) -> tuple[Any, Any]:
    tokenizer: Any = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<|endoftext|>"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    # Some remote-code models (Phi-3.5) call the Cache.get_usable_length API
    # removed in transformers 4.57 whenever a KV cache is requested; without
    # this, even generate() crashes on them. Benchmarks trade cache speed for
    # compatibility on every model.
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    return model, tokenizer


def compute_perplexity(
    model: Any, tokenizer: Any, dialogues: list[dict[str, Any]], max_seq_length: int, limit: int
) -> dict[str, Any]:
    """Full-sequence mean NLL / perplexity over up to ``limit`` held-out dialogues."""
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    used = 0
    model.eval()
    with torch.no_grad():
        for example in dialogues[:limit]:
            messages = example.get("messages", [])
            if not messages:
                continue
            text = _dialogue_text(tokenizer, messages)
            if not text.strip():
                continue
            enc = tokenizer(text, max_length=max_seq_length, truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                continue
            out = model(input_ids=input_ids, labels=input_ids)
            total_nll += float(out.loss) * (input_ids.shape[1] - 1)
            total_tokens += input_ids.shape[1] - 1
            used += 1
    if total_tokens == 0:
        return {"status": "no_tokens", "mean_nll": None, "perplexity": None, "dialogues": 0, "tokens": 0}
    mean_nll = total_nll / total_tokens
    return {
        "status": "ok",
        "mean_nll": round(mean_nll, 4),
        "perplexity": round(math.exp(min(mean_nll, 50.0)), 4),
        "dialogues": used,
        "tokens": total_tokens,
    }


def generate_samples(model: Any, tokenizer: Any, prompts: list[str], max_new_tokens: int) -> list[str]:
    device = next(model.parameters()).device
    model.eval()
    outputs: list[str] = []
    with torch.no_grad():
        for prompt in prompts:
            text = _prompt_text(tokenizer, prompt)
            enc = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=enc["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            outputs.append(tokenizer.decode(gen[0][input_ids.shape[1] :], skip_special_tokens=True).strip())
    return outputs


def run_lm_eval(tasks: str, base_model: str, adapter_dir: Path, out_dir: Path) -> dict[str, Any]:
    if find_spec("lm_eval") is None:
        return {"status": "skipped", "reason": "lm-eval not installed (pip install lm-eval)"}
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={base_model},peft={adapter_dir}",
        "--tasks",
        tasks,
        "--output_path",
        str(out_dir / "lm_eval"),
        "--device",
        "cuda:0" if torch.cuda.is_available() else "cpu",
    ]
    logger.info(f"running lm-eval: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "tasks": tasks,
    }


def benchmark_adapter(adapter_dir: Path, holdout: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    adapter_dir = adapter_dir.resolve()
    config = cast(dict[str, Any], json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8")))
    base_model = str(config["base_model_name_or_path"])
    logger.info(f"benchmarking {adapter_dir.name}: base={base_model}, holdout={len(holdout)} dialogues")

    model, tokenizer = load_base(base_model)
    base_ppl = compute_perplexity(model, tokenizer, holdout, args.max_seq_length, args.ppl_limit)
    base_gens = generate_samples(model, tokenizer, BENCH_PROMPTS, args.max_new_tokens)

    peft_model: Any = cast(Any, PeftModel.from_pretrained(model, str(adapter_dir))).eval()
    lora_ppl = compute_perplexity(peft_model, tokenizer, holdout, args.max_seq_length, args.ppl_limit)
    lora_gens = generate_samples(peft_model, tokenizer, BENCH_PROMPTS, args.max_new_tokens)

    report: dict[str, Any] = {
        "adapter": adapter_dir.name,
        "adapter_dir": str(adapter_dir),
        "base_model": base_model,
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha256(args.dataset),
        "holdout_tail": args.tail,
        "holdout_dialogues": len(holdout),
        "perplexity": {"base": base_ppl, "lora": lora_ppl},
        "generations": [
            {"prompt": prompt, "base": base_out, "lora": lora_out}
            for prompt, base_out, lora_out in zip(BENCH_PROMPTS, base_gens, lora_gens, strict=True)
        ],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "scripts/run_lora_benchmark.py",
    }
    if args.lm_eval_tasks:
        report["lm_eval"] = run_lm_eval(args.lm_eval_tasks, base_model, adapter_dir, args.out_dir)

    del peft_model, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    base = report["perplexity"]["base"]
    lora = report["perplexity"]["lora"]
    lines = [
        f"# LoRA benchmark — `{report['adapter']}`",
        "",
        f"- Base model: `{report['base_model']}`",
        f"- Held-out tail: last {report['holdout_tail']} dialogues of `{report['dataset']}`",
        f"- Evaluated on: {lora['dialogues']} dialogues / {lora['tokens']} tokens",
        "",
        "## Held-out perplexity (full-sequence, lower is better)",
        "",
        "| Model | mean NLL | perplexity |",
        "| --- | ---: | ---: |",
        f"| base | {base['mean_nll']} | {base['perplexity']} |",
        f"| + LoRA | {lora['mean_nll']} | {lora['perplexity']} |",
        "",
        "## Generation snapshots (greedy)",
        "",
    ]
    for item in report["generations"]:
        lines += [
            f"### {item['prompt']}",
            "",
            f"**base**: {item['base'] or '(empty)'}",
            "",
            f"**+ LoRA**: {item['lora'] or '(empty)'}",
            "",
        ]
    if "lm_eval" in report:
        lines += ["## lm-eval", "", f"`{json.dumps(report['lm_eval'], ensure_ascii=False)}`", ""]
    lines += [f"_Generated {report['generated_at_utc']} by `{report['generator']}`._", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Held-out LoRA benchmarks: base vs adapter perplexity + generations")
    parser.add_argument("--adapter", action="append", required=True, type=Path, help="Adapter dir (repeatable)")
    parser.add_argument(
        "--dataset", type=Path, default=Path(DEFAULT_DATASET), help=f"SFT JSONL path (default: {DEFAULT_DATASET})"
    )
    parser.add_argument("--tail", type=int, default=1000, help="Held-out tail size in dialogues (default: 1000)")
    parser.add_argument("--ppl-limit", type=int, default=100, help="Max dialogues for perplexity (default: 100)")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Truncation length (default: 512)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Greedy generation length (default: 128)")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"), help="Report output dir (default: reports)")
    parser.add_argument("--lm-eval-tasks", type=str, default=None, help='Optional lm-eval tasks, e.g. "humaneval"')
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        logger.error(f"dataset not found: {args.dataset}")
        return 1
    holdout = read_holdout_tail(args.dataset, args.tail)
    if not holdout:
        logger.error(f"no dialogues read from {args.dataset}")
        return 1
    if len(holdout) < args.tail:
        logger.warning(f"holdout has only {len(holdout)} dialogues (requested {args.tail})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for adapter_dir in args.adapter:
        if not (adapter_dir / "adapter_config.json").exists():
            logger.error(f"no adapter_config.json in {adapter_dir} — skipping")
            failures += 1
            continue
        try:
            report = benchmark_adapter(adapter_dir, holdout, args)
        except Exception:
            logger.exception(f"benchmark failed for {adapter_dir}")
            failures += 1
            continue
        json_path = args.out_dir / f"benchmark_{report['adapter']}.json"
        md_path = args.out_dir / f"benchmark_{report['adapter']}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        logger.info(
            f"{report['adapter']}: ppl base={report['perplexity']['base']['perplexity']} "
            f"→ lora={report['perplexity']['lora']['perplexity']} | reports: {json_path.name}, {md_path.name}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
