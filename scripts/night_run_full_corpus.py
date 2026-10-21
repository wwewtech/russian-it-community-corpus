"""night_run_full_corpus.py — one-command overnight full-corpus LoRA + benchmarks.

Chains time-boxed full-corpus ``batch_train_zoo.py`` runs with honest held-out
benchmarks from ``run_lora_benchmark.py``:

1. trains each requested model with ``--profile full``, a wall-clock budget
   (``--hours``, graceful stop + save) and checkpoints kept for resume;
2. reserves the last ``--holdout-tail`` dialogues of the corpus for evaluation
   (excluded from training by every run);
3. benchmarks each produced adapter on that tail: base-vs-LoRA perplexity plus
   greedy generation snapshots (``--lm-eval-tasks`` adds lm-eval if installed);
4. writes ``reports/LORA_NIGHT_RUN_SUMMARY.md`` with the whole run digest.

Existing adapters are retrained from scratch (``--force``); interrupted runs
(``checkpoint-*`` without a final adapter) resume from the latest checkpoint.

Example — 10 hours, phi-3.5-mini (3.8B) + qwen2.5-1.5b, 60/40 budget split::

    python scripts/night_run_full_corpus.py --total-hours 10 \
        --models phi_3.5_mini_instruct qwen2.5_1.5b_instruct --split-weights "60,40"

Tip: keep the machine awake for the night (``powercfg /change standby-timeout-ac 0``).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NightRun")

DEFAULT_DATASET = "dataset_output/jsonl/sft_openai_messages.jsonl"
SUMMARY_NAME = "LORA_NIGHT_RUN_SUMMARY.md"


def parse_weights(text: str, count: int) -> list[float]:
    """Parse ``"60,40"`` into normalized weights; empty string means equal split."""
    if not text.strip():
        return [1.0] * count
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != count:
        raise ValueError(f"--split-weights has {len(parts)} value(s) for {count} model(s)")
    weights = [float(p) for p in parts]
    if any(w <= 0 for w in weights):
        raise ValueError("--split-weights values must be positive")
    return weights


def adapter_completed(out_dir: Path) -> bool:
    return (out_dir / "adapter_model.safetensors").exists() and (out_dir / "adapter_config.json").exists()


def has_checkpoints(out_dir: Path) -> bool:
    return any(p.is_dir() for p in out_dir.glob("checkpoint-*"))


def train_command(model: str, hours: float, args: argparse.Namespace) -> list[str]:
    out_dir = args.output_dir / model
    cmd = [
        sys.executable,
        "-m",
        "ricc.lora.batch_train_zoo",
        "--only",
        model,
        "--profile",
        "full",
        "--samples",
        str(args.samples),
        "--steps",
        str(args.max_steps),
        "--hours",
        f"{hours:.3f}",
        "--holdout-tail",
        str(args.holdout_tail),
        "--eval-frac",
        str(args.eval_frac),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output_dir),
        "--report-dir",
        str(args.report_dir),
        "--no-upload",
    ]
    if adapter_completed(out_dir) and not has_checkpoints(out_dir):
        cmd.append("--force")
    if args.keep_checkpoints:
        cmd.append("--keep-checkpoints")
    return cmd


def benchmark_command(adapters: list[Path], args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(SCRIPT_DIR / "run_lora_benchmark.py")]
    for adapter in adapters:
        cmd += ["--adapter", str(adapter)]
    cmd += [
        "--dataset",
        str(args.dataset),
        "--tail",
        str(args.holdout_tail),
        "--ppl-limit",
        str(args.ppl_limit),
        "--max-seq-length",
        str(args.max_seq_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--out-dir",
        str(args.report_dir),
    ]
    if args.lm_eval_tasks:
        cmd += ["--lm-eval-tasks", args.lm_eval_tasks]
    return cmd


def _read_benchmark_summary(report_dir: Path, model: str) -> str:
    path = report_dir / f"benchmark_{model}.json"
    if not path.exists():
        return "—"
    try:
        data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        base = data["perplexity"]["base"]["perplexity"]
        lora = data["perplexity"]["lora"]["perplexity"]
        return f"{base} → {lora}"
    except Exception:
        return "see report"


def render_summary(plan: list[dict[str, Any]], results: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# LoRA night run — full-corpus training + held-out benchmarks",
        "",
        f"- Started: {plan[0]['started_utc']} · finished: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"- Total budget: {args.total_hours} h (bench reserve: {0 if args.skip_benchmarks else args.bench_reserve_hours} h)",
        f"- Corpus: `{args.dataset}` minus the last {args.holdout_tail} dialogues (held-out tail)",
        "- Budget per model: " + ", ".join(f"`{p['model']}` {p['hours']:.2f} h" for p in plan),
        "",
        "| Model | Budget (h) | Training | Adapter | Perplexity base → LoRA |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for result in results:
        bench = _read_benchmark_summary(args.report_dir, result["model"]) if result["bench_rc"] is not None else "—"
        lines.append(
            f"| `{result['model']}` | {result['hours']:.2f} | rc={result['train_rc']} "
            f"({result['train_time_min']:.0f} min) | {'yes' if result['adapter_ok'] else 'no'} | {bench} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Adapters: `{args.output_dir}/<model>/` (incl. `training_meta.json`, `training_log.jsonl`)",
        f"- Benchmarks: `{args.report_dir}/benchmark_<model>.json` + `.md`",
        f"- Zoo run report: `{args.report_dir}/LORA_ZOO_TRAINING_REPORT.md`",
        "",
        "_Note: a single night does not cover a full epoch on a 3.8B model — see `corpus_coverage_pct` "
        "in each `training_meta.json` for the honest fraction of the corpus seen._",
        "",
        "_Generated by `scripts/night_run_full_corpus.py`._",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Overnight full-corpus LoRA training + held-out benchmarks")
    parser.add_argument("--total-hours", type=float, required=True, help="Total wall-clock budget for the night")
    parser.add_argument("--models", nargs="+", required=True, help="Adapter ids from the zoo to train")
    parser.add_argument(
        "--split-weights", type=str, default="", help='Per-model budget share, e.g. "60,40" (default: equal)'
    )
    parser.add_argument("--bench-reserve-hours", type=float, default=1.0, help="Reserve for benchmarks (default: 1.0)")
    parser.add_argument("--holdout-tail", type=int, default=1000, help="Held-out tail dialogues (default: 1000)")
    parser.add_argument("--eval-frac", type=float, default=0.005, help="Inner eval split fraction (default: 0.005)")
    parser.add_argument(
        "--samples", type=int, default=200_000, help="Max training dialogues per model (default: 200000)"
    )
    parser.add_argument("--max-steps", type=int, default=1_000_000, help="Step ceiling per model (default: 1000000)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed (default: 42)")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Training/eval truncation (default: 512)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Benchmark generation length (default: 128)")
    parser.add_argument("--ppl-limit", type=int, default=100, help="Benchmark perplexity dialogues (default: 100)")
    parser.add_argument(
        "--dataset", type=Path, default=Path(DEFAULT_DATASET), help=f"SFT JSONL (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("lora_adapters"), help="Adapter root (default: lora_adapters)"
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"), help="Report dir (default: reports)")
    parser.add_argument("--no-keep-checkpoints", action="store_true", help="Drop checkpoints after each successful run")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Train only, no benchmarks")
    parser.add_argument("--lm-eval-tasks", type=str, default=None, help='Optional lm-eval tasks, e.g. "humaneval"')
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and commands without running")
    args = parser.parse_args(argv)

    args.keep_checkpoints = not args.no_keep_checkpoints

    try:
        weights = parse_weights(args.split_weights, len(args.models))
    except ValueError as exc:
        parser.error(str(exc))
    if args.holdout_tail < 0 or args.samples <= 0 or args.max_steps <= 0:
        parser.error("--holdout-tail must be >= 0 and --samples/--max-steps must be > 0")
    if not args.dataset.exists():
        logger.error(f"dataset not found: {args.dataset}")
        return 1

    reserve = 0.0 if args.skip_benchmarks else args.bench_reserve_hours
    train_budget = args.total_hours - reserve
    if train_budget <= 0:
        parser.error(f"--total-hours must exceed the {reserve} h benchmark reserve")

    weight_sum = sum(weights)
    per_model_hours = [train_budget * w / weight_sum for w in weights]
    if min(per_model_hours) <= 0.05:
        parser.error("per-model budget too small (< 0.05 h); adjust --total-hours/--split-weights")

    plan: list[dict[str, Any]] = []
    for model, hours in zip(args.models, per_model_hours, strict=True):
        plan.append(
            {
                "model": model,
                "hours": hours,
                "train_cmd": train_command(model, hours, args),
                "started_utc": "",
            }
        )
    if args.skip_benchmarks:
        logger.info("benchmarks skipped (--skip-benchmarks)")
    if args.dry_run:
        logger.info(f"plan: total={args.total_hours} h, reserve={reserve} h, train budget={train_budget:.2f} h")
        for entry in plan:
            logger.info(f"  {entry['model']}: {entry['hours']:.2f} h")
        for entry in plan:
            logger.info("  $" + " ".join(entry["train_cmd"]))
        adapters = [args.output_dir / m for m in args.models]
        if not args.skip_benchmarks:
            logger.info("  $" + " ".join(benchmark_command(adapters, args)))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for entry in plan:
        entry["started_utc"] = started

    results: list[dict[str, Any]] = []
    for entry in plan:
        model = entry["model"]
        logger.info(f"=== TRAIN {model}: budget {entry['hours']:.2f} h ===")
        t0 = time.time()
        proc = subprocess.run(cast(list[str], entry["train_cmd"]), cwd=REPO_ROOT, check=False)
        train_min = (time.time() - t0) / 60
        out_dir = args.output_dir / model
        results.append(
            {
                "model": model,
                "hours": entry["hours"],
                "train_rc": proc.returncode,
                "train_time_min": train_min,
                "adapter_ok": adapter_completed(out_dir),
                "bench_rc": None,
            }
        )
        logger.info(
            f"=== {model}: rc={proc.returncode}, adapter={'ok' if adapter_completed(out_dir) else 'MISSING'} ==="
        )

    bench_rc = 0
    if not args.skip_benchmarks:
        adapters = [args.output_dir / r["model"] for r in results if r["adapter_ok"]]
        if adapters:
            logger.info(f"=== BENCHMARKS: {len(adapters)} adapter(s) ===")
            proc = subprocess.run(benchmark_command(adapters, args), cwd=REPO_ROOT, check=False)
            bench_rc = proc.returncode
            for result in results:
                if result["adapter_ok"]:
                    result["bench_rc"] = bench_rc
        else:
            bench_rc = 1
            logger.error("no adapters produced — benchmarks skipped")

    summary_path = args.report_dir / SUMMARY_NAME
    summary_path.write_text(render_summary(plan, results, args), encoding="utf-8")
    logger.info(f"summary written: {summary_path}")

    failed = [r["model"] for r in results if r["train_rc"] != 0 or not r["adapter_ok"]]
    if failed:
        logger.error(f"training issues for: {', '.join(failed)}")
    if not args.skip_benchmarks and bench_rc != 0:
        logger.error("benchmark phase reported failures")
    return 1 if failed or (not args.skip_benchmarks and bench_rc != 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
