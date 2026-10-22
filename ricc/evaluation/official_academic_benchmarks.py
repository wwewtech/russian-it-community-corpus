"""
Academic Benchmark Evaluation Harness for Russian IT LLM Ecosystem.
Evaluates models against representative task subsets:
1. OpenAI HumanEval (40 curated code execution tasks, pass@1)
2. Sber AI / HSE RuMMLU CS (50 Computer Science & Architecture QA questions)
3. Test Set Perplexity (PPL = exp(loss))
4. Text Overlap: ROUGE-1, ROUGE-2, ROUGE-L (via Hugging Face Evaluate)

NOTE: Prior published aggregate numbers from this harness were WITHDRAWN pending fresh
GPU re-evaluation due to earlier answer-parsing and column-mapping defects.
"""

import argparse
import hashlib
import json
import logging
import math
import re
import subprocess  # nosec B404
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ricc.bootstrap import setup_runtime_env
from ricc.evaluation.academic_benchmark_data import (
    _ANSWER_MARKER_PATTERN,  # noqa: F401
    _BRACKETED_LETTER_RE,
    _CYRILLIC_TO_LATIN,
    _LATIN_LETTER_RE,
    _MARKER_SEPARATOR_RE,
    _MARKER_STEMS,  # noqa: F401
    _MARKER_WHITESPACE_RE,
    _SAFE_BUILTINS,
    HUMANEVAL_TASKS,
    RUMMLU_CS_QUESTIONS,
)

setup_runtime_env(pytorch_alloc_conf=True)

import evaluate
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ricc.evaluation.statistical_power import mcnemar_test, wilson_interval
from ricc.ml_runtime import set_deterministic_seed
from ricc.observability import record_classified_ml_error
from ricc.rag.rag_pipeline import LocalRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AcademicBenchmark")

# ==========================================
# RETRACTION GATE
# ==========================================
# Published artifacts stay flagged RETRACTED until a reviewed re-run on the
# fixed harness. Flip BENCHMARK_STATUS to "VALID" and add `rerun_evidence`
# (commit sha + raw-output review notes) in matrix_res only after that review;
# tests/test_academic_benchmark_matrix_consistency.py enforces the gate.
BENCHMARK_STATUS = "RETRACTED"
RETRACTION_REASON = (
    "harness defects in the original run: substring MCQ scoring, Cyrillic homoglyph "
    "MCQ scoring (Russian prose such as 'в чём дело' scored as answer B), perplexity "
    "on blank generations, silent Base->LoRA copy when the adapter failed to load, "
    "and a HumanEval sandbox that was escapable and had no real containment"
)


def parse_mc_answer(response: str, valid_letters: str = "ABCD") -> str | None:
    """
    Strictly extract a standalone multiple-choice letter from a model response.

    The rule has two halves, because the Latin and the Cyrillic letter
    behave very differently in free-form Russian/English prose.

    LATIN letters (A-D) are accepted when they appear as a standalone
    UPPERCASE token:

        "A" -> "A"
        "Ответ: B" -> "B"
        "(C)" -> "C"
        "The answer is D" -> "D"

    Lowercase Latin is never an answer ("Compose a docker file" -> None),
    because "a" is the English article.

    CYRILLIC homoglyphs (А/В/С/Е) are accepted ONLY when the letter is an
    ISOLATED ANSWER TOKEN, which requires one of:

    1. BRACKETS: the letter is wrapped in round/square/curly brackets.
       The brackets alone are enough, no marker needed:

           "(в)" -> "B", "(с)" -> "C"

    2. MARKER + PUNCTUATION SEPARATOR: the letter directly follows an
       explicit answer marker (ответ / answer / вариант / option / choice)
       separated from it by ":", "=", "-", "–", "—" and/or spaces:

           "Ответ: в" -> "B"
           "Ответ - а" -> "A"
           "Answer: c" -> "C"

    3. MARKER + WHITESPACE, UPPERCASE LETTER: the marker is followed by
       whitespace and then an UPPERCASE letter. The uppercase requirement
       is what separates "Вариант В" (answer B) from "вариант с" ("the
       variant with", not an answer) — a lowercase Cyrillic homoglyph
       after a marker is still an ordinary Russian function word.

    4. MARKER + BRACKETS: the marker is followed by optional separator
       and then a bracketed letter:

           "Ответ (с)" -> "C"

    A bare Cyrillic homoglyph in prose is NEVER an answer. The previous
    implementation mapped the homoglyphs at ANY position, which is the same
    class of defect as the retracted substring bug (it inflated the RuMMLU
    accuracy with answers the model never gave):

        "Я не знаю в чём дело" -> None   ("в" is a Russian preposition)
        "Это вариант с"         -> None   ("с" is a Russian preposition)
        "а и б"                 -> None   ("а" is a Russian conjunction)

    Note that the marker must be IMMEDIATELY followed by the letter. The
    word "вариант" alone does not license the letter that follows it
    somewhere later in the sentence: "Это вариант с" is "this is the
    variant with", not "this is variant C".

    Returns the uppercase Latin letter, or None if no answer is present.
    """
    if not response:
        return None
    text = response.strip()

    def _normalize(raw: str) -> str | None:
        letter = raw.upper().translate(_CYRILLIC_TO_LATIN)
        return letter if letter in valid_letters else None

    # 1) Explicit marker immediately followed by the letter. This is the
    #    only place where a bare Cyrillic homoglyph counts.
    for pattern in (_MARKER_SEPARATOR_RE, _MARKER_WHITESPACE_RE):
        match = pattern.search(text)
        if match:
            letter = _normalize(match.group("letter"))
            if letter is not None:
                return letter

    # 2) A bracketed letter. The brackets make a Cyrillic homoglyph
    #    unambiguous, and this is also how "(C)" keeps working.
    match = _BRACKETED_LETTER_RE.search(text)
    if match:
        letter = _normalize(match.group("letter"))
        if letter is not None:
            return letter

    # 3) A standalone UPPERCASE LATIN letter anywhere. The uppercase
    #    requirement is what keeps "Compose a docker file" from being read
    #    as the answer "A" (the article "a" is lowercase).
    latin_match = _LATIN_LETTER_RE.search(text)
    if latin_match:
        letter = latin_match.group(1)
        if letter in valid_letters:
            return letter

    return None


def _run_in_isolated_process(code_str: str, timeout_sec: float = 2.0) -> bool:
    """Execute code_str inside an isolated child OS process.

    ⚠️ THIS IS NOT A SECURITY BOUNDARY. It is a *resource* isolation
    mechanism only: unlike threads, an isolated child process can be
    forcibly terminated (SIGKILL / TerminateProcess) when a timeout
    occurs, preventing infinite loops or runaway memory allocations
    from hanging or degrading the runner.

    The ``__builtins__`` allowlist below is trivially escapable. The
    classic escape reaches a live class whose ``__init__.__globals__``
    exposes ``os``:

        ().__class__.__base__.__subclasses__()[N].__init__.__globals__['os'].system('...')

    and the child then runs that command with the FULL privileges of the
    invoking user (no filesystem, network or uid separation). This was
    verified to actually write files on the host.

    Use ``_run_in_container`` (the default of ``execute_humaneval_code``)
    whenever the code is untrusted and containment matters.
    """
    runner_script = (
        "import builtins, sys\n"
        "safe_names = " + repr(sorted(list(_SAFE_BUILTINS))) + "\n"
        "safe_builtins = {k: getattr(builtins, k) for k in safe_names if hasattr(builtins, k)}\n"
        "code_to_run = sys.stdin.read()\n"
        "exec(code_to_run, {'__builtins__': safe_builtins}, {})\n"
    )

    try:
        # Fixed interpreter invocation, no shell, no untrusted input — model
        # code arrives via stdin into the sandbox.
        proc = subprocess.Popen(  # nosec B603
            [sys.executable, "-I", "-s", "-c", runner_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as err:
        logger.warning(f"Failed to spawn isolated execution process: {err}")
        return False

    try:
        proc.communicate(input=code_str, timeout=timeout_sec)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.communicate()
        except Exception:
            pass  # deliberate: process already terminating, nothing to report
        return False
    except Exception as err:
        try:
            proc.kill()
            proc.communicate()
        except Exception:
            pass  # deliberate: cleanup-of-cleanup; err is logged below
        logger.debug(f"Execution error in isolated process: {err}")
        return False


def _docker_available() -> bool:
    """Whether a usable Docker daemon is reachable."""
    try:
        proc = subprocess.run(  # nosec B603
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _docker_daemon_unreachable(stderr: str) -> bool:
    """Whether a failed ``docker run`` was caused by an absent daemon.

    The docker CLI can be installed while the daemon is stopped, in which
    case ``docker run`` exits non-zero with a connection error instead of
    raising ``OSError``. Such a run never executed the model code, so it
    must be reported as "container isolation unavailable" rather than as
    "the code failed".
    """
    lowered = stderr.lower()
    return any(
        needle in lowered
        for needle in (
            "cannot connect to the docker daemon",
            "docker daemon is not running",
            "error during connect",
            "the system cannot find the file specified",
            "open //./pipe/docker_engine",
        )
    )


class ContainerUnavailableError(RuntimeError):
    """Raised when container isolation is requested but Docker is unusable."""


def _run_in_container(code_str: str, timeout_sec: float = 2.0, image: str = "python:3.11-slim") -> bool:
    """Execute code_str inside a hardened container with no network and no host access.

    This is the only mode that provides a real security boundary. The
    process mode limits the code to an allowlisted ``__builtins__`` dict,
    but that is NOT a sandbox — ``().__class__.__base__.__subclasses__()``
    reaches a live class whose ``__init__.__globals__`` exposes ``os``,
    and the model code can then run arbitrary commands with the CI
    user's privileges.

    Containment measures applied to the container:

    * ``--network none`` — no network access at all.
    * ``--read-only`` + ``--tmpfs /tmp`` — the container filesystem is
      immutable except for a small in-memory ``/tmp``.
    * ``--user 65534:65534`` — runs as ``nobody``, not as the CI user and
      not as root.
    * ``--cap-drop ALL`` / ``--security-opt no-new-privileges`` — no
      Linux capabilities, no privilege escalation.
    * ``--pids-limit`` — no fork bombs.
    * ``--memory`` / ``--cpus`` — hard resource limits.
    * No host bind mounts — the host filesystem is invisible.

    Raises:
        ContainerUnavailableError: if the Docker daemon is missing or
            unusable. Callers MUST NOT silently treat that as a
            successful (contained) run.
    """
    runner_script = (
        "import builtins, sys\n"
        "safe_names = " + repr(sorted(list(_SAFE_BUILTINS))) + "\n"
        "safe_builtins = {k: getattr(builtins, k) for k in safe_names if hasattr(builtins, k)}\n"
        "code_to_run = sys.stdin.read()\n"
        "exec(code_to_run, {'__builtins__': safe_builtins}, {})\n"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory=256m",
        "--cpus=1.0",
        image,
        "python",
        "-I",
        "-s",
        "-c",
        runner_script,
    ]
    proc: subprocess.Popen[str] | None = None
    try:
        # Fixed docker invocation with pinned flags (--network none,
        # --read-only, --cap-drop ALL, memory/cpu limits); model code
        # arrives via stdin.
        proc = subprocess.Popen(  # nosec B603
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _stdout, stderr = proc.communicate(input=code_str, timeout=timeout_sec)
        if proc.returncode != 0 and _docker_daemon_unreachable(stderr):
            # The docker CLI exists but the daemon does not answer. The
            # model code never ran, so there is no containment question
            # here — but the caller must know that container isolation is
            # NOT available.
            raise ContainerUnavailableError("docker daemon is not reachable")
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                proc.kill()
                proc.communicate()
            except Exception:
                pass  # deliberate: process already terminating, nothing to report
        return False
    except (OSError, subprocess.SubprocessError):
        # Docker is unavailable or failed to start: the container mode
        # cannot provide containment. The caller decides how to degrade.
        logger.warning("Container isolation unavailable: %s", cmd[0])
        raise ContainerUnavailableError("docker is not available") from None


def execute_humaneval_code(
    generated_code: str,
    task: dict[str, Any],
    timeout_sec: float = 2.0,
    *,
    isolation_mode: str = "container",
    container_image: str = "python:3.11-slim",
) -> bool:
    """Execute generated Python code against standard test assertions.

    ``isolation_mode``:

    * ``"container"`` (DEFAULT) — the code runs inside a hardened Docker
      container with ``--network none``, a read-only filesystem, a
      non-root uid, no capabilities and hard memory/CPU limits. This is
      the only mode that provides a real containment boundary, and it is
      the default precisely because the model-generated code is
      untrusted.

    * ``"process"`` — the code runs in a child Python process with an
      allowlisted ``__builtins__`` dict and a timeout. ⚠️ THIS IS NOT A
      SECURITY SANDBOX. The allowlist is trivially escapable via
      ``().__class__.__base__.__subclasses__()`` (a live class's
      ``__init__.__globals__`` exposes ``os``), so model code can run
      arbitrary commands with the caller's full privileges — including
      writing to the host filesystem and reaching the network. It is kept
      as an EXPLICIT opt-in for trusted code only, and for environments
      without Docker where the operator has accepted that risk.

    Degradation when Docker is unavailable:

    The SAFE behaviour is to REFUSE the execution and report the task as
    failed (``False``). Falling back to the process mode would silently
    run untrusted model code with the host user's privileges, which is
    exactly the vulnerability this default is meant to close. The
    container mode therefore never degrades into the process mode on its
    own: it returns ``False`` and logs a loud warning. To run without
    Docker you must opt in explicitly with ``isolation_mode="process"``,
    which is a deliberate operator decision, not a silent fallback.

    Returns:
        True only if the code ran AND the tests passed. ``False`` means
        "did not pass" (assertion error, timeout, crash) or "was not
        run at all" (no container runtime available).
    """
    code_match = re.search(r"```(?:python|py)?\n(.*?)```", generated_code, re.DOTALL)
    code_to_exec = code_match.group(1).strip() if code_match else generated_code.strip()

    if task["entry_point"] not in code_to_exec:
        code_to_exec = task["prompt"] + "\n" + code_to_exec

    full_program = f"{code_to_exec}\n\n{task['test']}"

    if isolation_mode == "container":
        try:
            return _run_in_container(full_program, timeout_sec=timeout_sec, image=container_image)
        except ContainerUnavailableError:
            # SAFE degradation: do NOT fall back to the process mode.
            # The process mode is escapable, so running untrusted model
            # code there would re-open the host-filesystem hole this
            # default is meant to close. Refuse the execution instead
            # and make the reason impossible to miss.
            logger.error(
                "CONTAINER ISOLATION UNAVAILABLE — refusing to execute the model code. "
                "Falling back to the process mode is NOT safe (the __subclasses__ escape "
                "gives the model code the host user's full privileges). "
                "Install/start Docker, or call with isolation_mode='process' to accept "
                "the risk explicitly. The task is reported as FAILED."
            )
            return False
    if isolation_mode == "process":
        return _run_in_isolated_process(full_program, timeout_sec=timeout_sec)
    raise ValueError(f"Unknown isolation_mode: {isolation_mode!r} (expected 'container' or 'process')")


def _paired_significance(
    outcomes: dict[str, dict[str, bool]],
    comparisons: tuple[tuple[str, str], ...] = (("base", "lora"), ("base", "hybrid"), ("lora", "hybrid")),
) -> dict[str, dict[str, object]]:
    """McNemar's exact test for every variant pair.

    ``outcomes`` maps a variant name to ``{task_key: passed}``. The
    variants are evaluated on the same tasks, so the comparison is
    paired and McNemar's test is the correct one — comparing two
    independent proportions (or eyeballing overlapping Wilson
    intervals) cannot answer "is B better than A?".

    A comparison whose p-value is above the threshold is reported as
    NOT significant, so the report cannot claim an improvement that the
    data does not support.
    """
    results: dict[str, dict[str, object]] = {}
    for variant_a, variant_b in comparisons:
        a = outcomes.get(variant_a, {})
        b = outcomes.get(variant_b, {})
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        pairs = [(a[key], b[key]) for key in shared]
        result = mcnemar_test(pairs, confidence=0.95)
        result["variant_a"] = variant_a
        result["variant_b"] = variant_b
        results[f"{variant_a}_vs_{variant_b}"] = result
    return results


def run_official_academic_benchmarks(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_id: str = "qwen2.5_1.5b_instruct",
    seed: int = 42,
    isolation_mode: str = "container",
) -> dict[str, Any]:
    """Run the academic benchmark suite.

    ``isolation_mode`` is forwarded to ``execute_humaneval_code`` and
    defaults to ``"container"`` — the only mode that actually contains
    the untrusted model-generated code. With ``"process"`` the HumanEval
    tasks are executed WITHOUT a security boundary (see
    ``execute_humaneval_code``); that opt-in must be deliberate.
    """
    logger.info(f"=== Running Official Academic Scientific Benchmarks for {model_name} ===")
    set_deterministic_seed(seed)
    if isolation_mode != "container" and not _docker_available():
        logger.warning(
            "HumanEval tasks will run WITHOUT container isolation and WITHOUT Docker "
            "available: the model-generated code is NOT contained and can reach the "
            "host filesystem and network."
        )

    rouge = evaluate.load("rouge")
    rag_kb = LocalRAGPipeline(Path("dataset_output/parquet/rag_knowledge_base.parquet"))

    tokenizer: Any = cast(Any, AutoTokenizer).from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model: Any = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Attach LoRA
    adapter_path = Path(f"lora_adapters/{adapter_id}")
    lora_model: Any = None
    if adapter_path.exists() and (adapter_path / "adapter_model.safetensors").exists():
        try:
            lora_model = cast(Any, PeftModel).from_pretrained(base_model, str(adapter_path))
            logger.info(f"Attached LoRA Adapter from {adapter_path}")
        except Exception as e:
            typed = record_classified_ml_error(e, model_name=model_name, adapter_id=adapter_id)
            logger.warning(f"Could not load LoRA: {typed}")

    if lora_model is None:
        raise RuntimeError(
            f"LoRA adapter '{adapter_id}' could not be loaded from '{adapter_path}'. "
            "Refusing to run comparative benchmarks: previously this branch silently copied "
            "base-model results into the LoRA/hybrid columns, producing identical fake metrics."
        )

    def generate_fn(model: Any, prompt_str: str, max_tokens: int = 256) -> str:
        messages = [{"role": "user", "content": prompt_str}]
        try:
            inp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            inp = prompt_str

        inputs = tokenizer(inp, return_tensors="pt", max_length=512, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,  # Standard low temperature for academic coding & QA benchmarks
                do_sample=False,  # Greedy decoding for deterministic benchmark reproduction
                pad_token_id=tokenizer.pad_token_id,
            )
        text: str = cast(str, tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text.strip()

    def base_generate(prompt_str: str, max_tokens: int = 256) -> str:
        """Generate with the LoRA adapter DISABLED -> true base-model behaviour.

        Fixes the in-place-injection bug: ``PeftModel.from_pretrained(base, ...)``
        wraps the *same* model object, so without explicitly disabling the adapter
        the \"Base\" column silently measured the adapter-active model (identical to
        LoRA byte-for-byte). See _tmp_probe2.py / _probe2_results.json.
        """
        with lora_model.disable_adapter():
            return generate_fn(lora_model, prompt_str, max_tokens=max_tokens)

    raw_outputs_path = Path("reports/academic_benchmarks_raw_outputs.jsonl")
    raw_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_outputs_path.exists():
        raw_outputs_path.unlink()

    def record_raw_output(item: dict[str, Any]) -> None:
        with open(raw_outputs_path, "a", encoding="utf-8") as f_raw:
            f_raw.write(json.dumps(item, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------
    # 1. EVALUATION ON OPENAI HUMANEVAL (pass@1 Deterministic Code Execution)
    # -------------------------------------------------------------
    logger.info("Running Benchmark 1: OpenAI HumanEval Code Execution (pass@1)...")
    humaneval_results = {"base": 0, "rag": 0, "lora": 0, "hybrid": 0, "total": len(HUMANEVAL_TASKS)}
    task_exec_records = []
    # Per-task outcomes, so the variants can be compared with a PAIRED
    # test (McNemar) instead of two independent proportions. The variants
    # run on the same tasks, so the pairing is exact.
    humaneval_outcomes: dict[str, dict[str, bool]] = {"base": {}, "rag": {}, "lora": {}, "hybrid": {}}

    for task in HUMANEVAL_TASKS:
        # Base (adapter disabled)
        base_code = base_generate(task["prompt"], max_tokens=150)
        b_ok = execute_humaneval_code(base_code, task, isolation_mode=isolation_mode)
        humaneval_outcomes["base"][task["task_id"]] = b_ok
        if b_ok:
            humaneval_results["base"] += 1
        record_raw_output(
            {
                "benchmark": "HumanEval",
                "task_id": task["task_id"],
                "entry_point": task["entry_point"],
                "variant": "base",
                "prompt": task["prompt"],
                "raw_output": base_code,
                "passed": b_ok,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

        # RAG (adapter disabled, base + context)
        rag_hits = rag_kb.search(task["prompt"], top_k=1)
        rag_ctx = rag_hits[0].get("content", "")[:150] if rag_hits else ""
        rag_code = base_generate(f"Reference code:\n{rag_ctx}\n\nTask:\n{task['prompt']}", max_tokens=150)
        r_ok = execute_humaneval_code(rag_code, task, isolation_mode=isolation_mode)
        humaneval_outcomes["rag"][task["task_id"]] = r_ok
        if r_ok:
            humaneval_results["rag"] += 1
        record_raw_output(
            {
                "benchmark": "HumanEval",
                "task_id": task["task_id"],
                "entry_point": task["entry_point"],
                "variant": "rag",
                "prompt": f"Reference code:\n{rag_ctx}\n\nTask:\n{task['prompt']}",
                "raw_output": rag_code,
                "passed": r_ok,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

        # LoRA & Hybrid (lora_model is guaranteed non-None by the fail-fast check above)
        lora_code = generate_fn(lora_model, task["prompt"], max_tokens=150)
        l_ok = execute_humaneval_code(lora_code, task, isolation_mode=isolation_mode)
        humaneval_outcomes["lora"][task["task_id"]] = l_ok
        if l_ok:
            humaneval_results["lora"] += 1
        record_raw_output(
            {
                "benchmark": "HumanEval",
                "task_id": task["task_id"],
                "entry_point": task["entry_point"],
                "variant": "lora",
                "prompt": task["prompt"],
                "raw_output": lora_code,
                "passed": l_ok,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

        hyb_code = generate_fn(lora_model, f"Reference code:\n{rag_ctx}\n\nTask:\n{task['prompt']}", max_tokens=150)
        h_ok = execute_humaneval_code(hyb_code, task, isolation_mode=isolation_mode)
        humaneval_outcomes["hybrid"][task["task_id"]] = h_ok
        if h_ok:
            humaneval_results["hybrid"] += 1
        record_raw_output(
            {
                "benchmark": "HumanEval",
                "task_id": task["task_id"],
                "entry_point": task["entry_point"],
                "variant": "hybrid",
                "prompt": f"Reference code:\n{rag_ctx}\n\nTask:\n{task['prompt']}",
                "raw_output": hyb_code,
                "passed": h_ok,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

        task_exec_records.append(
            {
                "task_id": task["task_id"],
                "entry_point": task["entry_point"],
                "base_ok": b_ok,
                "lora_ok": l_ok,
                "hybrid_ok": h_ok,
            }
        )

    pass_at_1 = {k: round((v / len(HUMANEVAL_TASKS)) * 100.0, 1) for k, v in humaneval_results.items() if k != "total"}
    humaneval_ci = {
        k: wilson_interval(humaneval_results[k], len(HUMANEVAL_TASKS), 0.95) for k in ("base", "rag", "lora", "hybrid")
    }
    # Paired significance (McNemar) — see _paired_significance.
    humaneval_significance = _paired_significance(humaneval_outcomes)

    # -------------------------------------------------------------
    # 2. EVALUATION ON RUMMLU CS (Exact Multiple-Choice Accuracy)
    # -------------------------------------------------------------
    logger.info("Running Benchmark 2: Sber AI / HSE RuMMLU CS Accuracy...")
    rummlu_results = {"base": 0, "rag": 0, "lora": 0, "hybrid": 0, "total": len(RUMMLU_CS_QUESTIONS)}
    # Per-question outcomes, so the variants can be compared with a PAIRED
    # test (McNemar) instead of two independent proportions. The variants
    # run on the same questions, so the pairing is exact.
    rummlu_outcomes: dict[str, dict[str, bool]] = {"base": {}, "rag": {}, "lora": {}, "hybrid": {}}

    for q in RUMMLU_CS_QUESTIONS:
        prompt_q = (
            f"Вопрос: {q['question']}\nВарианты ответа:\n"
            + "\n".join(q["options"])
            + "\nУкажи только одну букву правильного ответа (A, B, C или D):"
        )

        # Base (adapter disabled)
        b_ans = base_generate(prompt_q, max_tokens=10)
        b_ok = parse_mc_answer(b_ans) == q["answer"]
        rummlu_outcomes["base"][q["question"]] = b_ok
        if b_ok:
            rummlu_results["base"] += 1
        record_raw_output(
            {
                "benchmark": "RuMMLU_CS",
                "question": q["question"],
                "variant": "base",
                "prompt": prompt_q,
                "raw_output": b_ans,
                "parsed_answer": parse_mc_answer(b_ans),
                "expected_answer": q["answer"],
                "passed": b_ok,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

        # RAG (adapter disabled, base + context)
        rag_hits = rag_kb.search(q["question"], top_k=1)
        rag_ctx = rag_hits[0].get("content", "")[:200] if rag_hits else ""
        rag_prompt = f"Контекст:\n{rag_ctx}\n\n{prompt_q}"
        r_ans = base_generate(rag_prompt, max_tokens=10)
        r_ok = parse_mc_answer(r_ans) == q["answer"]
        rummlu_outcomes["rag"][q["question"]] = r_ok
        if r_ok:
            rummlu_results["rag"] += 1
        record_raw_output(
            {
                "benchmark": "RuMMLU_CS",
                "question": q["question"],
                "variant": "rag",
                "prompt": rag_prompt,
                "raw_output": r_ans,
                "parsed_answer": parse_mc_answer(r_ans),
                "expected_answer": q["answer"],
                "passed": r_ok,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

        # LoRA & Hybrid
        l_ans = generate_fn(lora_model, prompt_q, max_tokens=10)
        l_ok = parse_mc_answer(l_ans) == q["answer"]
        rummlu_outcomes["lora"][q["question"]] = l_ok
        if l_ok:
            rummlu_results["lora"] += 1
        record_raw_output(
            {
                "benchmark": "RuMMLU_CS",
                "question": q["question"],
                "variant": "lora",
                "prompt": prompt_q,
                "raw_output": l_ans,
                "parsed_answer": parse_mc_answer(l_ans),
                "expected_answer": q["answer"],
                "passed": l_ok,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

        h_ans = generate_fn(lora_model, rag_prompt, max_tokens=10)
        h_ok = parse_mc_answer(h_ans) == q["answer"]
        rummlu_outcomes["hybrid"][q["question"]] = h_ok
        if h_ok:
            rummlu_results["hybrid"] += 1
        record_raw_output(
            {
                "benchmark": "RuMMLU_CS",
                "question": q["question"],
                "variant": "hybrid",
                "prompt": rag_prompt,
                "raw_output": h_ans,
                "parsed_answer": parse_mc_answer(h_ans),
                "expected_answer": q["answer"],
                "passed": h_ok,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    rummlu_acc = {
        k: round((v / len(RUMMLU_CS_QUESTIONS)) * 100.0, 1) for k, v in rummlu_results.items() if k != "total"
    }
    rummlu_ci = {
        k: wilson_interval(rummlu_results[k], len(RUMMLU_CS_QUESTIONS), 0.95) for k in ("base", "rag", "lora", "hybrid")
    }

    # Paired significance (McNemar). Every variant runs on the SAME
    # questions, so the outcomes are paired and McNemar's exact test is
    # the right tool. The report previously asserted an improvement
    # whose confidence intervals OVERLAP; on the published raw outputs
    # no comparison reaches significance (p >= 0.125).
    rummlu_significance = _paired_significance(rummlu_outcomes)

    # -------------------------------------------------------------
    # 3. MATHEMATICAL INFORMATION-THEORETIC PERPLEXITY (PPL)
    # -------------------------------------------------------------
    logger.info("Running Benchmark 3: Mathematical Perplexity (PPL) on Held-Out Test Set...")
    test_df = pd.read_parquet("dataset_output/parquet/sft_dialogues.parquet").sample(n=50, random_state=42)
    # NOTE: the SFT parquet schema stores dialogues in a `messages` column of
    # role/content dicts. The previous implementation read non-existent
    # `query`/`response` columns and computed PPL on the constant string
    # "None None", which made base and LoRA perplexities identical.
    test_texts = []
    for _, row in test_df.iterrows():
        turns = row.get("messages")
        if turns is None:
            continue
        text = " ".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in turns if isinstance(t, dict))
        if text.strip():
            test_texts.append(text)
        if len(test_texts) >= 30:
            break

    def compute_ppl(model_to_eval: Any) -> float:
        nlls = []
        for text in test_texts:
            enc = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
            if torch.cuda.is_available():
                enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.no_grad():
                outputs = model_to_eval(**enc, labels=enc["input_ids"])
                neg_log_likelihood = outputs.loss
                if not torch.isnan(neg_log_likelihood):
                    nlls.append(neg_log_likelihood.item())
        mean_nll = float(np.mean(nlls)) if nlls else 2.5
        return round(float(math.exp(mean_nll)), 2)

    def compute_base_ppl() -> float:
        """PPL on the true base model (adapter disabled, see base_generate)."""
        with lora_model.disable_adapter():
            return compute_ppl(lora_model)

    base_ppl = compute_base_ppl()
    lora_ppl = compute_ppl(lora_model) if lora_model else base_ppl
    record_raw_output(
        {
            "benchmark": "Perplexity",
            "variant": "base",
            "ppl": base_ppl,
            "sample_size": len(test_texts),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    record_raw_output(
        {
            "benchmark": "Perplexity",
            "variant": "lora",
            "ppl": lora_ppl,
            "sample_size": len(test_texts),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )

    # -------------------------------------------------------------
    # 4. ROUGE ACADEMIC TEXT SIMILARITY
    # -------------------------------------------------------------
    logger.info("Running Benchmark 4: ROUGE-1/2/L Evaluation...")
    ref_answers = [
        "Для решения проблемы рассинхронизации iptables в Kubernetes настраивается preStop хук со sleep 15 и readinessProbe для плавного завершения соединений без 502 Bad Gateway.",
        "Паттерн Transactional Outbox решает проблему распределенной транзакции путем записи события в локальную таблицу outbox в рамках одной ACID транзакции с последующим чтением через Debezium CDC в Kafka.",
    ]
    eval_prompts = [
        "Как устранить 502 Bad Gateway при rolling update в Kubernetes?",
        "Как устроен Transactional Outbox Pattern в PostgreSQL и Kafka?",
    ]

    base_preds = [base_generate(p, max_tokens=80) for p in eval_prompts]
    lora_preds = [generate_fn(lora_model, p, max_tokens=80) for p in eval_prompts]

    base_rouge = rouge.compute(predictions=base_preds, references=ref_answers)
    lora_rouge = rouge.compute(predictions=lora_preds, references=ref_answers)
    record_raw_output(
        {
            "benchmark": "ROUGE",
            "variant": "base",
            "scores": base_rouge,
            "prompts": eval_prompts,
            "predictions": base_preds,
            "references": ref_answers,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    record_raw_output(
        {
            "benchmark": "ROUGE",
            "variant": "lora",
            "scores": lora_rouge,
            "prompts": eval_prompts,
            "predictions": lora_preds,
            "references": ref_answers,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )

    # -------------------------------------------------------------
    # 5. GENERATE SCIENTIFIC REPORT
    # -------------------------------------------------------------
    output_md = Path("reports/OFFICIAL_ACADEMIC_SCIENTIFIC_BENCHMARKS.md")
    report_lines = [
        "# Отчет об академической оценке (HumanEval Sample, RuMMLU Sample, PPL)",
        f"**Оценочная модель:** `{model_name}` | **LoRA Адаптер:** `{adapter_id}` | **GPU:** `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}`",
        f"**Дата проведения:** `{time.strftime('%Y-%m-%dT%H:%M:%S')}`",
        "",
        "---",
        "",
        "## 1. Методология измерений (Methodology)",
        "",
        "Данный скрипт выполняет детерминированную проверку на контрольной выборке:",
        "",
        f"1. **OpenAI HumanEval subset ({len(HUMANEVAL_TASKS)} задач)**: Сгенерированный код запускается в изолированном интерпретаторе Python с набором unit-тестов. $\\text{{pass@1}} = \\frac{{N_{{\\text{{passed}}}}}}{{N_{{\\text{{total}}}}}} \\times 100\\%$.",
        f"2. **RuMMLU CS subset ({len(RUMMLU_CS_QUESTIONS)} вопросов)**: Выборка по направлениям Databases, Networking, Algorithms, OS, Distributed Systems, Security, Programming Languages. Балл — процент правильных ответов (Accuracy).",
        f"   Выборки расширены с 8 до {len(HUMANEVAL_TASKS)} задач / {len(RUMMLU_CS_QUESTIONS)} вопросов для статистической силы: "
        f"при N=8 доверительный интервал Вильсона (95%) достигает ±20 п.п., при N={len(RUMMLU_CS_QUESTIONS)} — уже "
        f"±{round((rummlu_ci['hybrid'][1] - rummlu_ci['hybrid'][0]) / 2 * 100, 1)} п.п. Все публикуемые точности сопровождаются интервалами.",
        "3. **Информационно-теоретическая перплексия (PPL)**: $\\text{PPL} = \\exp\\left(-\\frac{1}{T}\\sum_{t=1}^T \\ln P(w_t \\mid w_{<t})\\right)$ на отложенной тестовой выборке диалогов.",
        "4. **ROUGE-1 / ROUGE-L**: Оценка лексического перекрытия с эталонными ответами через библиотеку `evaluate`.",
        "",
        "---",
        "",
        "## 2. Сводные результаты",
        "",
        "| Бенчмарк / Метрика | Метрика | Базовая модель (Base) | Базовая + RAG | Domain LoRA | Гибрид (LoRA + RAG) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **HumanEval Subset ({len(HUMANEVAL_TASKS)} задач)** | `pass@1 (%)` | **{pass_at_1['base']}%** | **{pass_at_1['rag']}%** | **{pass_at_1['lora']}%** | **{pass_at_1['hybrid']}%** |",
        f"| **RuMMLU CS Subset ({len(RUMMLU_CS_QUESTIONS)} вопр.)** | `Accuracy (%)` | **{rummlu_acc['base']}%** | **{rummlu_acc['rag']}%** | **{rummlu_acc['lora']}%** | **{rummlu_acc['hybrid']}%** |",
        f"| **HumanEval pass@1 — 95% Wilson CI** | `интервал` | {humaneval_ci['base'][0] * 100:.1f}–{humaneval_ci['base'][1] * 100:.1f}% | {humaneval_ci['rag'][0] * 100:.1f}–{humaneval_ci['rag'][1] * 100:.1f}% | {humaneval_ci['lora'][0] * 100:.1f}–{humaneval_ci['lora'][1] * 100:.1f}% | {humaneval_ci['hybrid'][0] * 100:.1f}–{humaneval_ci['hybrid'][1] * 100:.1f}% |",
        f"| **RuMMLU accuracy — 95% Wilson CI** | `интервал` | {rummlu_ci['base'][0] * 100:.1f}–{rummlu_ci['base'][1] * 100:.1f}% | {rummlu_ci['rag'][0] * 100:.1f}–{rummlu_ci['rag'][1] * 100:.1f}% | {rummlu_ci['lora'][0] * 100:.1f}–{rummlu_ci['lora'][1] * 100:.1f}% | {rummlu_ci['hybrid'][0] * 100:.1f}–{rummlu_ci['hybrid'][1] * 100:.1f}% |",
        # The Hybrid column is NOT a copy of the LoRA column: hybrid PPL
        # and ROUGE were never measured (the RAG context is prepended to
        # the prompt, which changes what a perplexity over the answer
        # means). Printing the LoRA value in the Hybrid column made the
        # table claim a measurement that does not exist.
        f"| **Test Set Perplexity** | `PPL (ниже = лучше)` | `{base_ppl}` | N/A | **`{lora_ppl}`** | `н/д` |",
        f"| **ROUGE-1 F1** | `Overlap (%)` | `{round(base_rouge['rouge1'] * 100, 1)}%` | N/A | **`{round(lora_rouge['rouge1'] * 100, 1)}%`** | `н/д` |",
        f"| **ROUGE-L F1** | `LCS Overlap (%)` | `{round(base_rouge['rougeL'] * 100, 1)}%` | N/A | **`{round(lora_rouge['rougeL'] * 100, 1)}%`** | `н/д` |",
        "",
        "### Значимость различий (парный тест Мак-Немера, 95%)",
        "",
        "Все варианты прогоняются на ОДНИХ И ТЕХ ЖЕ задачах, поэтому сравнение парное:",
        "используется точный тест Мак-Немера по дискордантным парам, а не сравнение двух",
        "независимых долей. Перекрывающиеся доверительные интервалы Уилсона — только грубый",
        "фильтр; вывод делается по p-значению.",
        "",
        "| Бенчмарк | Сравнение | Δ (п.п.) | p-значение | Вывод |",
        "| :--- | :--- | ---: | ---: | :--- |",
    ]

    for bench_name, significance in (
        ("HumanEval", humaneval_significance),
        ("RuMMLU_CS", rummlu_significance),
    ):
        for res in significance.values():
            delta = float(cast(float, res["difference_in_proportion"])) * 100.0
            verdict = "значимо" if res["significant"] else "**не значимо**"
            report_lines.append(
                f"| {bench_name} | {res['variant_a']} vs {res['variant_b']} | {delta:+.1f} | "
                f"{float(cast(float, res['p_value'])):.4f} | {verdict} |"
            )

    report_lines.extend(
        [
            "",
            "> Если p-значение выше 0.05, различие нельзя объявлять улучшением: на этой",
            "> выборке варианты статистически неразличимы.",
            "",
            "---",
            "",
            "## 3. Детальный разбор выполнения HumanEval subset",
            "",
            "| Задача HumanEval | Сигнатура функции | Unit-тесты Base | Unit-тесты LoRA | Unit-тесты Hybrid |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ]
    )

    if BENCHMARK_STATUS == "RETRACTED":
        report_lines[1:1] = [
            "",
            "> [!WARNING]",
            f"> **RETRACTED — do not cite.** {RETRACTION_REASON}.",
            "> A fresh, reviewed GPU re-run is required before any accuracy claims.",
        ]

    for rec in task_exec_records:
        b_str = "PASSED" if rec["base_ok"] else "FAILED"
        l_str = "PASSED" if rec["lora_ok"] else "FAILED"
        h_str = "PASSED" if rec["hybrid_ok"] else "FAILED"
        report_lines.append(f"| `{rec['task_id']}` | `{rec['entry_point']}` | {b_str} | {l_str} | {h_str} |")

    # The conclusions must follow the data. Every comparison in the
    # published run is NOT significant (p >= 0.125), so the report may
    # not claim that one variant beats another; it states the measured
    # values and says the variants are indistinguishable.
    significant_comparisons = [
        f"{res['variant_a']} vs {res['variant_b']} ({bench}, p={float(cast(float, res['p_value'])):.3f})"
        for bench, significance in (
            ("HumanEval", humaneval_significance),
            ("RuMMLU_CS", rummlu_significance),
        )
        for res in significance.values()
        if res["significant"]
    ]
    if significant_comparisons:
        significance_line = (
            "Значимые различия (парный тест Мак-Немера, p < 0.05): " + ", ".join(significant_comparisons) + "."
        )
    else:
        significance_line = (
            "**Ни одно сравнение вариантов не является статистически значимым** "
            "(парный тест Мак-Немера, p >= 0.05): при такой выборке варианты статистически "
            "неразличимы, поэтому утверждать превосходство одного над другим нельзя."
        )

    report_lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Выводы",
            "",
            f"1. **Перплексия на доменном тесте (PPL base={base_ppl}, LoRA={lora_ppl})**: измерено "
            f"на отложенной выборке; гибридный PPL не измерялся и не приводится.",
            f"2. **Кодогенерация HumanEval (pass@1: base={pass_at_1['base']}%, rag={pass_at_1['rag']}%, "
            f"LoRA={pass_at_1['lora']}%, hybrid={pass_at_1['hybrid']}%)**: проверка работоспособности "
            f"сгенерированных Python-функций на тестовых ассертах.",
            f"3. **RuMMLU (accuracy: base={rummlu_acc['base']}%, rag={rummlu_acc['rag']}%, "
            f"LoRA={rummlu_acc['lora']}%, hybrid={rummlu_acc['hybrid']}%)**: точность выбора "
            f"вариантов ответа на контрольных вопросах по архитектуре БД, сетей и ОС.",
            f"4. **{significance_line}**",
        ]
    )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    matrix_res = {
        "humaneval_pass_at_1": pass_at_1,
        "humaneval_pass_at_1_ci95": {k: [round(v[0] * 100, 1), round(v[1] * 100, 1)] for k, v in humaneval_ci.items()},
        "rummlu_accuracy": rummlu_acc,
        "rummlu_accuracy_ci95": {k: [round(v[0] * 100, 1), round(v[1] * 100, 1)] for k, v in rummlu_ci.items()},
        # Paired significance: the variants run on the same tasks, so
        # McNemar's exact test is the right comparison. The published
        # run has NO significant difference (p >= 0.125) — the point
        # estimates alone must not be read as an improvement.
        "significance": {
            "method": "McNemar exact test on paired per-task outcomes, two-sided, 95%",
            "humaneval": humaneval_significance,
            "rummlu": rummlu_significance,
            "any_significant": bool(
                [r for s in (humaneval_significance, rummlu_significance) for r in s.values() if r["significant"]]
            ),
        },
        "sample_sizes": {
            "humaneval_tasks": len(HUMANEVAL_TASKS),
            "rummlu_questions": len(RUMMLU_CS_QUESTIONS),
        },
        "perplexity": {"base": base_ppl, "lora": lora_ppl, "hybrid": None},
        "rouge": {"base": base_rouge, "lora": lora_rouge, "hybrid": None},
        "not_measured": {
            "hybrid_perplexity": "the RAG context is prepended to the prompt, so a perplexity "
            "over the answer is not the same quantity for the hybrid variant; it is not reported "
            "rather than copied from the LoRA column",
            "hybrid_rouge": "same reason; not measured and not reported",
        },
        # Provenance: every number above must be recomputable from the raw
        # generation logs; the sha256 pins the exact log this matrix was
        # derived from (verified by tests/test_academic_benchmark_matrix_consistency.py).
        "raw_outputs_sha256": hashlib.sha256(raw_outputs_path.read_bytes()).hexdigest(),
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_name": model_name,
        "adapter_id": adapter_id,
        "status": BENCHMARK_STATUS,
        "retraction_reason": RETRACTION_REASON if BENCHMARK_STATUS == "RETRACTED" else None,
    }
    output_json = Path("reports/academic_scientific_benchmarks_matrix.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(matrix_res, f, ensure_ascii=False, indent=2)

    logger.info(f"Academic Benchmark evaluation finished! Report written to {output_md}")
    return matrix_res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default="qwen2.5_1.5b_instruct")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for reproducible runs")
    parser.add_argument(
        "--isolation-mode",
        type=str,
        default="container",
        choices=("container", "process"),
        help=(
            "HumanEval execution isolation. 'container' (default) contains the "
            "untrusted model code; 'process' does NOT and is only for trusted code."
        ),
    )
    args = parser.parse_args()

    run_official_academic_benchmarks(
        model_name=args.model,
        adapter_id=args.adapter,
        seed=args.seed,
        isolation_mode=args.isolation_mode,
    )


if __name__ == "__main__":
    main()
