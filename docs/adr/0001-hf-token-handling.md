# ADR 0001 — Hugging Face Token Handling

- **Status:** Accepted
- **Date:** 2026-08-29
- **Deciders:** @wwewtech

## Context

The project publishes its LoRA adapter zoo and dataset releases to the
Hugging Face Hub under `wwewtech/russian-it-community-lora` and
`wwewtech/russian-it-community-corpus`. Mirroring the local `registry.json`,
dataset cards, and parquet outputs to those repositories requires a write
token.

Two natural ways to feed that token into automation:

1. Pass it through a chat session or paste it into the repo.
2. Read it from an environment variable or GitHub Actions secret.

Option (1) is unsafe: the token ends up in the chat provider's logs, in
the agent's `environment_details` cache, in shell history, and — if
accidentally committed — in git history. Even one rotation per incident
is not free: HF tokens gate `wwewtech/*` namespaces, so a leak requires
manual cleanup of every dataset/model card on the org.

## Decision

**The HF write token is never embedded in source code, never pasted into
chat, and never passed as a CLI flag to scripts that ship to disk.**

All scripts and CI workflows read the token from one of:

* `HF_TOKEN` environment variable (exported by the developer locally or
  injected by GitHub Actions from a secret).
* `~/.huggingface/token` (the standard `huggingface-cli login` location).
* GitHub Actions secret `HF_TOKEN` at the org or repo level.

Scripts MUST call `huggingface_hub.HfApi(token=os.environ["HF_TOKEN"])`
or — preferably — rely on `huggingface_hub`'s built-in environment
auto-detection (`HfApi()` with no argument).

## Consequences

* If someone pastes a token into chat, the correct response is to revoke
  the token at https://huggingface.co/settings/tokens and rotate. The
  pasted value must never be wired into a script by an agent.
* A new `scripts/sync_to_hub.py` is provided that wraps the safe path:
  it reads the token from the environment, refuses to run if `HF_TOKEN`
  is unset, and never logs the token. See
  [`../../scripts/sync_to_hub.py`](../../scripts/sync_to_hub.py).
* The local `lora_adapters/registry.json` and `lora_adapters/SUMMARY.md`
  remain the **single source of truth** for the adapter zoo. HF Hub is
  treated as a derived artifact; if the two ever disagree, the local
  files win until the next sync.
* CI does **not** push to HF Hub automatically. Any HF push is an
  explicit, reviewed action: a maintainer runs the script locally with
  a properly-sourced `HF_TOKEN`, or triggers a dedicated `release-hf`
  workflow that consumes the org-level `HF_TOKEN` secret.

## Alternatives considered

* **Embed the token in a `.env` file committed to the repo.** Rejected:
  the same exposure as pasting into chat, plus accidental commits.
* **Per-developer token with least-privilege scopes.** Rejected for now:
  the project is single-maintainer; a single org-level token with
  write access to `wwewtech/*` is acceptable. Revisit if a second
  maintainer joins.
* **No HF sync at all.** Rejected: the existing `wwewtech/russian-it-community-lora`
  repo is a real distribution channel for users; cutting it off would
  degrade the project's value.

## References

* HF token management: https://huggingface.co/docs/hub/security-tokens
* `huggingface_hub` auth docs: https://huggingface.co/docs/huggingface_hub/authentication
