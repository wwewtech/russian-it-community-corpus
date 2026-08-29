# ADR 0001: Hugging Face Token Handling

**Status:** Accepted
**Date:** 2026-08-29
**Authors:** RICC maintainers
**Supersedes:** —

## Context

The `lora_adapters/` directory is mirrored to `wwewtech/russian-it-community-lora`
on the Hugging Face Hub via `scripts/sync_to_hub.py`. The Hub requires an
authentication token, which is a long-lived secret with write access to that
repo.

During the 2026-08-29 audit we discovered that a real HF token was pasted
into chat, and the conversation log was then summarized into a new session
context. Tokens that travel through chat are considered compromised: they
end up in scrollback, screen-shares, exported transcripts, and downstream
context windows, and once shipped there is no reliable way to scrub them.

## Decision

The HF token **MUST NOT** appear in:

- Source code, comments, or docstrings
- CLI flags or positional arguments
- Chat, issues, pull requests, or commit messages
- Documentation examples
- Test fixtures

The token **MUST** be supplied through one of the following channels only:

1. The `HF_TOKEN` environment variable (preferred for CI and one-off runs).
2. The `~/.huggingface/token` cache file (preferred for interactive use,
   written by `huggingface-cli login`).

`scripts/sync_to_hub.py` is the only sanctioned upload path and enforces
these rules:

- Reads `HF_TOKEN` from the environment first, then falls back to
  `HfFolder.get_token()`.
- Exits with code 3 if neither source provides a token.
- Never prints the token value, even when `--apply` is passed.
- Defaults to a dry-run; `--apply` is required for a real upload.

If a token is ever exposed through chat, logs, or commit history, it
**MUST** be revoked at https://huggingface.co/settings/tokens and rotated
before any further sync.

## Consequences

Positive:

- Single, auditable entry point for Hub writes.
- No accidental token leakage in screenshots, recordings, or context
  windows.
- The default dry-run mode prevents accidental writes from a copy-paste
  or a stray shell history entry.

Negative / costs:

- Interactive `huggingface-cli login` is required on each new machine;
  there is no shared "dev token" baked into the repo.
- CI must use GitHub Actions secrets (`HF_TOKEN` is already configured
  there, scoped to this repo).
- Local developers must remember to revoke any token they ever type
  into a chat window.

## Alternatives considered

- **Embed the token in `.env` and load it via `python-dotenv`.** Rejected:
  `.env` files are routinely pasted into chats and shown in screen-shares,
  so this provides only a thin layer of protection.
- **Use a per-developer token file path.** Rejected: the standard
  `~/.huggingface/token` is already supported and is the recommended
  Hugging Face workflow.
- **Mint a fine-grained, read-only token for the local script.** Rejected:
  the sync script needs write access to push `registry.json`.

## References

- `scripts/sync_to_hub.py` — the implementation that enforces this policy.
- Hugging Face Hub authentication docs:
  https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication
