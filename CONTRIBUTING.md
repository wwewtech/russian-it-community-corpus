# Contributing to Russian IT Community Corpus

Thank you for your interest in contributing to the Russian IT Community Data Engineering Platform.

---

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all participants with respect and professionalism.

---

## Development Setup

### 1. Prerequisites
- Python **3.11, 3.12, or 3.13**
- Git & Git LFS
- NVIDIA GPU with CUDA support (optional, for LoRA training and embeddings)

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/wwewtech/russian-it-community-corpus.git
cd russian-it-community-corpus

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies in editable mode
pip install -r requirements.txt
```

---

## Contribution Workflow

1. **Fork** the repository on GitHub.
2. Create a topic branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```
3. Make your changes adhering to code standards.
4. Run the test suite and Zero-PII audit:
   ```bash
   python -m unittest discover -s tests
   make audit
   ```
5. Commit your changes using conventional commit messages (`feat:`, `fix:`, `docs:`, `refactor:`).
6. Push to your fork and submit a **Pull Request**.

---

## Pull Request Guidelines

- Ensure all existing and new unit tests pass (`python -m unittest discover -s tests`).
- Include tests for any new functionality (e.g. new PII scrubber patterns or taxonomy domains).
- If your change affects dataset outputs, ensure no personal data (PII) is introduced.
- Provide a clear explanation in the PR description following our [PR Template](.github/PULL_REQUEST_TEMPLATE.md).

---

## Zero-PII & Privacy Strict Rules

If you are modifying anonymization logic (`src/pii/`):
- Never commit real personal names, phone numbers, API keys, or private links.
- Run `make audit` to verify against the adversarial red-team suite.
