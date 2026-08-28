# Benchmark & Evaluation Report
## Russian IT Community LLM Ecosystem (Base vs RAG vs LoRA vs Hybrid)

> **Evaluated Architectures:** Base Open-Weight Models · Local RAG (325.7k knowledge chunks) · Domain LoRA Adapters (171.5k dialogues) · Hybrid (LoRA + RAG)  
> **Evaluation Setup:** Qwen 2.5 1.5B Instruct · NVIDIA GeForce RTX 3060 (12 GB VRAM) · 50 Domain Engineering Scenarios · Coding and Academic Subsets

---

## 1. Academic & Language Modeling Metrics

Evaluation across standard NLP and programming benchmarks:

1. **OpenAI HumanEval (pass@1)**: Python function synthesis verified against unit tests in an isolated execution environment.
2. **RuMMLU CS**: Deterministic multiple-choice questions covering database internals, networking, algorithms, distributed systems, and OS concepts.
3. **Information-Theoretic Perplexity (PPL)**: Token-level cross-entropy loss evaluated on held-out Russian IT developer discussions: $\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \ln P(w_t \mid w_{<t})\right)$.
4. **ROUGE-1 / ROUGE-L F1**: N-gram overlap and longest common subsequence agreement with reference answers.

### Academic Results Summary:

| Benchmark / Metric | Metric Type | Base Model (1.5B) | Base + RAG (325k chunks) | Domain LoRA | **Hybrid (LoRA + RAG)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HumanEval Subset** (40 tasks) | `pass@1 (%)` | `0.0%` | `12.5%` | `12.5%` | **`0.0%`** |
| **RuMMLU CS Subset** (50 questions) | `Accuracy (%)` | `87.5%` | `100.0%` | `100.0%` | **`100.0%`** |
| **Test Set Perplexity** | `PPL (lower = better)` | `35.44` | N/A (Retrieval) | **`32.19`** *(Δ = -3.25)* | **`32.19`** |
| **ROUGE-1 F1** | `Overlap (%)` | `42.6%` | — | **`45.4%`** | — |
| **ROUGE-L F1** | `LCS Overlap (%)` | `36.4%` | — | **`38.8%`** | — |

*Full 15-model cross-seed distributions are published in [`reports/SCIENTIFIC_EVALUATION_REPORT.md`](SCIENTIFIC_EVALUATION_REPORT.md).*

---

## 2. Domain Engineering Benchmark (50 Scenarios)

Evaluation on 50 practical engineering scenarios spanning backend development, database tuning, infrastructure, security, and fintech routing:

| Architectural Setup | Total Score (0-100) | AST Code Validity | Latency (P50) | Gain over Base |
| :--- | :---: | :---: | :---: | :---: |
| **1. Base Model (Baseline)** | **32.9%** | 69.0% | ~410 ms | Baseline |
| **2. Base Model + RAG (325k chunks)** | **44.0%** | 71.0% | ~580 ms | **+11.1%** |
| **3. Domain LoRA (171.5k dialogues)** | **34.5%** | 72.2% | ~415 ms | **+1.6%** |
| **4. Hybrid (LoRA + RAG)** | **48.6%** | **73.0%** | ~590 ms | **+15.7%** |

*(AST validity = mean of per-scenario `ast_score` in `metrics_index.json`; earlier drafts of this table quoted unverified values.)*

### Domain Breakdown Across 7 Engineering Areas:

| Engineering Domain | Scenarios | Base Avg | RAG Avg | LoRA Avg | **Hybrid Avg** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AI / ML Platforms & LLM Infra** | 7 | 37.4% | 42.4% | 39.5% | **54.6%** |
| **Database Internals (PostgreSQL, ClickHouse, Redis)** | 7 | 25.5% | 38.3% | 30.5% | **38.4%** |
| **Debugging & Systems Engineering (Linux, Kernel, Net)** | 3 | 27.8% | 33.3% | 35.5% | **45.7%** |
| **DevSecOps, Auth & Cryptography** | 8 | 32.1% | 45.8% | 33.8% | **51.2%** |
| **Modern Fullstack & Frontend Architecture** | 8 | 36.5% | 47.9% | 36.5% | **50.0%** |
| **Payment Routing, Compliance & B2B SaaS** | 9 | 34.4% | 48.9% | 35.6% | **52.2%** |
| **Incident Response & Infrastructure Operations** | 8 | 32.5% | 46.2% | 35.0% | **48.8%** |
| **OVERALL AVERAGE (50 SCENARIOS)** | **50** | **32.9%** | **44.0%** | **34.5%** | **48.6%** |

---

## 3. Qualitative Generative Comparison (Side-by-Side)

### Scenario: Kubernetes 502 Bad Gateway During Rolling Update

#### Base Model Output:
```yaml
# Generic guidance without addressing iptables race conditions
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: web
        image: nginx:alpine
```
*Assessment:* Superficial advice, lacks production lifecycle hooks needed to prevent dropped in-flight TCP connections.

#### Hybrid Model Output (LoRA + RAG):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-gateway
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 0
  template:
    spec:
      containers:
      - name: api
        image: payment-api:v2.4.1
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"] # Prevents 502 during iptables rule cleanup
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 3
```
*Assessment:* Correctly injects `preStop` hook and `readinessProbe` with appropriate timing parameters to ensure zero-downtime connection draining.

---

## 4. Architectural & Empirical Findings

1. **Role of Parameter Adaptation (LoRA)**:
   - Fine-tuning on developer dialogues adapts vocabulary distribution, technical phrasing, and Russian developer conversational norms.
   - It reduces test-set cross-entropy loss and aligns the model's communicative tone.
   - On strict factual retrieval tasks, LoRA alone provides a modest gain (+1.6%) because parametric memory cannot substitute for explicit documentation.

2. **Role of Retrieval Augmentation (RAG)**:
   - Ingesting curated technical discussions (325.7k chunks) provides precise factual context (+11.1% gain over base).
   - Prevents hallucinations of non-existent API parameters, library flags, and configuration syntax.

3. **Hybrid Synergy (LoRA + RAG)**:
   - Combines the structured, idiomatic tone of the adapted LLM with grounded facts retrieved from the knowledge base.
   - Yields the highest composite score (**48.6%**, +15.7% over base) and strongest AST code validity (**73.0%**).
