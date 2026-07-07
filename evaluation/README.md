# Evaluation

Reproducible evaluation of AgentPick against a gold-standard question set,
with the baseline comparisons promised in the thesis proposal.

## Dataset

[`dataset.json`](dataset.json) — 20 questions, every gold answer verified
against the live catalog (1,709 models), in six categories:

| Category | n | Gold answer | What a good system does |
|---|---|---|---|
| `deterministic` | 3 | one exact model (alternatives listed where defensible) | returns it first |
| `ranking` | 9 | ordered list of 3 models | reproduces the ranking |
| `ambiguous` | 2 | acceptable suggestions | suggests options **and** asks a clarifying question |
| `impossible` | 3 | none | abstains (contradiction, nonexistent model, empty filter) |
| `multi_turn` | 2 | ordered list of 3 models | asks a clarifying question on turn 1, then answers turn 2 using the dialogue context |
| `off_topic` | 1 | none | politely redirects without recommending any model |

Multi-turn questions have a `turns` array instead of `question`; the runner
feeds the turns one by one, passing the growing dialogue each time.

## Systems compared

| System | Description |
|---|---|
| `agent` | Full AgentPick: LLM orchestrator + catalog tools (Qdrant + PostgreSQL) |
| `llm_only` | Same LLM, no catalog access — answers from parametric knowledge |

## Metrics

Answers are free text; a ranked prediction is extracted by matching
`org/model` ids in order of first mention (case-insensitive). Gold models
mentioned by bare repo name (e.g. "Qwen2.5-Coder-7B-Instruct" without the
org prefix) are also credited, so systems are not penalized for id
formatting.

- **deterministic / ranking:** precision@3, recall@3, MRR, nDCG@3
  (graded relevance from the gold order, so rank errors are penalized).
- **ambiguous:** clarification rate (answer asks a question) and expected-model
  mention rate.
- **impossible:** abstention rate (states no model fits / is not in the catalog).
- **multi_turn:** turn-1 clarification rate plus the ranking metrics on the
  final answer.
- **off_topic:** redirect rate (answer recommends no models).
- **all:** wall-clock latency per answer (whole dialogue for multi-turn).

Per-category summaries report the mean of each metric with a
percentile-bootstrap 95% confidence interval (when n ≥ 2). The abstention and
clarification metrics are phrase/pattern heuristics; every raw answer is
stored in the results JSON so scores can be audited and answers additionally
graded by a human (e.g. explanation quality).

## Running

Requires the data stores (Qdrant + Postgres, populated) and `OPENAI_API_KEY`
for the LLM-based systems:

Run from the repository root with the backend virtualenv (the module puts
`backend/` on `sys.path` itself):

```bash
export OPENAI_API_KEY=sk-... QDRANT_URL=http://localhost:6333 POSTGRES_PORT=5433

backend/.venv/bin/python -m evaluation.run
backend/.venv/bin/python -m evaluation.run --systems agent          # agent only
backend/.venv/bin/python -m evaluation.run --ids D1 Q5 Q20          # subset
backend/.venv/bin/python -m evaluation.run --categories ranking     # one category
```

Results land in `evaluation/results/<timestamp>_<system>.json` (per-question
answers + scores, per-category means with bootstrap 95% CIs) and a summary is
printed.

### Rescoring and system comparison

Because raw answers are stored, scores can be recomputed after a metrics
change without re-calling the API (writes `<name>_rescored.json`):

```bash
backend/.venv/bin/python -m evaluation.run --rescore evaluation/results/<file>.json
```

Two systems are compared with a paired randomization (sign-flip permutation)
test and a bootstrap 95% CI of the mean paired difference, per metric over
the questions both share:

```bash
backend/.venv/bin/python -m evaluation.compare results/A.json results/B.json
```

Unit tests: `backend/.venv/bin/python -m pytest evaluation/tests` (from the
repository root).
