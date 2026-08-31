# Evaluation

Reproducible evaluation of AgentPick against a gold-standard question set,
with the baseline comparisons promised in the thesis proposal.

## Dataset

[`dataset.json`](dataset.json) — 20 questions, every gold answer verified
against the live catalog (1,709 models), in six categories:

| Category | n | Gold answer | What a good system does |
|---|---|---|---|
| `deterministic` | 4 | one exact model (alternatives listed where defensible) | returns it first |
| `ranking` | 8 | ordered list of 3+ models | reproduces the ranking |
| `ambiguous` | 2 | acceptable suggestions | suggests options **and** asks a clarifying question |
| `impossible` | 3 | none | abstains (contradiction, nonexistent model, empty filter) |
| `multi_turn` | 2 | ordered list of 3 models | asks a clarifying question on turn 1, then answers turn 2 using the dialogue context |
| `off_topic` | 1 | none | politely redirects without recommending any model |

Multi-turn questions have a `turns` array instead of `question`; the runner
feeds the turns one by one, passing the growing dialogue each time.

## Systems compared

| System | Description |
|---|---|
| `agent` | Full AgentPick: LLM orchestrator + catalog tools (Qdrant + PostgreSQL) in an open-ended loop — the model sees tool results and decides what to run next |
| `llm_only` | Same LLM, no catalog access — answers from parametric knowledge |
| `single_round` | LLM-parameterized retrieval without a loop, as a fixed code pipeline: one planning completion translates the request into query parameters (JSON), code runs exactly one structured filter (PostgreSQL) and one semantic search (Qdrant) with them, and a second completion answers from those results. The LLM adapts the queries but never sees results before they are final — the `agent` − `single_round` delta isolates the agentic loop itself |
| `qdrant_only` | Fixed vanilla RAG: code (not the LLM) embeds the user's words, retrieves top-8 from Qdrant, and the LLM answers in one completion. No structured store, no adaptivity |

The ladder reads bottom-up: `llm_only` → `qdrant_only` adds retrieval grounding,
`qdrant_only` → `single_round` adds LLM-directed retrieval (structured store +
adapted parameters), and `single_round` → `agent` adds only the ability to react
to results and query again — the agentic loop.

## Metrics

Answers are free text; a ranked prediction is extracted by matching
`org/model` ids in order of first mention (case-insensitive). Extracted ids
are validated against a committed snapshot of the catalog's model ids
([`catalog_ids.txt`](catalog_ids.txt)); out-of-catalog mentions are kept when
they plausibly name a real repo (so hallucinated picks still cost precision)
and dropped when they are slash-separated prose ("chat/instruction-tuned").
Gold models mentioned by bare repo name (e.g. "Qwen2.5-Coder-7B-Instruct"
without the org prefix) are also credited, so systems are not penalized for
id formatting.

- **deterministic / ranking / multi_turn:** precision@3, recall@3, MRR, nDCG@3
  (graded relevance from the gold order, so rank errors are penalized),
  computed on the final answer.
- **ambiguous:** expected-model mention rate.
- **impossible / off_topic:** no automatic score — a correct answer names no
  model, which no ranking metric can express.
- **all:** wall-clock latency per answer (whole dialogue for multi-turn).

Per-category summaries report the mean of each metric with a
percentile-bootstrap 95% confidence interval (when n ≥ 2). Every raw answer is
stored in the results JSON so scores can be audited and the unscored
categories graded by a human (as can explanation quality).

## Running

Requires the data stores (Qdrant + Postgres, populated) and `OPENAI_API_KEY`
for the LLM-based systems.

Run from the repository root with the backend virtualenv (the module puts
`backend/` on `sys.path` itself):

```bash
export OPENAI_API_KEY=sk-... QDRANT_URL=http://localhost:6333 POSTGRES_PORT=5433

backend/.venv/bin/python -m evaluation.run
backend/.venv/bin/python -m evaluation.run --systems agent          # agent only
backend/.venv/bin/python -m evaluation.run --ids D1 Q5 Q20          # subset
backend/.venv/bin/python -m evaluation.run --categories ranking     # one category
backend/.venv/bin/python -m evaluation.run --runs 5                 # 5 repetitions
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

### Multi-run pooled comparison

Single runs are noisy (the same question can flip 0↔1 between runs with
identical code), so headline claims rest on several repetitions:
`--runs 5` interleaves five full repetitions of every system, and
`evaluation.pooled` groups the result files by system, averages each
question's scores across runs, and tests the pooled per-question paired
differences with an *exact* sign-flip permutation test (all 2^n sign
assignments), a bootstrap 95% CI, per-question win/tie/loss counts, and —
with ≥ 3 runs per system — the p-value range when any single run is left
out. The reference system is tested against every other system, so the
p-values of one metric form a family and are reported Holm-corrected as well.

It also derives a per-question `composite` task-success score so the scored
categories enter one paired test: nDCG@k for deterministic/ranking/multi_turn
and mention rate for ambiguous, over 16 of the 20 questions.

impossible and off_topic carry **no** composite. The only rule expressible
without phrase heuristics — "correct iff the answer names no model" — measures
silence rather than correctness: a retrieval baseline that happens to find
nothing scores 1.0, while an answer that correctly denies the request and
offers a verified alternative scores 0.0. It also contradicts the dataset it
scores, whose N5 justification accepts naming a real alternative alongside the
denial. Those answers are graded by hand from the stored results instead.

The out-of-catalog recommendation rate the same command reports per system as
grounding evidence does not count model ids the user named in the question (a
correct answer to "should I use *nonexistent model*?" quotes that id back in
order to deny it).

```bash
backend/.venv/bin/python -m evaluation.run --runs 5
backend/.venv/bin/python -m evaluation.pooled                     # latest batch
backend/.venv/bin/python -m evaluation.pooled 20260713_235740     # by timestamp
backend/.venv/bin/python -m evaluation.pooled --metrics all --per-question
backend/.venv/bin/python -m evaluation.pooled --reference single_round
```

Unit tests: `backend/.venv/bin/python -m pytest evaluation/tests` (from the
repository root).
