"""Unit tests for the evaluation module (metrics, extraction, dataset)."""

import math

from evaluation import metrics
from evaluation.dataset import CATEGORIES, load_dataset

GOLD = [
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

GOLD_CODER = [
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
]


# ---------------------------------------------------------------------------
# Model-id extraction
# ---------------------------------------------------------------------------

def test_extract_model_ids_from_answer_text():
    answer = (
        "For 8 GB VRAM I recommend:\n"
        "1. Qwen/Qwen2.5-3B-Instruct — strong multilingual support.\n"
        "2. meta-llama/Llama-3.2-3B-Instruct — solid general assistant.\n"
        "See https://huggingface.co/some/model for details. "
        "Qwen/Qwen2.5-3B-Instruct fits in 8 GB."
    )
    assert metrics.extract_model_ids(answer) == [
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]


def test_extract_ignores_fractions_and_urls():
    assert metrics.extract_model_ids("Use 3/4 of the RAM, see https://a.co/b-c") == []


def test_extract_ignores_prose_word_pairs():
    answer = (
        "It has conversational/chat tags, runs on PyTorch/transformers at "
        "40 tokens/s, but openai/gpt-oss-120b and deepseek-ai/DeepSeek-R1 fit."
    )
    assert metrics.extract_model_ids(answer) == [
        "openai/gpt-oss-120b",
        "deepseek-ai/DeepSeek-R1",
    ]


def test_extract_ignores_quantities_and_compounds():
    # Junk shapes actually produced by eval answers.
    answer = (
        "It's chat/instruction-tuned, quantized 4-bit/8-bit or FP32/FP16, "
        "runs on GGUF/llama.cpp with a bigger/70B or 3B/0.6B option in the "
        "Llama/Qwen-style family for Spanish/French users on GPU/CPU."
    )
    assert metrics.extract_model_ids(answer) == []


def test_extract_keeps_digitless_real_ids():
    answer = (
        "Try stanford-crfm/BioMedLM, Helsinki-NLP/opus-mt-mul-en, or "
        "echarlaix/tiny-random-PhiForCausalLM."
    )
    assert metrics.extract_model_ids(answer) == [
        "stanford-crfm/BioMedLM",
        "Helsinki-NLP/opus-mt-mul-en",
        "echarlaix/tiny-random-PhiForCausalLM",
    ]


def test_extract_empty_text():
    assert metrics.extract_model_ids("") == []


def test_extract_ignores_quant_levels_and_bare_versions():
    # Junk actually produced by eval answers: GGUF quant levels ("Q4/Q5",
    # "Q4_K_M/Q5_0") and runtime versions ("exllama/v2").
    answer = "GGUF quant levels like Q4/Q5/Q8 via exllama/v2 or Q4_K_M/Q5_0."
    assert metrics.extract_model_ids(answer) == []


def test_extract_predictions_credits_bare_gold_repo_name():
    # llm_only Q6: the top pick was written without the org prefix and
    # previously scored 0.
    answer = (
        "1. **Qwen2.5-Coder-7B-Instruct** — strong coding model.\n"
        "2. microsoft/Phi-3.5-mini-instruct — lightweight alternative."
    )
    assert metrics.extract_predictions(answer, GOLD_CODER) == [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
    ]


def test_extract_predictions_no_credit_inside_longer_id():
    # "pearl-ai/Llama-3.3-70B-Instruct-pearl" must not credit the gold
    # "meta-llama/Llama-3.3-70B-Instruct" (different model).
    answer = "1. pearl-ai/Llama-3.3-70B-Instruct-pearl — largest instruct model."
    gold = ["meta-llama/Llama-3.3-70B-Instruct"]
    assert metrics.extract_predictions(answer, gold) == [
        "pearl-ai/Llama-3.3-70B-Instruct-pearl"
    ]


def test_extract_predictions_without_gold_matches_plain_extraction():
    answer = "Try Qwen/Qwen2.5-3B-Instruct."
    assert metrics.extract_predictions(answer, ()) == metrics.extract_model_ids(answer)


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def test_precision_and_recall_at_k():
    predicted = [GOLD[0], "other/model", GOLD[2]]
    assert metrics.precision_at_k(predicted, GOLD, 3) == 2 / 3
    assert metrics.recall_at_k(predicted, GOLD, 3) == 2 / 3
    assert metrics.recall_at_k(predicted, GOLD[:1], 3) == 1.0


def test_matching_is_case_insensitive():
    assert metrics.precision_at_k(["qwen/qwen2.5-3b-instruct"], GOLD, 1) == 1.0


def test_mrr():
    assert metrics.mrr(GOLD, GOLD) == 1.0
    assert metrics.mrr(["other/model", GOLD[1]], GOLD) == 0.5
    assert metrics.mrr(["other/model"], GOLD) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert metrics.ndcg_at_k(GOLD, GOLD, 3) == 1.0


def test_ndcg_swapped_top_two():
    predicted = [GOLD[1], GOLD[0], GOLD[2]]
    # grades: gold[0]=3, gold[1]=2, gold[2]=1
    dcg = 2 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert metrics.ndcg_at_k(predicted, GOLD, 3) == dcg / idcg
    assert metrics.ndcg_at_k(predicted, GOLD, 3) < 1.0


def test_ndcg_no_relevant_predictions():
    assert metrics.ndcg_at_k(["a/b", "c/d"], GOLD, 3) == 0.0


# ---------------------------------------------------------------------------
# Behavioral heuristics
# ---------------------------------------------------------------------------

def test_detects_impossible():
    assert metrics.detects_impossible(
        "No catalog model satisfies these constraints: a model cannot be "
        "under 1B and over 70B at the same time."
    )
    assert metrics.detects_impossible(
        "meta-llama/Llama-5-70B-Instruct is not in the catalog; consider "
        "meta-llama/Llama-3.3-70B-Instruct instead."
    )
    assert not metrics.detects_impossible("1. Qwen/Qwen2.5-3B-Instruct — best fit.")


def test_detects_impossible_cant_find_with_curly_apostrophe():
    assert metrics.detects_impossible(
        "I can\u2019t find a model card for meta-llama/Llama-5-70B-Instruct."
    )
    assert metrics.detects_impossible(
        "I can\u2019t satisfy that requirement as stated."
    )


def test_detects_impossible_empty_scan_with_markdown():
    # Agent N6: abstains but the phrase is wrapped in markdown emphasis.
    assert metrics.detects_impossible(
        "An HF catalog scan returns **nothing**—the largest models are 70B."
    )
    assert metrics.detects_impossible("None meet a **2 trillion** threshold.")


def test_detects_impossible_show_up_and_not_aware_paraphrases():
    # gpt-5.4-mini paraphrases from the 20260709_010449 run: agent N5
    # ("didn't show up in the catalog search") and llm_only N6
    # ("not aware of any ... model").
    assert metrics.detects_impossible(
        "I wouldn’t choose **meta-llama/Llama-5-70B-Instruct** as my "
        "default chatbot pick here, because it didn’t show up in the "
        "catalog search at all."
    )
    assert metrics.detects_impossible(
        "I’m not aware of any **open-weight, instruction-tuned** model "
        "on the Hugging Face Hub with **at least 2 trillion parameters**."
    )


def test_asks_clarification():
    assert metrics.asks_clarification("Here are options. What hardware do you have?")
    assert not metrics.asks_clarification("1. org/model — best fit.")


def test_asks_clarification_imperative_without_question_mark():
    # Agent N1 turn 1: a clarification request phrased imperatively.
    assert metrics.asks_clarification(
        "If you tell me your target (GPU/CPU + max memory) and whether you "
        "want Python-only or multi-language, I can narrow it to the best size."
    )


def test_asks_clarification_please_tell_me_without_question_mark():
    # Agent N2 turn 1 (184045 run): numbered setup questions without "?".
    assert metrics.asks_clarification(
        "Please tell me one thing about your setup:\n"
        "1) What GPU/RAM do you have (e.g., 8GB CPU only, 12GB GPU, 24GB GPU), and\n"
        "2) Your main use-case (chat/Q&A, coding, summarization, or translation)."
    )


# ---------------------------------------------------------------------------
# Explanation-quality text metrics (loads the BERTScore model — slow once)
# ---------------------------------------------------------------------------

def test_text_scores_identity_beats_unrelated():
    reference = (
        "The Qwen2.5 series is selected for its superior multilingual "
        "capabilities in smaller parameter sizes."
    )
    identical = metrics.text_scores(reference, reference)
    unrelated = metrics.text_scores("Photosynthesis converts sunlight.", reference)
    assert identical["rougeL"] == 1.0
    assert identical["bleu"] > 0.9
    for key in ("rougeL", "bleu", "bertscore_f1"):
        assert identical[key] > unrelated[key]


def test_text_scores_empty_answer_is_zero():
    assert metrics.text_scores("", "reference text") == {
        "rougeL": 0.0, "bleu": 0.0, "bertscore_f1": 0.0,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_dataset_loads_and_is_well_formed():
    questions = load_dataset()
    assert len(questions) == 20
    for q in questions:
        assert q.category in CATEGORIES
        assert q.question
        assert len(q.turns) == (2 if q.category == "multi_turn" else 1)
        if q.category in ("impossible", "off_topic"):
            assert q.expected_models == ()
        else:
            assert len(q.expected_models) >= 1
            assert all("/" in m for m in q.expected_models)


def test_dataset_category_balance():
    counts = {c: len(load_dataset(categories=[c])) for c in CATEGORIES}
    assert counts == {
        "deterministic": 3,
        "ranking": 9,
        "ambiguous": 2,
        "impossible": 3,
        "multi_turn": 2,
        "off_topic": 1,
    }


def test_dataset_filters():
    assert [q.id for q in load_dataset(ids=["D1", "Q20"])] == ["D1", "Q20"]
    ranking = load_dataset(categories=["ranking"])
    assert len(ranking) == 9
    assert all(len(q.expected_models) == 3 for q in ranking)


# ---------------------------------------------------------------------------
# Statistics (bootstrap CI, paired comparison)
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_mean_and_is_deterministic():
    from evaluation.run import bootstrap_ci

    values = [0.0, 0.5, 0.5, 1.0, 1.0, 1.0]
    lo, hi = bootstrap_ci(values)
    assert lo <= sum(values) / len(values) <= hi
    assert bootstrap_ci(values) == [lo, hi]  # seeded, reproducible


def _report(system: str, scores_by_id: dict[str, float]) -> dict:
    return {
        "system": system,
        "results": [
            {"id": qid, "scores": {"ndcg@3": s}, "latency_s": 1.0}
            for qid, s in scores_by_id.items()
        ],
    }


def test_compare_identical_systems():
    from evaluation.compare import compare

    report = _report("a", {f"Q{i}": 0.5 for i in range(10)})
    rows = {r["metric"]: r for r in compare(report, _report("b", {f"Q{i}": 0.5 for i in range(10)}))}
    row = rows["ndcg@3"]
    assert row["diff"] == 0.0
    assert row["p_value"] == 1.0
    assert row["diff_ci95"] == [0.0, 0.0]


def test_compare_detects_consistent_difference():
    from evaluation.compare import compare

    report_a = _report("a", {f"Q{i}": 0.9 for i in range(10)})
    report_b = _report("b", {f"Q{i}": 0.2 for i in range(10)})
    rows = {r["metric"]: r for r in compare(report_a, report_b)}
    row = rows["ndcg@3"]
    assert row["n"] == 10
    assert row["diff"] == 0.7
    assert row["p_value"] < 0.05
    assert row["diff_ci95"][0] > 0  # CI excludes zero


def test_run_dialogue_passes_growing_history():
    import asyncio

    from evaluation.run import run_dialogue

    seen: list[list[dict]] = []

    async def record_messages(messages: list[dict]) -> str:
        seen.append([dict(m) for m in messages])
        return f"answer-{len(seen)}"

    answers = asyncio.run(
        run_dialogue(record_messages, ["turn one", "turn two"], question_id="N2")
    )
    assert answers == ["answer-1", "answer-2"]
    assert seen[0] == [{"role": "user", "content": "turn one"}]
    assert seen[1] == [
        {"role": "user", "content": "turn one"},
        {"role": "assistant", "content": "answer-1"},
        {"role": "user", "content": "turn two"},
    ]
