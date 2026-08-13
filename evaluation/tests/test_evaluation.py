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


def test_extract_keeps_word_shaped_catalog_ids():
    # Catalog membership keeps ids whose repo side looks like a prose word
    # ("biogpt", "starcoder"); shape heuristics alone would drop them.
    answer = "Compare microsoft/biogpt and bigcode/starcoder, not vendor/chatbot."
    assert metrics.extract_model_ids(answer) == [
        "microsoft/biogpt",
        "bigcode/starcoder",
    ]


def test_extract_empty_text():
    assert metrics.extract_model_ids("") == []


def test_extract_ignores_quant_levels_and_bare_versions():
    # Junk actually produced by eval answers: GGUF quant levels ("Q4/Q5",
    # "Q4_K_M/Q5_0") and runtime versions ("exllama/v2").
    answer = "GGUF quant levels like Q4/Q5/Q8 via exllama/v2 or Q4_K_M/Q5_0."
    assert metrics.extract_model_ids(answer) == []


def test_extract_ignores_prose_junk_pairs_from_eval_answers():
    # Junk shapes actually produced by the 2026-07-13/14 runs: family names
    # with dotted versions, size bounds, format tokens, prose compounds, and
    # GPU names, each written as an "org/repo"-looking pair.
    answer = (
        "Qwen2.5/CodeLlama families use context scaling via Qwen3/Qwen2.5 "
        "recipes; pick a mini/under-35B model shipped as GGUF/MLX-4bit or an "
        "instruction-tuned/open-instruct-style checkpoint. For "
        "document/seq-to-seq translation on compute/RAM-efficient hardware "
        "(A100/A6000), llama.cpp/llama.cpp-compatible builds handle "
        "under-1B/near-1B drafts and PubMed/PubMed-like corpora."
    )
    assert metrics.extract_model_ids(answer) == []


def test_extract_drops_family_umbrella_before_member_ids():
    # Agent N12: "I'd deploy Qwen/Qwen2.5-Coder as the family" preceded the
    # actual picks and was scored as the (wrong) top prediction.
    answer = (
        "I'd deploy Qwen/Qwen2.5-Coder as the family: "
        "1. Qwen/Qwen2.5-Coder-32B-Instruct — large tier. "
        "2. Qwen/Qwen2.5-Coder-7B-Instruct — mid tier."
    )
    assert metrics.extract_model_ids(answer) == [
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ]


def test_extract_keeps_umbrella_shaped_id_when_it_is_gold():
    # A gold id that a longer mentioned id happens to extend must survive.
    gold = ["Qwen/Qwen3-32B"]
    answer = "Use Qwen/Qwen3-32B; avoid the Qwen/Qwen3-32B-AWQ re-upload."
    assert metrics.extract_predictions(answer, gold) == [
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-32B-AWQ",
    ]


def test_extract_predictions_credits_verbatim_full_gold_id():
    # "microsoft/biogpt" (N3 gold): the all-lowercase single-token repo fails
    # the shape heuristics and the bare-name fallback refused matches after
    # "/", so the exact gold id written verbatim previously scored 0.
    gold = ["BioMistral/BioMistral-7B", "microsoft/biogpt"]
    answer = "1. microsoft/biogpt — the classic biomedical generator."
    assert metrics.extract_predictions(answer, gold) == ["microsoft/biogpt"]


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
# Scoring
# ---------------------------------------------------------------------------

def _question(category: str, gold: tuple[str, ...] = ()):
    from evaluation.dataset import EvalQuestion

    return EvalQuestion(id="X", category=category, turns=("q",), expected_models=gold)


def test_score_answer_ranked_categories_report_the_ranking_metrics():
    from evaluation.run import score_answer

    scores = score_answer(_question("ranking", tuple(GOLD)), [f"1. {GOLD[0]}"], 3)
    assert set(scores) == {"predicted_models", "precision@3", "recall@3", "mrr", "ndcg@3"}
    assert scores["mrr"] == 1.0


def test_score_answer_leaves_no_model_categories_unscored():
    # A correct impossible/off_topic answer names no model, which no ranking
    # metric can express — those questions are graded qualitatively instead.
    from evaluation.run import score_answer

    for category in ("impossible", "off_topic"):
        scores = score_answer(_question(category), ["No catalog model fits."], 3)
        assert scores == {"predicted_models": []}


def test_score_answer_ambiguous_reports_only_expected_mention():
    from evaluation.run import score_answer

    scores = score_answer(_question("ambiguous", tuple(GOLD)), [f"Maybe {GOLD[2]}."], 3)
    assert scores["mentions_expected"] == 1.0
    assert "asks_clarification" not in scores


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
        "deterministic": 4,
        "ranking": 8,
        "ambiguous": 2,
        "impossible": 3,
        "multi_turn": 2,
        "off_topic": 1,
    }


def test_dataset_filters():
    assert [q.id for q in load_dataset(ids=["D1", "Q20"])] == ["D1", "Q20"]
    ranking = load_dataset(categories=["ranking"])
    assert len(ranking) == 8
    assert all(len(q.expected_models) >= 3 for q in ranking)


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


# ---------------------------------------------------------------------------
# Pooled multi-run comparison (exact permutation test, Holm, composite)
# ---------------------------------------------------------------------------

def test_exact_sign_flip_matches_brute_force_enumeration():
    """The meet-in-the-middle count must equal a naive 2^n enumeration."""
    import itertools
    import random

    from evaluation.pooled import exact_sign_flip_p

    rng = random.Random(7)
    for n in range(2, 11):
        diffs = [round(rng.uniform(-1, 1), 3) for _ in range(n)]
        observed = abs(sum(diffs)) - 1e-12
        brute = sum(
            1
            for signs in itertools.product((1, -1), repeat=n)
            if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed
        ) / 2**n
        assert exact_sign_flip_p(diffs) == brute


def test_exact_sign_flip_edge_cases():
    from evaluation.pooled import exact_sign_flip_p

    assert exact_sign_flip_p([0.0, 0.0, 0.0]) == 1.0  # nothing to test
    assert exact_sign_flip_p([0.4]) == 1.0  # one question can never be significant
    assert exact_sign_flip_p([1.0] * 5) == 2 / 32  # only all-plus and all-minus
    # A consistent difference over enough questions clears the usual threshold.
    assert exact_sign_flip_p([0.3] * 14) < 0.05


def test_holm_adjusts_by_rank_and_stays_monotone():
    from evaluation.pooled import holm

    # Smallest p gets the full family size, the largest gets 1x, and the
    # adjusted values may never decrease as the raw ones increase.
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]  # capped at 1


def _pooled_result(category: str, scores: dict, question: str = "pick a model") -> dict:
    return {"id": "X1", "category": category, "question": question, "scores": scores}


def test_composite_uses_the_primary_metric_of_each_category():
    from evaluation.pooled import _composite

    ranked = _pooled_result("ranking", {"ndcg@3": 0.75, "predicted_models": ["a/b"]})
    ambiguous = _pooled_result("ambiguous", {"mentions_expected": 1.0, "predicted_models": ["a/b"]})
    assert _composite(ranked, 3) == 0.75
    assert _composite(ambiguous, 3) == 1.0


def test_composite_leaves_the_abstention_categories_unscored():
    """"Named no model" measures silence, not correctness — a baseline that finds
    nothing would outscore a correct denial that offers a verified alternative, which
    the dataset explicitly accepts (N5). Those two categories are graded by hand."""
    from evaluation.pooled import _composite

    for category in ("impossible", "off_topic"):
        assert _composite(_pooled_result(category, {"predicted_models": []}), 3) is None
        assert _composite(_pooled_result(category, {"predicted_models": ["a/b"]}), 3) is None


def test_every_baseline_is_given_the_agents_domain_knowledge():
    """The systems must differ in what they can do, not in what they were told about
    the catalog: a fact given to one system has to reach all four, or the comparison
    measures prompt content instead of architecture."""
    from evaluation import systems

    for prompt in (
        systems.LLM_ONLY_INSTRUCTIONS,
        systems.SINGLE_ROUND_PLAN_INSTRUCTIONS,
        systems.SINGLE_ROUND_INSTRUCTIONS,
        systems.QDRANT_ONLY_INSTRUCTIONS,
    ):
        assert systems.CATALOG_KNOWLEDGE in prompt


def test_no_baseline_is_told_more_than_the_agent_about_off_topic_requests():
    """The off-topic questions are graded by hand, so the scope rule must be worded
    the same everywhere: telling only the baselines "do not name any model" grades
    them on an instruction the agent never got."""
    from evaluation import systems

    scope_rule = "say briefly that you\n  only help with picking models from the catalog."
    assert scope_rule in systems._ANSWER_RULES
    assert "help with picking models." in systems.LLM_ONLY_INSTRUCTIONS
    for prompt in (systems.LLM_ONLY_INSTRUCTIONS, systems._ANSWER_RULES):
        assert "do not name any model" not in prompt


def test_single_round_can_express_every_filter_the_agent_can():
    """The planner's filter must cover the same parameters as the agent's
    filter_models tool, so the delta is the loop and not a narrower query API."""
    import inspect

    from src import catalog
    from evaluation.systems import _filter_kwargs

    tool_params = set(inspect.signature(catalog.filter_models).parameters) - {"limit"}
    plan = {"filter": {"task_type": "translation", "tag": "code",
                       "name_contains": "instruct", "min_params_b": 1,
                       "max_params_b": 9, "sort_by": "largest"}}
    assert set(_filter_kwargs(plan)) == tool_params


def test_out_of_catalog_rate_ignores_a_model_id_the_user_named():
    """N5 quotes a nonexistent id back to deny it — that is not a recommendation,
    so it must not count against the system's grounding."""
    from evaluation.pooled import _system_picks

    fake = "meta-llama/Llama-5-70B-Instruct"
    denial = _pooled_result(
        "impossible",
        {"predicted_models": [fake]},
        question=f"I read about {fake}. Should I use it for my chatbot?",
    )
    assert _system_picks(denial) == []
    # An alternative the system itself put forward still counts as a pick.
    denial["scores"]["predicted_models"] = [fake, "Qwen/Qwen2.5-7B-Instruct"]
    assert _system_picks(denial) == ["Qwen/Qwen2.5-7B-Instruct"]
