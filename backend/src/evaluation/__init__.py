"""
Evaluation package.

Contains metrics computation and benchmark suite for the recommendation system.
"""

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    latency_stats,
    compute_evaluation_metrics,
)
from src.evaluation.benchmarks import (
    run_benchmark,
    run_benchmark_subset,
    print_benchmark_report,
    BENCHMARK_QUERIES,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "latency_stats",
    "compute_evaluation_metrics",
    "run_benchmark",
    "run_benchmark_subset",
    "print_benchmark_report",
    "BENCHMARK_QUERIES",
]
