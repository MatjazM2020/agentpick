"""
Evaluation metrics for the recommendation system.

Implements standard IR metrics:
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)
- Latency measurement

All metrics assume:
- recommended: list of model_ids in ranked order
- relevant: set of relevant model_ids (ground truth)
"""

import logging
from typing import List, Set
import statistics


logger = logging.getLogger(__name__)


def precision_at_k(
    recommended: List[str],
    relevant: Set[str],
    k: int = 5
) -> float:
    """
    Compute Precision@K.
    
    Precision@K = (number of relevant models in top-K) / K
    
    Range: [0, 1]
    
    Args:
        recommended: List of recommended model IDs (ranked order)
        relevant: Set of relevant model IDs
        k: Cutoff for top-K
        
    Returns:
        Precision score [0, 1]
    """
    if k <= 0:
        return 0.0
    
    if not recommended or not relevant:
        return 0.0
    
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    
    precision = hits / k
    logger.debug(f"Precision@{k}: {precision:.3f} ({hits}/{k})")
    
    return precision


def recall_at_k(
    recommended: List[str],
    relevant: Set[str],
    k: int = 5
) -> float:
    """
    Compute Recall@K.
    
    Recall@K = (number of relevant models in top-K) / (total relevant models)
    
    Range: [0, 1]
    
    Args:
        recommended: List of recommended model IDs (ranked order)
        relevant: Set of relevant model IDs
        k: Cutoff for top-K
        
    Returns:
        Recall score [0, 1]
    """
    if not relevant:
        return 0.0
    
    if not recommended:
        return 0.0
    
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    
    recall = hits / len(relevant)
    logger.debug(f"Recall@{k}: {recall:.3f} ({hits}/{len(relevant)})")
    
    return recall


def mean_reciprocal_rank(
    recommended: List[str],
    relevant: Set[str]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    MRR = 1 / rank_of_first_relevant
    
    If no relevant item found, MRR = 0.0
    
    Range: [0, 1]
    
    Args:
        recommended: List of recommended model IDs (ranked order)
        relevant: Set of relevant model IDs
        
    Returns:
        MRR score [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    
    for rank, model_id in enumerate(recommended, start=1):
        if model_id in relevant:
            mrr = 1.0 / rank
            logger.debug(f"MRR: {mrr:.3f} (first match at rank {rank})")
            return mrr
    
    logger.debug("MRR: 0.0 (no match found)")
    return 0.0


def latency_stats(latencies: List[float]) -> dict:
    """
    Compute latency statistics.
    
    Args:
        latencies: List of latency measurements in seconds
        
    Returns:
        Dict with:
        - mean: Average latency (seconds)
        - median: Median latency (seconds)
        - min: Minimum latency (seconds)
        - max: Maximum latency (seconds)
        - stdev: Standard deviation (seconds), or 0 if < 2 samples
        - count: Number of measurements
    """
    if not latencies:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stdev": 0.0,
            "count": 0
        }
    
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    stdev = statistics.stdev(latencies) if len(latencies) >= 2 else 0.0
    
    stats = {
        "mean": round(mean, 3),
        "median": round(median, 3),
        "min": round(min_lat, 3),
        "max": round(max_lat, 3),
        "stdev": round(stdev, 3),
        "count": len(latencies)
    }
    
    logger.debug(f"Latency stats: {stats}")
    
    return stats


def compute_evaluation_metrics(
    recommended_list: List[List[str]],
    relevant_list: List[Set[str]],
    latencies: List[float],
    k_values: List[int] = [5, 10]
) -> dict:
    """
    Compute evaluation metrics for a batch of queries.
    
    Args:
        recommended_list: List of ranked recommendation lists
        relevant_list: List of relevant model sets (ground truth)
        latencies: List of latency measurements
        k_values: List of K values for Precision@K and Recall@K
        
    Returns:
        Dict with aggregated metrics:
        {
            "precision": {k: average_precision_at_k, ...},
            "recall": {k: average_recall_at_k, ...},
            "mrr": mean_reciprocal_rank,
            "latency": latency_stats_dict,
            "num_queries": count
        }
    """
    
    if len(recommended_list) != len(relevant_list):
        raise ValueError(
            f"Length mismatch: {len(recommended_list)} recommendations "
            f"vs {len(relevant_list)} relevant sets"
        )
    
    num_queries = len(recommended_list)
    
    if num_queries == 0:
        logger.warning("Empty evaluation set")
        return {
            "precision": {k: 0.0 for k in k_values},
            "recall": {k: 0.0 for k in k_values},
            "mrr": 0.0,
            "latency": latency_stats([]),
            "num_queries": 0
        }
    
    # Compute metrics for each query
    precision_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    mrr_scores = []
    
    for recommended, relevant in zip(recommended_list, relevant_list):
        # Precision and Recall
        for k in k_values:
            prec = precision_at_k(recommended, relevant, k)
            recall = recall_at_k(recommended, relevant, k)
            precision_scores[k].append(prec)
            recall_scores[k].append(recall)
        
        # MRR
        mrr = mean_reciprocal_rank(recommended, relevant)
        mrr_scores.append(mrr)
    
    # Average metrics
    avg_precision = {
        k: round(statistics.mean(scores), 3) if scores else 0.0
        for k, scores in precision_scores.items()
    }
    avg_recall = {
        k: round(statistics.mean(scores), 3) if scores else 0.0
        for k, scores in recall_scores.items()
    }
    avg_mrr = round(statistics.mean(mrr_scores), 3) if mrr_scores else 0.0
    
    latency = latency_stats(latencies)
    
    result = {
        "precision": avg_precision,
        "recall": avg_recall,
        "mrr": avg_mrr,
        "latency": latency,
        "num_queries": num_queries
    }
    
    logger.info(
        f"Evaluation results: "
        f"Precision@5={avg_precision.get(5, 0.0)}, "
        f"Recall@5={avg_recall.get(5, 0.0)}, "
        f"MRR={avg_mrr}, "
        f"Queries={num_queries}"
    )
    
    return result
