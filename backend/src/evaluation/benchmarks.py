"""
Benchmark suite for evaluating the recommendation system.

Provides:
- Synthetic evaluation dataset with ground truth
- End-to-end evaluation harness
- Result aggregation and reporting
"""

import logging
import asyncio
import time
from typing import List, Tuple
from datetime import datetime

from backend.src.services.recommendation_pipeline import run_recommendation
from backend.src.core.state import RecommendationState
from backend.src.evaluation.metrics import compute_evaluation_metrics


logger = logging.getLogger(__name__)


# === Synthetic Evaluation Dataset ===

BENCHMARK_QUERIES = [
    {
        "query": "cheap embedding model for semantic search",
        "relevant": {"bge-small", "all-minilm", "sentence-transformers-v1"}
    },
    {
        "query": "LLM for legal document summarization",
        "relevant": {"legal-bert", "longformer", "bigbird"}
    },
    {
        "query": "fast chatbot model for CPU only",
        "relevant": {"distilgpt2", "phi-2", "tiny-llama"}
    },
    {
        "query": "small model for classification on resource-constrained device",
        "relevant": {"distilbert", "mobilebert", "squeezebert"}
    },
    {
        "query": "image recognition model with high accuracy",
        "relevant": {"efficientnet-b7", "resnet50", "vit-large"}
    },
    {
        "query": "question answering on knowledge base",
        "relevant": {"bert-qa", "albert-base", "roberta-base"}
    },
    {
        "query": "code generation model with MIT license",
        "relevant": {"codegen-small", "codeparrot-small", "code-llama"}
    },
    {
        "query": "low-latency speech recognition",
        "relevant": {"wav2vec2-small", "whisper-base", "deepspeech"}
    },
    {
        "query": "transformer model for machine translation",
        "relevant": {"m2m-100", "mBART", "t5-base"}
    },
    {
        "query": "zero-shot classification with no fine-tuning",
        "relevant": {"zero-shot-classifier", "xlm-roberta-large", "mplug"}
    }
]


async def run_benchmark(
    num_queries: int = len(BENCHMARK_QUERIES),
    timeout_per_query: float = 60.0
) -> dict:
    """
    Run evaluation benchmark on synthetic dataset.
    
    Responsibilities:
    - Execute recommendation pipeline on each query
    - Measure latency and quality
    - Compute evaluation metrics
    - Return structured report
    
    Args:
        num_queries: Number of queries to evaluate (0 for all)
        timeout_per_query: Timeout per query in seconds
        
    Returns:
        Benchmark report dict:
        {
            "timestamp": ISO timestamp,
            "num_queries": count,
            "metrics": {
                "precision": {...},
                "recall": {...},
                "mrr": ...,
                "latency": {...}
            },
            "per_query_results": [
                {
                    "query": query_text,
                    "latency": seconds,
                    "recommended": [model_ids],
                    "relevant": [model_ids],
                    "status": "success" | "timeout" | "error"
                },
                ...
            ]
        }
    """
    
    logger.info(f"[Benchmark] Starting evaluation on {num_queries} queries")
    
    start_time = time.time()
    
    # Select queries
    queries_to_evaluate = BENCHMARK_QUERIES[:num_queries] if num_queries > 0 else BENCHMARK_QUERIES
    
    # Execute pipeline on each query
    per_query_results = []
    latencies = []
    recommended_lists = []
    relevant_lists = []
    
    for idx, query_entry in enumerate(queries_to_evaluate, start=1):
        query = query_entry["query"]
        relevant = query_entry["relevant"]
        
        logger.info(f"[Benchmark] Query {idx}/{len(queries_to_evaluate)}: {query[:60]}...")
        
        query_start = time.time()
        
        try:
            # Run recommendation with timeout
            state = await asyncio.wait_for(
                run_recommendation(query),
                timeout=timeout_per_query
            )
            
            query_latency = time.time() - query_start
            latencies.append(query_latency)
            
            # Extract recommended model IDs
            recommended_ids = [
                rec.model_id for rec in state.final_recommendations
            ]
            recommended_lists.append(recommended_ids)
            relevant_lists.append(relevant)
            
            status = "success"
            logger.info(
                f"[Benchmark] Query {idx} completed in {query_latency:.2f}s. "
                f"Recommendations: {len(recommended_ids)}"
            )
        
        except asyncio.TimeoutError:
            query_latency = time.time() - query_start
            latencies.append(query_latency)
            recommended_lists.append([])
            relevant_lists.append(relevant)
            status = "timeout"
            logger.warning(f"[Benchmark] Query {idx} timed out after {query_latency:.2f}s")
        
        except Exception as e:
            query_latency = time.time() - query_start
            latencies.append(query_latency)
            recommended_lists.append([])
            relevant_lists.append(relevant)
            status = "error"
            logger.error(f"[Benchmark] Query {idx} failed: {e}")
        
        per_query_results.append({
            "query": query,
            "latency": round(query_latency, 3),
            "recommended": recommended_lists[-1],
            "relevant": list(relevant),
            "status": status
        })
    
    # Compute aggregate metrics
    metrics = compute_evaluation_metrics(
        recommended_lists,
        relevant_lists,
        latencies,
        k_values=[5, 10]
    )
    
    total_time = time.time() - start_time
    
    # Build report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "benchmark_duration": round(total_time, 2),
        "num_queries": len(queries_to_evaluate),
        "metrics": metrics,
        "per_query_results": per_query_results
    }
    
    logger.info(
        f"[Benchmark] Complete. Duration: {total_time:.2f}s. "
        f"Precision@5: {metrics['precision'].get(5, 0.0)}, "
        f"MRR: {metrics['mrr']}"
    )
    
    return report


async def run_benchmark_subset(
    queries: List[str],
    timeout_per_query: float = 60.0
) -> dict:
    """
    Run evaluation on a custom subset of queries.
    
    Useful for:
    - Testing specific query patterns
    - Quick smoke tests
    - Targeted evaluation
    
    Args:
        queries: List of query strings
        timeout_per_query: Timeout per query in seconds
        
    Returns:
        Benchmark report dict (see run_benchmark)
    """
    
    logger.info(f"[Benchmark] Running custom benchmark on {len(queries)} queries")
    
    if not queries:
        logger.warning("[Benchmark] Empty query list")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "benchmark_duration": 0.0,
            "num_queries": 0,
            "metrics": {
                "precision": {},
                "recall": {},
                "mrr": 0.0,
                "latency": {}
            },
            "per_query_results": []
        }
    
    start_time = time.time()
    
    per_query_results = []
    latencies = []
    
    # Execute on each query
    for idx, query in enumerate(queries, start=1):
        logger.info(f"[Benchmark] Custom query {idx}/{len(queries)}: {query[:60]}...")
        
        query_start = time.time()
        
        try:
            state = await asyncio.wait_for(
                run_recommendation(query),
                timeout=timeout_per_query
            )
            
            query_latency = time.time() - query_start
            latencies.append(query_latency)
            
            recommended_ids = [rec.model_id for rec in state.final_recommendations]
            status = "success"
            logger.info(f"[Benchmark] Query {idx} completed in {query_latency:.2f}s")
        
        except asyncio.TimeoutError:
            query_latency = time.time() - query_start
            latencies.append(query_latency)
            recommended_ids = []
            status = "timeout"
            logger.warning(f"[Benchmark] Query {idx} timed out")
        
        except Exception as e:
            query_latency = time.time() - query_start
            latencies.append(query_latency)
            recommended_ids = []
            status = "error"
            logger.error(f"[Benchmark] Query {idx} failed: {e}")
        
        per_query_results.append({
            "query": query,
            "latency": round(query_latency, 3),
            "recommended": recommended_ids,
            "status": status
        })
    
    # Latency stats only (no ground truth for custom queries)
    from backend.src.evaluation.metrics import latency_stats
    latency = latency_stats(latencies)
    
    total_time = time.time() - start_time
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "benchmark_duration": round(total_time, 2),
        "num_queries": len(queries),
        "latency": latency,
        "per_query_results": per_query_results
    }
    
    logger.info(
        f"[Benchmark] Custom benchmark complete. Duration: {total_time:.2f}s"
    )
    
    return report


def print_benchmark_report(report: dict) -> None:
    """
    Pretty-print benchmark report to console.
    
    Args:
        report: Report dict from run_benchmark or run_benchmark_subset
    """
    
    print("\n" + "=" * 80)
    print("BENCHMARK REPORT")
    print("=" * 80)
    print(f"Timestamp: {report.get('timestamp', 'N/A')}")
    print(f"Duration: {report.get('benchmark_duration', 0.0)}s")
    print(f"Queries: {report.get('num_queries', 0)}")
    
    metrics = report.get("metrics", {})
    if metrics:
        print("\n[METRICS]")
        
        precision = metrics.get("precision", {})
        if precision:
            for k, v in sorted(precision.items()):
                print(f"  Precision@{k}: {v:.3f}")
        
        recall = metrics.get("recall", {})
        if recall:
            for k, v in sorted(recall.items()):
                print(f"  Recall@{k}: {v:.3f}")
        
        mrr = metrics.get("mrr", 0.0)
        print(f"  MRR: {mrr:.3f}")
        
        latency = metrics.get("latency", {})
        if latency:
            print("\n[LATENCY]")
            print(f"  Mean: {latency.get('mean', 0.0):.3f}s")
            print(f"  Median: {latency.get('median', 0.0):.3f}s")
            print(f"  Min: {latency.get('min', 0.0):.3f}s")
            print(f"  Max: {latency.get('max', 0.0):.3f}s")
            print(f"  Stdev: {latency.get('stdev', 0.0):.3f}s")
    
    results = report.get("per_query_results", [])
    if results:
        print("\n[PER-QUERY RESULTS]")
        for i, result in enumerate(results, start=1):
            status = result.get("status", "unknown")
            latency = result.get("latency", 0.0)
            recs_count = len(result.get("recommended", []))
            query = result.get("query", "")[:60]
            print(f"  {i}. [{status}] {latency:.2f}s - {recs_count} recs - {query}")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run benchmark from CLI
    report = asyncio.run(run_benchmark(num_queries=3))
    print_benchmark_report(report)
