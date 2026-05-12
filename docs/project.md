# Agent-Based System for Language Model Recommendation

## Overview

This project implements a **deterministic multi-agent system** for recommending suitable language models based on user intent. The system addresses inefficiencies in model discovery by transforming natural language queries into structured requirements and performing controlled retrieval and ranking.

The system is built using the **Microsoft Agent Framework**, with strict role separation and reproducible execution.

The final system is exposed as a **chat-based application (ChatGPT-like UI)** backed by a Flask API.

---

## Problem

Existing model repositories suffer from:

- Weak keyword-based search
- No mapping between **user intent → model capabilities**
- Slow, manual selection process
- No explainability

---

## Objective

Design a system that:

1. Interprets natural language queries
2. Extracts structured constraints
3. Retrieves candidates from Qdrant
4. Ranks them using explicit Python scoring
5. Returns **top-K explainable recommendations**
6. Supports **interactive refinement via conversation**

---

## System Architecture

### Agents

#### 1. Supervisor
- Orchestrates pipeline execution
- Enforces deterministic flow
- Prevents loops

#### 2. Requirements Analyst
- Converts natural language into structured constraints:
  - task (e.g., summarization, QA, code)
  - constraints (latency, memory, license)
  - preferences (speed vs accuracy)

#### 3. Retriever
- Queries Qdrant (`localhost:6333`)
- Performs:
  - semantic search
  - metadata filtering

#### 4. Evaluator (Core)
- Python-only scoring (no LLM)
- Produces:
  - final score
  - full breakdown

#### 5. Synthesizer
- Generates explanation
- Drives **interactive clarification loop**
  - asks follow-up questions
  - refines recommendations

---

## Data Source (Strict Constraint)

- Only source: **Qdrant**
- Endpoint: `localhost:6333`

Forbidden:
- No external APIs
- No Hugging Face calls
- No hardcoded models
- No hallucinated data

---

## Data Representation (Critical for Copilot)

### Pipeline Overview

1. README parsing → section splitting  
2. Token-based chunking (with overlap)  
3. Embedding generation (SentenceTransformer)  
4. Storage in Parquet  
5. Upload to Qdrant  

---

### Embeddings

- Model: `BAAI/bge-large-en-v1.5`
- Output:
  - `float32` vectors
  - dimension = `embedding_dim`
- Stored as:
  - `List[float]` in Parquet
  - converted to list for Qdrant ingestion

---

### Parquet Schema (Source of Truth Before Qdrant)

Each row represents **one chunk of one model README**:


{
id: int,
vector: List[float],
model_id: str,
section_header: str,
section_index: int,
chunk_index: int,
num_sections: int,
text: str,

downloads: int,
likes: int,
tags: List[str],
pipeline_tag: str,
library_name: str,
created_at: str,
last_modified: str
}


Key properties:

- **Granularity**: chunk-level (not model-level)
- **Duplicates**: multiple rows per model
- **Vectors**: aligned with text chunks
- **Metadata**: repeated per chunk

---

### Qdrant Storage Format

Each Parquet row becomes a Qdrant point:


{
id: int,
vector: [float, float, ...],
payload: {
model_id: str,
section_header: str,
section_index: int,
text: str,

downloads: int,
likes: int,
tags: List[str],
pipeline_tag: str,
library_name: str,
created_at: str,
last_modified: str

}
}


Important:

- **Deduplication must happen at query time** (by `model_id`)
- Retrieval returns **sections**, not models directly
- Aggregation step required in Retriever/Evaluator

---

## Retrieval Strategy

### Method

- Cosine similarity (vector search)
- Top-N chunk retrieval
- Deduplicate by `model_id`
- Aggregate scores per model

### Metadata Filtering

- task (`pipeline_tag`)
- tags
- license (if available)
- hardware constraints (derived)

---

## Ranking (Core Contribution)

Implemented strictly in Python.

### Scoring Function


final_score =
w1 * similarity +
w2 * popularity +
w3 * recency +
w4 * hardware_fit +
w5 * license_match


### Components

- Similarity (from Qdrant)
- Popularity (downloads, likes)
- Recency (timestamps)
- Hardware fit (heuristic)
- License match

### Requirements

- Configurable weights
- Deterministic output
- Full logging

---

## Logging (Mandatory)

Each candidate must log:

- raw scores
- normalized scores
- weighted contributions
- final score

---

## Output Specification

### Top-K Models

For each:

- model_id
- final_score
- score breakdown

### Explanation

- why selected
- strengths
- weaknesses
- trade-offs

---

## Conversational Refinement (Key Feature)

The system is **not one-shot**.

After initial results:

- ask clarification questions:
  - "Do you prioritize speed or accuracy?"
  - "What hardware do you have?"
  - "Is commercial use required?"

- re-run ranking with updated constraints

---

## API Design (Flask)

### Endpoints

#### `POST /api/v1/recommend`
- Input:
  - query
  - optional conversation state
- Output:
  - top-K models
  - explanation
  - follow-up questions

---

#### `POST /api/v1/recommend/debug`
- Includes:
  - agent outputs
  - retrieval results
  - scoring logs

---

#### `GET /api/v1/health`
- Health check

---

## Chat-Based Interface (Final Use Case)

The system is exposed via a **chat UI**.

### Behavior

- ChatGPT-like interaction
- Maintains conversation state
- Iteratively refines recommendations

### Flow

1. User describes task
2. System returns top-K models
3. System asks clarifying questions
4. User responds
5. System re-ranks and updates results

---

### Example


User:
"I need a small model for summarization on CPU."

System:
Top 3 models:

Model A
Model B
Model C

Follow-up:
"Do you need real-time performance or batch processing?"

User:
"Real-time."

System:
→ re-rank with latency constraint
→ updated recommendations


---

## Evaluation (Mandatory)

### Metrics

- Precision@K
- Recall@K
- MRR
- nDCG
- ROUGE
- BLEU
- BERTScore
- Human evaluation
- Time-to-selection

---

## Baselines

- LLM-only recommendation
- Keyword search

---

## Non-Negotiable Constraints

- Only Qdrant as data source
- No LLM ranking
- Full logging required
- Deterministic pipeline
- Strict agent separation

---

## Summary

This system enforces a **hybrid design**:

- LLMs → interpretation + interaction
- Python → ranking + logic
- Qdrant → retrieval

Result:

- fast selection
- explainable outputs
- iterative refinement
- production-ready architecture