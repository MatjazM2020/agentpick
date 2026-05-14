# AgentPick Backend Architecture

## Overview

The AgentPick backend is a FastAPI-based recommendation engine that provides an OpenAI-compatible API for intelligent model recommendations. It uses a multi-agent orchestration system powered by the `agent-framework` library to analyze user requirements and suggest optimal models based on vectorized semantic search and deterministic scoring.

## Technology Stack

- **Framework**: FastAPI with Uvicorn
- **Python Version**: 3.11
- **Agent Framework**: `agent-framework` (OpenAI Chat Completions)
- **Vector Database**: Qdrant (semantic search)
- **Embeddings**: sentence-transformers
- **ML/Math**: numpy, scipy, scikit-learn
- **Environment**: python-dotenv
- **HTTP Client**: httpx, requests

## Application Structure

```
backend/
├── main.py                 # Entry point (starts uvicorn server)
├── run.py                  # Alternative runner
├── requirements.txt        # Python dependencies
├── Dockerfile              # Multi-stage Docker build
│
├── app/                    # FastAPI application layer
│   ├── __init__.py
│   ├── main.py             # FastAPI factory & app creation
│   ├── routes/             # HTTP endpoint handlers
│   │   ├── chat.py         # POST /v1/chat/completions
│   │   ├── models.py       # GET /v1/models
│   │   └── health.py       # GET /health
│   ├── schemas/            # Pydantic request/response models
│   │   ├── openai_chat.py  # OpenAI ChatCompletionRequest/Response
│   │   └── openai_models.py
│   └── services/           # API adapter layer
│       └── recommendation_adapter.py  # OpenAI <-> internal format conversion
│
└── src/                    # Core business logic
    ├── agents/             # LLM agent implementations
    │   ├── supervisor.py          # Orchestration & pipeline control
    │   ├── requirements_analyst.py # Parse user query → structured requirements
    │   ├── retriever.py            # Semantic search for candidates
    │   ├── evaluator.py            # Deterministic model scoring
    │   ├── evaluator_scoring.py    # Scoring algorithms & heuristics
    │   ├── synthesizer.py          # Generate explanations for recommendations
    │   └── refinement_advisor.py   # Quality assessment & refinement decisions
    │
    ├── core/               # Core infrastructure
    │   ├── agent_factory.py    # Agent instantiation & initialization
    │   ├── llm.py              # LLM client management
    │   ├── state.py            # RecommendationState dataclass
    │   ├── config.py           # Configuration classes (Agent, Scoring, Retriever)
    │   └── query_specificity.py # Query refinement heuristics
    │
    ├── api/                # API infrastructure (reserved)
    │   └── __init__.py
    │
    ├── services/           # Service layer
    │   └── recommendation_pipeline.py  # Pipeline orchestration & state management
    │
    └── evaluation/         # Evaluation & monitoring
        ├── metrics.py      # Performance metrics
        └── benchmarks.py   # Benchmark suite
```

## Core Components

### 1. HTTP API Layer (`app/`)

#### FastAPI Application Factory (`app/main.py`)
- Creates and configures the FastAPI app
- Sets up CORS middleware (allows all origins for Open WebUI compatibility)
- Registers route handlers
- Manages application lifecycle (startup/shutdown)
- Logging configuration

**Key Routes:**
- `POST /v1/chat/completions` - Main recommendation endpoint
- `GET /v1/models` - List available models
- `GET /health` - Health check

#### Chat Completions Endpoint (`app/routes/chat.py`)
- Entry point for recommendation requests
- Accepts OpenAI-format `ChatCompletionRequest`
- Validates input messages
- Calls `run_recommendation()` from the service layer
- Converts internal state to `ChatCompletionResponse`
- Returns OpenAI-compatible JSON

#### Request/Response Schemas (`app/schemas/`)
- `ChatCompletionRequest` - OpenAI format with messages, model, optional parameters
- `ChatCompletionResponse` - OpenAI format with choices and usage metadata
- Pydantic models for validation and serialization

#### Recommendation Adapter (`app/services/recommendation_adapter.py`)
- Bridges OpenAI API format and internal format
- Extracts user query from OpenAI message list
- Extracts conversation context
- Converts `RecommendationState` to OpenAI response format
- Handles SSE (Server-Sent Events) streaming if needed

---

### 2. Service Layer (`src/services/`)

#### Recommendation Pipeline (`recommendation_pipeline.py`)
- **Entry point** for all recommendation requests
- Handles both **new requests** and **refinement requests**
- Manages `RecommendationState` lifecycle
- Initializes and orchestrates agents
- Delegates to `supervisor.run_pipeline()`

**Key Function:**
```python
async def run_recommendation(
    query: str,
    state: Optional[RecommendationState] = None,
    conversation_text: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    ...
) -> RecommendationState
```

**Flow:**
1. Determine if request is new or refinement
2. Create or update state
3. Initialize agents
4. Call supervisor pipeline
5. Return updated state with recommendations

---

### 3. Agent Orchestration (`src/agents/`)

#### Supervisor (`supervisor.py`)
- **Orchestrates the entire pipeline** with autonomous decision-making
- Implements **bounded autonomy** (max 2 refinement iterations)
- Manages the agentic loop with quality heuristics

**Pipeline Stages:**

```
1. Requirements Analyst
   └─> Extract structured requirements from natural language query
   
2. Retriever
   └─> Semantic search against model candidate database
   
3. Evaluator
   └─> Deterministic scoring of candidates
   
4. [Conditional] Refinement Loop (max 2 iterations)
   ├─> Refinement Advisor: assess quality
   ├─> If quality < threshold:
   │   ├─> Requirements Analyst (optional: re-analyze with hints)
   │   ├─> Retriever (with relaxed filters)
   │   └─> Evaluator (re-score)
   └─> Else: proceed to synthesis
   
5. Synthesizer
   └─> Generate explanations for recommendations
```

**Quality Heuristics:**
- Top recommendation score below threshold
- Low variance in scores (insufficient differentiation)
- Too few candidates retrieved
- Missing critical requirement slots

**Bounded Autonomy:**
- Maximum 2 refinement iterations
- Decision logging for all refinement triggers
- Quality metrics recorded

#### Requirements Analyst (`requirements_analyst.py`)
- Parses user query into **structured requirements**
- Uses OpenAI Chat Completions (via agent-framework)
- Extracts key parameters (model size, task type, hardware, latency, accuracy, etc.)
- Can be called with optional `hint` text for refinement context

#### Retriever (`retriever.py`)
- Performs **semantic search** against model candidate database
- Uses Qdrant vector database
- Embeds query/requirements using sentence-transformers
- Supports filtering and scoring parameters
- Returns ranked list of candidate models

#### Evaluator (`evaluator.py`, `evaluator_scoring.py`)
- **Deterministic scoring** of candidate models
- Implements scoring algorithms in `evaluator_scoring.py`
- Scores based on:
  - Requirement alignment
  - Performance metrics (latency, accuracy, throughput)
  - Resource constraints (memory, CPU, GPU)
  - User preferences
- Returns scored and ranked recommendations

#### Refinement Advisor (`refinement_advisor.py`)
- Assesses quality of current recommendations
- Determines if refinement is needed
- Recommends refinement strategy
- Triggered by quality metrics below threshold

#### Synthesizer (`synthesizer.py`)
- Generates **explanations** for recommendations
- Uses OpenAI Chat Completions
- Produces human-readable justifications
- Explains why each model is recommended

---

### 4. Core Infrastructure (`src/core/`)

#### State Management (`state.py`)
- `RecommendationState` dataclass
- Tracks entire recommendation lifecycle
- Fields:
  - `user_query` - Latest user query
  - `conversation_text` - Full conversation context
  - `messages` - Message history
  - `structured_requirements` - Parsed requirements
  - `candidate_models` - Retrieved candidates
  - `scored_candidates` - Ranked & scored models
  - `final_recommendations` - Top N recommendations
  - `explanations` - Synthesized explanations
  - `iteration` - Refinement iteration count
  - `refinement_assistant_text` - Refinement advisor feedback
  - `stopped_for_query_refinement` - User clarification needed flag

#### Agent Factory (`agent_factory.py`)
- Centralizes agent instantiation
- Creates `OpenAIChatClient` with configuration
- Initializes agents with consistent settings
- Handles environment variable configuration:
  - `OPENAI_API_KEY` - API key for LLM calls
  - `OPENAI_CHAT_MODEL_ID` - Model to use (default: `gpt-5.4-nano`)
  - `OPENAI_BASE_URL` - Optional custom endpoint

#### Configuration (`config.py`)
- `AgentConfig` - Agent behavior settings
- `ScoringConfig` - Scoring algorithm parameters
- `RetrieverConfig` - Semantic search parameters
- Defaults provided; can be overridden per request

#### LLM Client (`llm.py`)
- OpenAI Chat Completions client management
- Message formatting
- Token counting (if needed)

#### Query Specificity (`query_specificity.py`)
- Heuristics for query refinement decisions
- Missing requirement slot detection
- Fallback refinement text generation
- Determines if user should clarify intent

---

### 5. Evaluation & Monitoring (`src/evaluation/`)

#### Metrics (`metrics.py`)
- Performance metrics (latency, throughput, accuracy)
- Quality metrics (coverage, relevance)
- Recommendation diversity metrics

#### Benchmarks (`benchmarks.py`)
- Benchmark test suite
- Evaluates recommendation quality
- Measures end-to-end latency
- Validates result consistency

---

## Data Flow

### New Recommendation Request

```
User Query (OpenAI format)
       ↓
[POST /v1/chat/completions]
       ↓
extract_user_query()  ← extract from OpenAI messages
       ↓
run_recommendation(query)
       ↓
Create fresh RecommendationState
       ↓
Initialize Agents
       ↓
supervisor.run_pipeline()
  ├─ requirements_analyst.run()        → RecommendationState.structured_requirements
  ├─ retriever.run()                   → RecommendationState.candidate_models
  ├─ evaluator.run()                   → RecommendationState.scored_candidates
  ├─ [refinement loop]
  │  ├─ refinement_advisor.run()       → decision
  │  └─ [if needed] re-run retriever & evaluator
  └─ synthesizer.run()                 → RecommendationState.explanations
       ↓
RecommendationState (with final_recommendations)
       ↓
state_to_openai_response()  ← convert to OpenAI format
       ↓
ChatCompletionResponse
       ↓
Return JSON to Client
```

### Refinement Request

```
User Refinement Query
       ↓
run_recommendation(query, state=previous_state)
       ↓
Append to state.messages
Increment state.iteration
       ↓
supervisor.run_pipeline(state, ...)  ← pipeline continues with same state
       ↓
Updated RecommendationState
       ↓
Return to Client
```

---

## Configuration & Environment

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | 5000 | Server port |
| `HOST` | 0.0.0.0 | Server host |
| `ENV` | production | development/production mode (reload if dev) |
| `LOG_LEVEL` | INFO | Logging level |
| `OPENAI_API_KEY` | (required) | API key for LLM calls |
| `OPENAI_CHAT_MODEL_ID` | gpt-5.4-nano | Model for agent calls |
| `OPENAI_BASE_URL` | (optional) | Custom OpenAI-compatible endpoint |

### Configuration Classes

**AgentConfig** - Controls agent behavior
- Timeouts, retries, temperature, etc.

**ScoringConfig** - Scoring parameters
- Weights for different scoring factors
- Thresholds for quality gates

**RetrieverConfig** - Vector search parameters
- Number of candidates to retrieve
- Similarity threshold
- Filtering options

---

## API Compatibility

### OpenAI Chat Completions Format

The backend implements OpenAI's Chat Completions API format:

**Request:**
```json
{
  "model": "agentpick-recommendation",
  "messages": [
    {"role": "user", "content": "I need a small, fast model for text classification"}
  ],
  "temperature": 0.7,
  "max_tokens": 2000
}
```

**Response:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "agentpick-recommendation",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Based on your requirements, I recommend: [recommendations with explanations]"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 200,
    "total_tokens": 250
  }
}
```

### Supported Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/v1/chat/completions` | Get recommendations |
| GET | `/v1/models` | List available recommendation models |
| GET | `/health` | Health check |

---

## Docker Deployment

### Multi-Stage Build

**Stage 1: Builder**
- Base: Python 3.11-slim
- Installs build dependencies
- Installs Python packages

**Stage 2: Runtime**
- Base: Python 3.11-slim
- Copies packages from builder
- Copies application code
- Creates non-root user (appuser, UID 1000)
- Configures health check
- Exposes port 5000

### Environment Setup in Container

```dockerfile
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV HOST=0.0.0.0
ENV ENV=production
```

---

## Key Design Patterns

### 1. **Agentic Loop with Bounded Autonomy**
- Agents make decisions autonomously
- Bounded by max 2 refinement iterations
- Quality heuristics prevent infinite loops
- Decision logging for observability

### 2. **Structured State Management**
- Single `RecommendationState` object
- Immutable transitions through pipeline
- Supports multi-turn conversations
- Easy to serialize/persist

### 3. **Agent-Framework Abstraction**
- Uses `agent-framework` library
- Consistent agent interface
- Easy to swap LLM implementations
- Testable agent logic

### 4. **OpenAI API Compatibility**
- Implements standard OpenAI format
- Easy integration with existing tools (Open WebUI)
- Natural switching from OpenAI to local models
- Standard schema validation with Pydantic

### 5. **Deterministic Scoring**
- Reproducible recommendations
- Explainable decision logic
- No randomness in scoring phase
- Separate from LLM (which may be stochastic)

### 6. **Adapter Pattern**
- HTTP layer (OpenAI format) ← Adapter → Business logic (internal format)
- Clean separation of concerns
- Easy to support multiple API formats
- Testable in isolation

---

## Scalability & Performance

### Current Constraints
- Single-process Uvicorn server
- In-memory state management
- Sequential agent execution

### Scaling Opportunities
1. **Horizontal Scaling**
   - Multiple FastAPI instances behind load balancer
   - Distribute state to persistent store (Redis, DB)

2. **Async Optimization**
   - Parallelize agent stages (retriever + evaluator)
   - Concurrent LLM calls where possible

3. **Caching**
   - Cache embeddings for repeated queries
   - Cache scored candidates for similar requirements

4. **Vector DB Optimization**
   - Partition candidate database
   - Index by requirement types
   - Pre-compute common searches

---

## Error Handling & Resilience

### API Layer
- Pydantic validation catches invalid requests
- HTTP exceptions with descriptive messages
- Request validation before pipeline execution

### Agent Layer
- Try/catch around agent calls
- Fallback strategies if agent fails
- Quality gates trigger refinement

### Pipeline Layer
- State rollback on refinement
- Bounded iteration prevents infinite loops
- Logging for debugging

---

## Testing & Quality Assurance

### Evaluation Module
- `evaluation/metrics.py` - Measures recommendation quality
- `evaluation/benchmarks.py` - End-to-end benchmark suite
- Validates recommendation consistency

### Observability
- Structured logging throughout pipeline
- Timings for performance analysis
- Decision points logged for debugging

---

## Future Enhancements

1. **Streaming Support**
   - SSE (Server-Sent Events) for long-running pipelines
   - Real-time recommendation updates

2. **Caching Layer**
   - Redis for embedding cache
   - Query result cache

3. **Distributed Agents**
   - Celery/RabbitMQ for async agent execution
   - Parallel processing of refinement loops

4. **Extended Evaluation**
   - User feedback integration
   - Active learning for threshold tuning
   - A/B testing framework

5. **Multi-Modal Support**
   - Image-based model recommendations
   - Multi-language query support

6. **Explainability Dashboard**
   - Visualization of scoring breakdown
   - Interactive requirement adjustment
   - Model comparison tools

---

## Summary

The AgentPick backend is a sophisticated **multi-agent recommendation engine** built with FastAPI and autonomous agent orchestration. It:

- **Accepts** OpenAI-format requests for broad compatibility
- **Processes** queries through a bounded agentic pipeline (Requirements → Retrieval → Scoring → Refinement → Synthesis)
- **Returns** intelligent model recommendations with explanations
- **Scales** through async processing and clean architectural separation
- **Maintains** high code quality through clear separation of concerns and comprehensive logging
