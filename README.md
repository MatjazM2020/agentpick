# AgentPick: Agent-based system for recommending the choice of a language model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-00a67e?logo=fastapi)](https://fastapi.tiangolo.com/)

**AgentPick** is a multi-agent system that recommends suitable language models based on natural language user intent. Instead of manually searching through hundreds of models, describe what you need and let our AI agents find the perfect match for your use case.

## 🎯 What AgentPick Does

Given a user query like *"I need a fast, lightweight model for summarizing legal documents,"* AgentPick:

1. **Analyzes** your intent and extracts structured constraints (task, performance requirements, memory limits)
2. **Retrieves** candidate models from a vector database using semantic search
3. **Ranks** recommendations using explicit, transparent scoring criteria
4. **Explains** why each model is recommended with detailed reasoning
5. **Refines** recommendations through interactive conversation

## 🏗️ Architecture Overview

AgentPick is built with a modular, production-ready architecture:

```
┌─────────────────────────────────────────────┐
│  Frontend (Open WebUI)                      │
│  Chat-based interface on port 3000          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  Backend API (FastAPI)                      │
│  - Requirements Analyst (LLM)               │
│  - Retriever (semantic search)              │
│  - Evaluator (deterministic scoring)        │
│  - Synthesizer (explanations & refinement)  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  Qdrant Vector Database (port 6333)         │
│  - Embeddings of 1000+ HuggingFace models   │
└─────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Frontend** | Interactive chat interface | Open WebUI (Svelte/SvelteKit) |
| **Backend** | Multi-agent pipeline | FastAPI (Python) + Microsoft Agent Framework |
| **Vector DB** | Model embeddings storage | Qdrant |
| **Data Pipeline** | Model metadata processing | Python, SentenceTransformers (BAAI/bge-large-en-v1.5) |

---

## 🚀 Quick Start (Docker)

### Prerequisites

- Docker & Docker Compose
- (Optional) OpenAI API key for enhanced recommendations

### Step 1: Clone and Setup

```bash
cd /path/to/agentpick_dev
export OPENAI_API_KEY="your-openai-key"  # Optional, but recommended
```

### Step 2: Start All Services

```bash
docker-compose up -d
```

This will start:
- **Qdrant** vector database (port 6333)
- **Backend API** (port 5000)
- **Frontend UI** (port 3000)

### Step 3: Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

You'll see the AgentPick chat interface. Start by typing in a natural language query about what model you need:

**Example queries:**
- *"I need a small language model for mobile inference"*
- *"Show me fast models for real-time translation"*
- *"Which models are best for code generation?"*
- *"I need an open-source model with commercial license"*

### Step 4: Monitor Services

Check service health:
```bash
# Check all services
docker-compose ps

# View logs from specific service
docker-compose logs -f backend    # Backend logs
docker-compose logs -f qdrant     # Qdrant logs
docker-compose logs -f ui         # Frontend logs
```

### Step 5: Stop Services

```bash
docker-compose down
```

To preserve data volumes:
```bash
docker-compose down --remove-orphans
```

To clean up everything (including volumes):
```bash
docker-compose down -v
```

---

## 📊 System Architecture Deep Dive

### Agent Pipeline

AgentPick uses a **deterministic multi-agent system** with strict role separation:

1. **Supervisor**: Orchestrates the entire pipeline and enforces deterministic flow
2. **Requirements Analyst**: Converts natural language into structured constraints
   - Task category (summarization, QA, code generation, etc.)
   - Performance constraints (latency, memory, throughput)
   - Preferences (open-source, commercial, specific license)
3. **Retriever**: Performs semantic search on the Qdrant vector database
4. **Evaluator**: Scores models using transparent, deterministic rules (no LLM used)
5. **Synthesizer**: Generates explanations and drives interactive refinement

### Data Pipeline

Model data flows through this process:

```
HuggingFace Hub
    ↓
Download metadata (README, tags, stats)
    ↓
Parse & chunk README content
    ↓
Generate embeddings (BAAI/bge-large-en-v1.5)
    ↓
Store in Qdrant vector database
    ↓
Available for semantic search
```

---

## 🔧 Configuration

### Environment Variables

Key configuration is set in `docker-compose.yaml`:

```yaml
environment:
  # Backend
  - OPENAI_CHAT_MODEL_ID=gpt-4o          # LLM for Requirements Analyst & Synthesizer
  - OPENAI_API_KEY=${OPENAI_API_KEY}     # Your OpenAI API key
  - QDRANT_URL=http://qdrant:6333        # Vector database endpoint
  
  # Frontend
  - OPENAI_API_BASE_URL=http://backend:5000/v1
  - WEBUI_SECRET_KEY=${WEBUI_SECRET_KEY}
```

### Custom Configuration

For advanced configuration, edit:
- **Backend**: `backend/app/main.py` (FastAPI initialization)
- **Frontend**: `UI/src/lib/config.ts` (UI settings)
- **Qdrant**: `docker-compose.yaml` (vector database settings)

---

## 📚 Project Structure

```
agentpick_dev/
├── README.md                          # This file
├── docker-compose.yaml                # Service orchestration
│
├── backend/                           # FastAPI backend
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                   # FastAPI app factory
│   │   ├── routes/                   # API endpoints
│   │   ├── schemas/                  # Request/response schemas
│   │   └── services/                 # Business logic
│   └── src/
│       ├── agents/                   # Agent implementations
│       ├── api/                      # API utilities
│       ├── core/                     # Core logic (retrieval, scoring)
│       ├── evaluation/               # Evaluation metrics
│       └── services/                 # External service integrations
│
├── UI/                                # Frontend (Open WebUI)
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── app.html                  # Main HTML
│   │   ├── routes/                   # Page components
│   │   └── lib/                      # Utilities & components
│   └── build/                        # Production build output
│
├── agentpick_data/                   # Data processing pipeline
│   ├── README.md
│   ├── requirements.txt
│   ├── src/hf_vectorizer/           # Model vectorization
│   │   ├── vectorizer.py            # Main pipeline
│   │   ├── hf_client.py             # HuggingFace API client
│   │   ├── embeddings.py            # Embedding generation
│   │   ├── parsing.py               # README parsing
│   │   └── storage.py               # Parquet storage
│   └── scripts/
│       ├── run_vectorize.sh         # Local execution
│       ├── start_qdrant.sh          # Qdrant startup
│       └── submit_vectorize.sh      # HPC job submission
│
└── docs/                              # Documentation
    ├── project.md                    # Detailed project overview
    └── agent_patterns.py             # Agent workflow patterns
```

---

## 🔌 API Reference

The backend exposes an OpenAI-compatible API on `http://localhost:5000/v1`.

### Chat Endpoint

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "agentpick",
    "messages": [
      {
        "role": "user",
        "content": "I need a model for question answering"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 2000
  }'
```

### Health Check

```bash
curl http://localhost:5000/health
```

---

## 🛠️ Development

### Local Backend Development

```bash
cd backend
pip install -r requirements.txt
python main.py
```

The backend will start on `http://localhost:5000`.

### Local Frontend Development

```bash
cd UI
npm install
npm run dev
```

The frontend will start on `http://localhost:5173` with hot reload.

### Data Vectorization (HuggingFace Models)

To generate new embeddings for model recommendations:

```bash
cd agentpick_data
pip install -r requirements.txt

# Start Qdrant
./scripts/start_qdrant.sh

# Run vectorization
./scripts/run_vectorize.sh
```

---

## 📝 API Documentation

Once services are running, access interactive API docs:

- **Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

---

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/
```

### Frontend Tests

```bash
cd UI
npm run test:frontend
```

### E2E Tests (Cypress)

```bash
cd UI
npm run cy:open
```

---

## 📋 Troubleshooting

### Services won't start

```bash
# Check Docker daemon
docker ps

# View detailed logs
docker-compose logs

# Rebuild images
docker-compose build --no-cache
```

### Backend can't connect to Qdrant

```bash
# Verify Qdrant is running
docker-compose ps | grep qdrant

# Check Qdrant health
curl http://localhost:6333/health
```

### Frontend shows errors

```bash
# Clear browser cache and refresh
# Or, restart with fresh volumes
docker-compose down -v
docker-compose up -d
```

### Out of memory

Increase Docker memory allocation in Docker Desktop settings, then restart:
```bash
docker-compose restart
```

---

## 🔐 Security Notes

⚠️ **Production Deployment:**

1. Change `WEBUI_SECRET_KEY` in `.env`:
   ```bash
   WEBUI_SECRET_KEY=$(openssl rand -hex 32)
   ```

2. Set strong `QDRANT_API_KEY`

3. Use environment variables for sensitive data (never commit `.env`)

4. Enable HTTPS/TLS in production

5. Restrict API access with authentication

---

## 📖 Further Reading

- [Project Overview](docs/project.md) - Detailed architecture and design decisions
- [Agent Patterns](docs/agent_patterns.py) - Implementation patterns for the multi-agent system
- [Data Vectorization](agentpick_data/README.md) - HuggingFace model embedding generation
- [Workflow Patterns](docs/workflow_patterns.py) - Common usage workflows

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 💬 Support

For issues, questions, or suggestions:

1. Check existing [GitHub Issues](https://github.com/your-repo/issues)
2. Review the [Troubleshooting](#troubleshooting) section
3. Open a new issue with:
   - Clear description
   - Steps to reproduce
   - Docker logs output
   - System information (OS, Docker version)

---

## 🎓 Citation

If you use AgentPick in research, please cite:

```bibtex
@software{agentpick2024,
  title={AgentPick: Deterministic Multi-Agent System for Language Model Recommendation},
  year={2024},
  url={https://github.com/your-repo}
}
```

---

**Last Updated:** May 2024
**Status:** Active Development
