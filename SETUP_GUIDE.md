# AgentPick Application Setup Guide

## Architecture Overview

Your application uses a **multi-component architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Your Browser                              │
│                  localhost:8080 (single port)                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP/WebSocket
                      │
        ┌─────────────┴────────────────────────────┐
        │                                          │
   ┌────▼──────────────┐              ┌──────────▼──────────────┐
   │   Open WebUI      │              │   AgentPick Backend    │
   │   Frontend        │              │   (OpenAI-compatible)  │
   │   & Backend       │──────────┐   │                        │
   │   (port 8080)     │          │   │   (port 5000)          │
   └─────────────────────┘         │   └────────────────────────┘
                                   │
                    Uses for chat completions
```

## Component Details

### 1. **AgentPick Custom Backend** (`/backend`)
- **Technology**: FastAPI + Uvicorn
- **Port**: 5000 (internally), exposed as 5002 (localhost)
- **Purpose**: OpenAI-compatible recommendation engine
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /v1/models` - Returns available models
  - `POST /v1/chat/completions` - Main chat completions endpoint

### 2. **Open WebUI Frontend & Backend** (`/UI`)
- **Technology**: Svelte frontend + Python backend (FastAPI)
- **Port**: 8080
- **Purpose**: 
  - Serves the interactive web UI
  - Acts as API gateway
  - Can orchestrate multiple LLM backends
- **Configuration**: Uses your custom backend as OpenAI provider

### 3. **Vector Database** (Qdrant)
- **Port**: 6333-6334
- **Purpose**: Stores embeddings for RAG (Retrieval Augmented Generation)

## How It Works

1. **User accesses** `http://localhost:8080`
2. **Open WebUI backend** serves the Svelte frontend
3. **Frontend initializes** by calling `/api/config` to load configuration
4. **User sends a message** in the Web UI
5. **Open WebUI backend** routes the chat request to your custom backend at `http://backend:5000/v1`
6. **Custom backend** processes the recommendation and returns results
7. **Results** are displayed in the UI

## Running the Application

### Prerequisites
- Docker and Docker Compose installed
- Port 8080 available on your machine

### Start Everything

```bash
cd /Users/matjazmadon/Development/agentpick_dev

# Full rebuild
docker-compose down && docker rmi agentpick-backend:latest agentpick-ui:latest
docker-compose up -d --build

# Or just restart (faster if images exist)
docker-compose restart
```

### Access the Application

**Web UI**: http://localhost:8080

You should see the Open WebUI interface. If you see the error about "frontend only", the backend isn't responding yet. Wait a few seconds and refresh.

### Monitor Services

```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs -f ui          # Open WebUI logs
docker-compose logs -f backend     # AgentPick backend logs
docker-compose logs -f qdrant      # Qdrant logs

# Stop services
docker-compose down
```

## Troubleshooting

### Error: "Open WebUI Backend Required"

This means the frontend couldn't load the configuration from the Open WebUI backend:

1. **Check services are running**:
   ```bash
   docker-compose ps
   ```
   All should show "Up"

2. **Check backend is healthy**:
   ```bash
   # From your machine (not inside container)
   curl http://localhost:5002/health
   # Should return: {"status": "ok", "service": "agentpick-recommendation-api"}
   ```

3. **Check Open WebUI logs**:
   ```bash
   docker-compose logs ui | tail -50
   ```

4. **Wait longer**: The UI takes 30-40 seconds to start. Wait and refresh.

### Error: Connection refused

Ensure you're using the correct ports:
- **Web UI**: http://localhost:8080
- **Custom Backend** (for direct testing): http://localhost:5002
- **Qdrant** (for direct testing): http://localhost:6333

### Custom Backend Not Responding

Check that the custom backend is running:

```bash
# Test health check
curl http://localhost:5002/health

# Test models endpoint
curl http://localhost:5002/v1/models

# View backend logs
docker-compose logs backend | tail -50
```

## Environment Variables

### Open WebUI (Docker Compose)
- `OPENAI_API_BASE_URL=http://backend:5000/v1` - Your custom backend
- `OPENAI_API_KEY=sk-local-key` - Dummy key for local dev
- `PORT=8080` - Server port
- `WEBUI_SECRET_KEY` - Session secret (set in docker-compose or .env)

### AgentPick Backend (Docker Compose)
- `PORT=5000` - FastAPI server port
- `HOST=0.0.0.0` - Bind to all interfaces
- `QDRANT_HOST=qdrant` - Vector DB host
- `QDRANT_PORT=6333` - Vector DB port

## Development Notes

### Frontend Structure (`UI/src`)
- `/routes` - Page components
- `/lib/apis` - API calls to Open WebUI backend
- `/lib/stores` - Svelte stores (config, auth, etc.)
- `/lib/i18n` - Internationalization

### Backend Structure (`backend`)
- `/app/main.py` - FastAPI factory
- `/app/routes` - Endpoints (chat, models, health)
- `/app/schemas` - Pydantic validation
- `/app/services` - Business logic (recommendation adapter)
- `/src` - Core recommendation engine

## Next Steps

1. ✅ Start the application: `docker-compose up -d --build`
2. ✅ Access Web UI: http://localhost:8080
3. ✅ Verify custom backend is being used
4. 📝 Configure LLM providers in the Web UI settings
5. 🧪 Test chat functionality
6. 📊 Monitor Qdrant for embeddings

---

**Need Help?** Check Docker logs first with `docker-compose logs` for detailed error messages.
