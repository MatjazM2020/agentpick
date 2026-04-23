# Quick Reference: Custom Recommendation API Integration

## 🚀 Get Started in 30 Seconds

### 1. Ensure backend is running:
```bash
cd /Users/matjazmadon/Development/agentpick/backend
python -m src.api.app  # Runs on localhost:5000
```

### 2. Use in Open WebUI - Just Chat

Every message automatically goes to the custom backend:
```
User: I need a small embedding model for CPU
↓
Automatically sent to backend
↓
Response appears in chat
```

No model selection needed. No special markers required.

### 3. See recommendations appear in chat
- Response includes model rankings with scores
- Shows score breakdown and metadata
- Formatted as readable text in chat

---

## 📁 What Was Created

| File | Purpose |
|------|---------|
| `UI/src/lib/apis/recommendations/index.ts` | Core API service (150+ lines) |
| `UI/src/lib/components/chat/Chat.svelte` | Chat integration (~60 lines added) |
| `UI/src/lib/apis/index.ts` | API exports (8 lines added) |

---

## 🔧 Code Snippets

### Use recommendations API directly:
```typescript
import { fetchRecommendations } from '$lib/apis';

const response = await fetchRecommendations('small LLM');
console.log(response.recommendations);
```

### Format response as text:
```typescript
import { formatRecommendationResponse } from '$lib/apis';

const text = formatRecommendationResponse(response);
// Returns: "Found **3** recommendations:\n\n**1. model-id**\n- Score: ..."
```

### Call refinement endpoint:
```typescript
import { refineRecommendations } from '$lib/apis';

const refined = await refineRecommendations(
  'make it faster',
  previousResponse.state
);
```

---

## 🔗 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /api/v1/recommend` | Flask | Get recommendations |
| `POST /api/v1/recommend/refine` | Flask | Refine recommendations |
| `GET /api/v1/health` | Flask | Health check |

---

## 🎯 Behavior

### When message is sent:
```
User message
  ↓
ALWAYS: Send to custom backend
  ↓
POST localhost:5000/api/v1/recommend
  ↓
Format response & display in chat
```

### Response flow:
```
fetch() → Parse JSON → Check status → Format → Display in chat
            ↓                              ↓
     Error handling              Markdown formatting with scores
```

**No detection. No fallback. No alternatives.**

---

## ⚙️ Configuration

### Change backend URL:
Edit `UI/src/lib/apis/recommendations/index.ts`, line 9:
```typescript
const RECOMMENDATION_API_BASE_URL = 'http://your-server:5000/api/v1';
```

### All messages always use backend
No detection logic to configure. Every message routes to custom backend.

### Custom response formatting:
Edit `UI/src/lib/apis/recommendations/index.ts` function `formatRecommendationResponse()` to style differently.

---

## 🧪 Quick Test

### Terminal 1 - Start backend:
```bash
cd backend && python -m src.api.app
# Listens on localhost:5000
```

### Terminal 2 - Start UI (or use existing):
```bash
cd UI && npm run dev
# or use existing Open WebUI instance
```

### In UI Browser:
1. Type: `!recommend: fast image classification model`
2. Watch it call custom backend instead of OpenAI
3. See recommendations formatted in chat

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot reach server" | Start Flask backend: `python -m src.api.app` |
| Response not formatted | Check `formatRecommendationResponse` returns string |
| Using wrong backend | Verify model name has "recommend" or message has "!recommend:" |
| TypeScript errors | Restart TS server in VS Code |

---

## 📊 Response Example

**Request:**
```json
{"query": "small embedding model"}
```

**Response in Chat:**
```
Found **3** recommendations:

**1. distilbert-base**
- Score: 0.9500
- Score Breakdown:
  - semantic_similarity: 0.9200
  - popularity: 0.8800

**2. sentence-transformers**
- Score: 0.8900
...
```

---

## 🎁 Bonus Features (Optional)

### Refinement (NOT YET integrated):
```typescript
// Available to use but needs UI integration
const refined = await refineRecommendations(query, previousState);
```

### Streaming (Future enhancement):
Currently returns full response. Could be extended to stream like OpenAI.

### Custom formatting:
Replace default markdown with JSON, tables, or charts.

---

## 📖 Full Documentation

See `UI/RECOMMENDATION_API_INTEGRATION.md` for complete details.

---

**Status**: ✅ Ready to use
**Backend URL**: `http://localhost:5000/api/v1`
**Detection**: Model name or `!recommend:` prefix
