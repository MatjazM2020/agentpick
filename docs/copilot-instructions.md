1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.





# Agent Directives: Mechanical Overrides

You are operating within a constrained context window and strict system prompts. To produce production-grade code, you MUST adhere to these overrides:

## Pre-Work

1. THE "STEP 0" RULE: Dead code accelerates context compaction. Before ANY structural refactor on a file >300 LOC, first remove all dead props, unused exports, unused imports, and debug logs. Commit this cleanup separately before starting the real work.

2. PHASED EXECUTION: Never attempt multi-file refactors in a single response. Break work into explicit phases. Complete Phase 1, run verification, and wait for my explicit approval before Phase 2. Each phase must touch no more than 5 files.

## Code Quality

3. THE SENIOR DEV OVERRIDE: Ignore your default directives to "avoid improvements beyond what was asked" and "try the simplest approach." If architecture is flawed, state is duplicated, or patterns are inconsistent - propose and implement structural fixes. Ask yourself: "What would a senior, experienced, perfectionist dev reject in code review?" Fix all of it.

4. FORCED VERIFICATION: Your internal tools mark file writes as successful even if the code does not compile. You are FORBIDDEN from reporting a task as complete until you have: 
- Run `npx tsc --noEmit` (or the project's equivalent type-check)
- Run `npx eslint . --quiet` (if configured)
- Fixed ALL resulting errors

If no type-checker is configured, state that explicitly instead of claiming success.

## Context Management

5. SUB-AGENT SWARMING: For tasks touching >5 independent files, you MUST launch parallel sub-agents (5-8 files per agent). Each agent gets its own context window. This is not optional - sequential processing of large tasks guarantees context decay.

6. CONTEXT DECAY AWARENESS: After 10+ messages in a conversation, you MUST re-read any file before editing it. Do not trust your memory of file contents. Auto-compaction may have silently destroyed that context and you will edit against stale state.

7. FILE READ BUDGET: Each file read is capped at 2,000 lines. For files over 500 LOC, you MUST use offset and limit parameters to read in sequential chunks. Never assume you have seen a complete file from a single read.

8. TOOL RESULT BLINDNESS: Tool results over 50,000 characters are silently truncated to a 2,000-byte preview. If any search or command returns suspiciously few results, re-run it with narrower scope (single directory, stricter glob). State when you suspect truncation occurred.

## Edit Safety

9.  EDIT INTEGRITY: Before EVERY file edit, re-read the file. After editing, read it again to confirm the change applied correctly. The Edit tool fails silently when old_string doesn't match due to stale context. Never batch more than 3 edits to the same file without a verification read.

10. NO SEMANTIC SEARCH: You have grep, not an AST. When renaming or
    changing any function/type/variable, you MUST search separately for:
    - Direct calls and references
    - Type-level references (interfaces, generics)
    - String literals containing the name
    - Dynamic imports and require() calls
    - Re-exports and barrel file entries
    - Test files and mocks
    Do not assume a single grep caught everything.




# State-Shift Service - Claude Code Instructions

## Project Overview

Production financial data processing system that:
- Analyzes cryptocurrency market data to detect state shifts and trading patterns
- Processes 515+ symbols across multiple timeframes (5m, 15m, 60m)
- Uses 5-phase modular architecture with Redis caching and PostgreSQL storage
- Runs in Docker on VPS, triggered by data ingestion events

**Target Performance**: <10s per cycle | <500MB memory | 8 CPU cores

## Directory Structure

```
state-shift-service-prod/
├── main.py                    # Entry point (Docker CMD)
├── core/                      # Supporting infrastructure
│   ├── state_management/      # Redis state manager
│   ├── threshold_cache_loader.py  # Percentile thresholds (Redis-cached)
│   ├── redis_publisher.py     # Publishes results to Redis
│   ├── ingestion_event_listener.py  # Triggers processing cycles
│   └── helper_functions.py    # Utilities, TradeDefinition class
├── phases/                    # 5-phase processing pipeline
│   ├── phase1_indicators.py   # Technical indicators
│   ├── phase2_symbol_processing.py  # Symbol classification
│   ├── phase3_market_aggregation.py # Market-level aggregation
│   ├── phase4_regime_analysis.py    # Regime + symbol-market matching
│   └── phase5_trading.py      # Trade execution
├── processors/                # Modular processors
│   ├── indicators/            # Indicator calculations
│   ├── symbols/               # Trend/Vol/Orderflow classifiers
│   └── market/                # Market aggregators
├── trading/                   # Trading system
│   ├── orchestration/         # Trade idea generation
│   ├── execution/             # Account management, order execution
│   └── analytics/             # Performance tracking
├── configs/
│   └── setups/                # 40+ trade setup definitions
└── regime/                    # Market regime classification
```

## 5-Phase Architecture

### Phase 1: Indicators (`phase1_indicators.py`)
- Calculates technical indicators (RSI, MA, Bollinger, etc.)
- Input: Raw OHLCV from `ss_m1_indicators`
- Output: Updated indicator columns in same table
- Mode: Incremental (only new/changed rows)

### Phase 2: Symbol Processing (`phase2_symbol_processing.py`)
- **3 parallel processors**:
  - `IncrementalTrendProcessor` - Trend scores per window
  - `IncrementalVolatilityProcessor` - Volatility scores
  - `IncrementalOrderflowProcessor` - OI-based orderflow scores
- Output: `ss_token_*_classification_windowed` tables
- Each row has: `symbol, open_time, {metric}_5, {metric}_15, {metric}_60`

### Phase 3: Market Aggregation (`phase3_market_aggregation.py`)
- **4 parallel aggregators**:
  - Trend, Volatility, Orderflow (aggregate symbol scores to market level)
  - SyntheticPrice (equal + volume-weighted market price)
- Output: `ss_market_*_states` tables
- Provides market context for symbol-market matching

### Phase 4: Regime Analysis (`phase4_regime_analysis.py`)
- **Market Regime Classification**: Detects macro states (bullish/bearish/neutral)
  - Output: `market_regime_classification`, `market_regime_events`
- **Symbol-Market Matching**: Core trade signal generation
  - Loads setup rules from `configs/setups/`
  - Evaluates: trend_match, vol_match, ofc_match
  - Output: Trade ideas with match strength

### Phase 5: Trading (`phase5_trading.py`)
- **Optional** (`enable_trading=True/False`)
- Components:
  - `AccountManager` - Multi-account (internal simulated + prop firm live)
  - `TradeIdeaOrchestrator` - Generates ideas per setup
  - `TradeCloseSystem` - SL/TP/trailing stop management
  - `PositionMonitor` - Real-time position tracking
- Output: `ss_trades_open`, `ss_trades_closed`

## Trade Setup Configuration

Each setup lives in `configs/setups/{setup_name}/` with these files:

### `setup_config.py` - Metadata
```python
SETUP_TYPE = "multi_window"  # or "single_window"
SETUP_METADATA = {
    "name": "my_setup_v1",
    "description": "Description",
    "version": "1.0.0",
    "enabled": True
}
```

### `trade_settings.py` - Execution Parameters
```python
from core.helper_functions import TradeDefinition

TRADE_SETUPS = {
    "my_setup_v1": TradeDefinition(
        name="my_setup_v1",
        tp_pct=0.025,           # 2.5% take profit
        sl_pct=0.015,           # 1.5% stop loss
        max_hold_minutes=180,   # 3 hour max hold
        trailing_stop_enabled=True,
        trail_activation_pct=0.01,
        trail_distance_pct=0.01,
    )
}
```

### `symbol_market_matching_rules.py` - Matching Logic
```python
def trend_match(score, delta, state, market_score, market_delta, market_state, window):
    """Returns: 1 (STRONG), 0 (WEAK), -1 (NO_MATCH)"""
    # Use ThresholdLoader for dynamic percentile thresholds
    thresholds = ThresholdLoader.get_thresholds('trend', window)

    if score > thresholds['p75'] and delta > thresholds['delta_p75']:
        return 1  # Strong match
    elif score > thresholds['p50']:
        return 0  # Weak match
    return -1  # No match

# Similar for vol_match() and ofc_match()
```

### `market_regime_mappings_queries.py` - Market Filters
```python
REGIME_QUERIES = [{
    "name": "60_15_5_my_pattern",
    "sql": """
        SELECT open_time FROM ss_market_regime_numeric_codes
        WHERE rolling_window = 5 AND open_time = $1
          AND SUBSTRING(code_trend, 1, 1)::int IN (4,5)  -- 60m bullish
    """,
    "rule_set": "my_setup_v1",
}]
```

## Key Database Tables

### Source Data
- `ss_m1_indicators` - OHLCV + OI + technical indicators (main source)

### Symbol Classification (windowed)
- `ss_token_trend_classification_windowed`
- `ss_token_volatility_classification_windowed`
- `ss_token_orderflow_classification_windowed`

### Market States
- `ss_market_trend_states`
- `ss_market_volatility_states`
- `ss_market_orderflow_states`
- `ss_market_regime_numeric_codes` - Encoded regime (for fast SQL filtering)

### Trading
- `ss_trading_accounts` - Account config + capital tracking
- `ss_trades_open` - Active positions
- `ss_trades_closed` - Historical trades with P&L
- `ss_rule_performance` - Per-setup performance metrics

## Core Components

### ThresholdCacheLoader (`core/threshold_cache_loader.py`)
- Loads percentile thresholds from Redis (5m TTL, DB fallback)
- Keys: `threshold:{metric}:{window}` (e.g., `threshold:trend:60`)
- Used by matching rules for dynamic thresholds

### RedisStateManager (`core/state_management/`)
- Manages distributed state (positions, timers, locks)
- TTL-based cleanup for stale states

### StateShiftRedisPublisher (`core/redis_publisher.py`)
- Publishes trade ideas, regime changes, alerts
- Consumed by external services (webapp, notifications)

### IngestionEventListener (`core/ingestion_event_listener.py`)
- Listens to Redis for `ingestion_complete` events
- Triggers processing cycle when new data arrives

## Performance Guidelines

### Use Vectorized Operations
```python
# BAD: Loop
for symbol in symbols:
    process(symbol)

# GOOD: Polars vectorized
df.with_columns([
    pl.col("value").rolling_mean(5).alias("ma5")
])
```

### Use Bulk Database Operations
```python
# BAD: Individual inserts
for record in records:
    await conn.execute("INSERT...", record)

# GOOD: COPY
await conn.copy_records_to_table('table', records=records)
```

### Cache TTLs
- Market thresholds: 24h
- Rolling window states: 2h
- Symbol classifications: 5m
- Setup performance: 1h

## Account System

### Account Types
- **Internal**: Simulated execution (database only)
- **Prop Firm**: Live execution on Bybit with compliance monitoring

### Key Fields (`ss_trading_accounts`)
```sql
account_type        -- 'internal' or 'prop_firm'
execution_mode      -- 'simulated' or 'live'
setups_filter       -- JSONB array of allowed setup IDs
compliance_rules    -- JSONB prop firm rules
```

## Common Workflows

### Adding a New Trade Setup
1. Create directory: `configs/setups/{setup_name}/`
2. Add: `setup_config.py`, `trade_settings.py`, `symbol_market_matching_rules.py`, `market_regime_mappings_queries.py`, `__init__.py`
3. Add setup ID to `SETUP_IDS` env var in `docker-compose.yml`
4. Deploy with `/deploy` skill

### Modifying Processing Logic
- Phase 1-3: Edit processors in `processors/` directory
- Phase 4 matching: Edit setup's `symbol_market_matching_rules.py`
- Phase 5 execution: Edit `trading/orchestration/` or `trading/execution/`

### Debugging a Cycle
1. Check logs: `docker logs --tail 100 -f state-shift-service`
2. Look for phase timing in "Cycle completed" messages
3. Check Redis: `redis-cli KEYS "ss:*"`
4. Query recent classifications in database

## Environment Variables

```bash
PG_DSN=postgresql://<pg_user>:<pg_password>@<pg_host>:<pg_port>/<pg_db>
REDIS_URL=redis://:<redis_password>@<redis_host>:<redis_port>/<redis_db>
SETUP_IDS=bull_reversal,down_trend_v1,...  # Active setups
TRADE_CONFIG_ROOT=/app/configs
REDIS_USE_STREAMS=true
```

## Deployment

Use `/deploy` skill for VPS deployment instructions.

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

