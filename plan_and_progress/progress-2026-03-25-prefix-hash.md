# Progress: Prefix-Hash Routing (OpenAI-Style Dispatch)

**Date:** 2026-03-25
**Branch:** main

## What was done

1. **Research**: Confirmed OpenAI routes by hash of first ~256 tokens. Overflow at ~15 req/min. Creates hotspots.

2. **Implementation**:
   - New `PrefixHashDispatcher` class in `agentsim/core/des/dispatch.py`
   - Configurable `prefix_hash_tokens` (default 256) and `prefix_hash_overflow_threshold` (default 0.9)
   - `dispatch_algorithm = "prefix_hash"` as new option alongside push/pull
   - Tracks `overflow_count` and `total_dispatches`

3. **Tests**: 6 tests covering: runs, consistent routing, different profiles, configurable prefix length, overflow under pressure, same completed count as push.

4. **Experiment**: Compared push/pull/prefix_hash at peak=15 and peak=50.

## Key Findings

- **24-28× load imbalance** with prefix_hash (vs 3-6× for push/pull) — heavy coding has only 2-3 dominant prefixes
- **70-89% overflow rate** — most requests can't fit on target node
- **2× worse TTFT at peak=50** (12.3s vs 5.7s push) despite same 99.9% hit rate
- **Push wins at controlled load** — even distribution beats concentration
- Prefix-hash is designed for **high prefix diversity** (many different system prompts), not coding workloads dominated by 1-2 prompts
