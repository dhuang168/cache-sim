# Progress: LMCache-Compatible Features (Phase 2)

**Date:** 2026-03-22
**Status:** Complete

## What was done

Added chunk-level KV cache deduplication and demand-pull tier promotion as configurable alternatives:

1. **Config** (`sim/config.py`): 2 new `CacheConfig` fields (`deduplication`, `tier_migration`) + `validate()`
2. **Chunk store** (`sim/chunk_store.py`): New file with `ChunkObject`, `ChunkTierStore` (dedup-aware), `ChunkIndex` (consecutive lookup)
3. **Engine** (`sim/engine.py`): Chunk-mode branches in cache lookup, KV write, epoch reporting. Cascade eviction (L1→L2→L3A). Demand-pull promotion on cache hit.
4. **Metrics** (`sim/metrics.py`): chunk_dedup_hits, chunk_novel_inserts, tier_promotions
5. **Config** (`configs/lmcache.json`): Full LMCache-style config (chunk + demand_pull + LRU)
6. **Tests**: 10 unit tests (test_chunk_store.py) + 11 integration tests (test_chunk_dedup.py)
7. **Docs** (`CLAUDE.md`): Updated with chunk dedup architecture and config reference

## Results

103/103 tests pass (82 existing + 10 chunk store + 11 chunk dedup). Backward compatibility confirmed.

Key metrics from 2-minute sim (default workload, stressed L1):
- **Dedup ratio: 76%** — shared prefix chunks deduped across sessions
- **L1 saturation: 99.9%** → evictions cascade to L2 (10.7%)
- **Demand-pull promotions: 1,002** — chunks pulled back to L1 on access
- **Zero TTL migrations** in demand-pull mode (confirmed by test)

## What was deferred

- Comparison report (Phase 2 vs default model) — should run controlled experiments
- CacheBlend (non-prefix reuse for RAG) — only matters for RAG workloads
- CacheGen compression (3.5-4.3x) — bandwidth savings for remote tiers
- Multi-node chunk stores (currently chunk stores are per single node only)
- Layer-wise pipelining of chunk transfers

## Next step

Write Phase 2 analysis report comparing per-session vs chunk-dedup models.
