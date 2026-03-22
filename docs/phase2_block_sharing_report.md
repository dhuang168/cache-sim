# Phase 2: Cross-Session Block Sharing Report

Date: 2026-03-22

## Summary

Added multi-tier prefix sharing with configurable token counts and sharing group sizes. Shared prefix objects use reference counting — multiple sessions share one physical copy. Added cross-worker duplication tracking and global L3A bandwidth contention model.

## Configuration

```python
CacheConfig(
    sharing=SharingConfig(
        enabled=True,
        tiers=[
            SharingTier("framework", 56000, 1000),   # 56K tokens shared by 1000 users
            SharingTier("workspace", 15000, 10),      # 15K tokens shared by 10 users/team
        ],
    ),
)
```

### Three-Tier Sharing Model

| Tier | Tokens | Group Size | What's Shared |
|------|--------|-----------|---------------|
| Framework | 56K (configurable) | 1000 (configurable) | System prompt + tool defs — identical across all users of Claude Code / Cursor |
| Workspace | 15K (configurable) | 10 (configurable) | CLAUDE.md, .cursor/rules — shared within a team |
| Session | remainder | 1 (unique) | Conversation history, tool output |

Based on research: 70-99% of system prompt tokens are shared across users. 1-3 base templates serve millions of users. See `plan_and_progress/research_cross_session_sharing.md`.

## Features

### Reference counting
- When session joins a sharing group: find existing shared object → increment ref_count → record memory saved
- When session expires: decrement ref_count
- Object evictable only when ref_count = 0

### Cross-worker duplication tracking
With load-balanced dispatch, the framework prefix probabilistically gets duplicated across workers:
- `duplicate_block_bytes` — redundant bytes across workers per epoch
- `max_replication_factor` — most-replicated shared object (up to n_workers)
- `shared_prefix_worker_distribution` — per-key worker count

This is the key global vs local L3A tradeoff: global L3A deduplicates across workers, local L3A stores redundant copies.

### Global L3A bandwidth contention
When N concurrent requests access global L3A simultaneously:
- Bandwidth is shared: `transfer_time *= concurrent_reads / n_workers` (when > n_workers)
- Contention events tracked in `l3a_bandwidth_contention_events`
- Penalizes popular shared blocks (framework prefix accessed by many sessions)
- Local L3A doesn't have this problem — each worker has dedicated SSD bandwidth

## New Metrics

| Metric | Description |
|--------|-------------|
| `shared_block_groups` | Number of active sharing groups |
| `shared_block_memory_saved_bytes` | Memory saved by ref-counted sharing |
| `shared_block_ref_count_max` | Highest ref count observed |
| `duplicate_block_bytes` | Redundant bytes across workers (time series) |
| `max_replication_factor` | Most-replicated object (time series) |
| `l3a_concurrent_reads` | In-flight global L3A transfers (time series) |
| `l3a_bandwidth_contention_events` | Times contention factor > 1 |

## Tests

10 new tests in `tests/test_sharing.py`:
- Config defaults and tier stacking
- Ref count increment across sessions
- Memory saved metric > 0
- Shared block groups created
- Different profiles don't share (coding vs agentic_coding)
- Sharing disabled = backward compatible
- Cross-worker duplication tracked
- L3A bandwidth contention tracked
- Sharing reduces tier occupancy

**66 total tests, 5.97s.** All pass including 56 existing tests.
