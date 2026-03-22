# Phase 2: Cross-Session Block Sharing — Plan

Date: 2026-03-22

## Goal

Model three-tier prefix sharing with configurable token counts and sharing percentages. Identical token blocks across sessions share physical storage via reference counting.

## Config: `SharingConfig`

New config section with per-tier token counts and sharing fractions:

```python
@dataclass
class SharingTier:
    name: str                    # "framework", "workspace", "session"
    tokens: int                  # tokens in this tier (e.g., 56000)
    sharing_group_size: int      # users sharing this tier (e.g., 1000 for framework, 10 for workspace)

@dataclass
class SharingConfig:
    enabled: bool = False
    tiers: list[SharingTier] = field(default_factory=list)
    # Example: [
    #   SharingTier("framework", 56000, 1000),   # 56K tokens shared by 1000 users
    #   SharingTier("workspace", 15000, 10),      # 15K tokens shared by 10 users
    #   SharingTier("session", 0, 1),              # remainder is session-unique
    # ]
```

The tiers stack: framework prefix is the outermost (shared by most users), workspace is next, session is innermost (unique).

Total shared_system_prefix_tokens from the profile = sum of sharing tier tokens.

## How It Works

### Session creation
When a session is created with profile `shared_system_prefix_tokens = 56000 + 15000 = 71000`:
1. **Framework tier** (56K tokens): Look up shared block group `framework:{profile_name}`. If exists, increment ref_count on those blocks. If not, create blocks and store.
2. **Workspace tier** (15K tokens): Look up `workspace:{profile_name}:{workspace_id}`. Workspace_id derived from `session_id % sharing_group_size` (simulates team grouping). Increment ref or create.
3. **Session tier** (remaining tokens): Always unique per session.

### Block ref counting
- Each `CacheObject` (or future `KVBlock`) has `ref_count`
- When a new session joins a sharing group, ref_count increments
- When a session ends, ref_count decrements
- Object is only eligible for eviction when ref_count == 0

### Memory savings
- Without sharing: N sessions × 71K tokens × 327KB/token = N × 22.2 GB
- With sharing (1000 users, 10/workspace): 1 × 56K framework + 100 × 15K workspace + 1000 × unique session
- Savings tracked as new metric: `shared_block_memory_saved_bytes`

### Metrics
- `shared_block_groups: int` — number of active sharing groups
- `shared_block_memory_saved_bytes: int` — memory saved by block sharing
- `shared_block_ref_count_mean: float` — average ref count across shared objects
- `subagent_sharing_hits: int` — subagent invocations that reused parent's shared blocks

## Additional Considerations (updated)

### (a) Probabilistic duplication
Duplication of shared prefixes across workers is probabilistic, not guaranteed. A framework prefix gets duplicated to worker W only if at least one session of that type lands on W. Track **actual observed** distribution, not assumed full replication.

### (b) Global L3A bandwidth contention
When N concurrent requests access the same global L3A block, they share SSD bandwidth:
- Transfer time = `size / (bandwidth / concurrent_accessors)` = `size * N / bandwidth`
- This penalizes popular shared blocks (framework prefix accessed by many sessions simultaneously)
- Local L3A doesn't have this problem — each worker has its own SSD bandwidth

Implementation: track concurrent L3A accesses per epoch. For global L3A, scale transfer time by contention factor: `transfer_time *= max(1, concurrent_l3a_reads / n_workers)`.

New metrics:
- `l3a_concurrent_reads: list[int]` — concurrent L3A reads at each epoch
- `l3a_bandwidth_contention_factor: list[float]` — effective bandwidth reduction

## Engine Changes

### `_place_kv_object` modification
Before creating a new CacheObject for the shared prefix portion:
1. Check if a shared block group already exists for this tier
2. If yes: increment ref_count, update last_accessed_at_us, record memory saved
3. If no: create normally, register in shared block group index

### New: `_shared_block_index`
Dict mapping sharing group key → cache_key of the shared CacheObject:
```python
self._shared_block_index: dict[str, str] = {}
# e.g., "framework:coding" → "sp-framework-coding"
#       "workspace:coding:3" → "sp-workspace-coding-3"
```

### Session end cleanup
When session ends, decrement ref_count on its sharing group objects.
If ref_count hits 0, object becomes evictable.

## Tests

### Unit tests
1. `test_sharing_config_defaults` — SharingConfig disabled by default
2. `test_sharing_tier_stacking` — framework + workspace + session tokens sum correctly
3. `test_ref_count_increment` — two sessions in same group → ref_count=2
4. `test_ref_count_decrement` — session end → ref_count decrements
5. `test_memory_saved_metric` — N sessions sharing = (N-1) × shared_size saved

### Extreme condition tests
6. `test_1000_sessions_same_framework` — 1000 sessions, 1 physical copy of 56K prefix
7. `test_different_frameworks_no_sharing` — coding vs agentic_coding → separate shared blocks
8. `test_workspace_groups` — 100 sessions, 10/workspace → 10 workspace prefix copies
9. `test_sharing_disabled_backward_compat` — sharing off → identical to Phase 1 behavior

### Integration
10. `test_sharing_reduces_memory` — sim with sharing → lower tier occupancy than without
11. `test_subagent_sharing` — if modeled, subagent reuses parent's framework blocks

## Implementation Sequence

1. Add `SharingTier` and `SharingConfig` to config.py
2. Add `shared_block_memory_saved_bytes` and sharing metrics to metrics.py
3. Add `_shared_block_index` to engine
4. Modify `_place_kv_object` to check shared index before creating
5. Add session-end ref_count cleanup
6. Add tests
7. Verify all existing tests pass (sharing disabled by default)
8. Create `docs/phase2_block_sharing_report.md`
9. Crosscheck, commit, push
