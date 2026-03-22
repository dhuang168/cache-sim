# Phase 3: Reactive Eviction — Plan

Date: 2026-03-22

## Goal

Replace TTL-driven tier migration with reactive LRU eviction (vLLM-style). Objects stay in their current tier until space is needed — no proactive demotion by timer.

## Current behavior (TTL-driven)
1. Object placed in L1
2. After `ttl_l2_s` seconds, TTL_FIRE event moves it L1→L2 (regardless of L1 pressure)
3. After `ttl_l3a_s` seconds, another TTL_FIRE moves it L2→L3A
4. Epoch report also scans for TTL-expired objects

## New behavior (reactive LRU)
1. Object placed in L1
2. Object stays in L1 **indefinitely** until L1 needs space
3. When L1 needs space: evict LRU object (ref_count=0, oldest last_accessed_at_us) to L2
4. When L2 needs space: evict LRU to L3A
5. Objects linger after their session ends (ref_count=0) — future sessions can reclaim them
6. No TTL_FIRE events for tier migration

## Config

New field in `CacheConfig`:
```python
eviction_policy: str = "ttl"  # "ttl" (current) or "lru" (reactive, vLLM-style)
```

When `eviction_policy="lru"`:
- `ttl_l2_s` and `ttl_l3a_s` are ignored
- No TTL_FIRE events scheduled for tier migration
- Eviction triggered only by capacity pressure in `_place_kv_object`
- Epoch report skips TTL scans

When `eviction_policy="ttl"` (default):
- Current behavior unchanged — full backward compatibility

## Key difference: objects linger

With TTL, an unused object is proactively moved down the tier hierarchy. With LRU, it stays put until something needs its space. This means:
- L1 stays fuller (objects don't migrate until pressure)
- L3A is less populated (objects don't flow down proactively)
- Cache hit rates should be **higher** with LRU (objects stay in fast tiers longer)
- But capacity pressure events are more bursty (many evictions at once when space needed)

## Engine changes

### `_place_kv_object`
- Already has pressure-driven eviction (evict L1→L2 when can't fit)
- No change needed — this IS the reactive eviction path

### TTL_FIRE scheduling
- When `eviction_policy="lru"`: skip scheduling TTL_FIRE events in `_place_kv_object`
- TTL_FIRE handler: still works for TTL mode, no-op for LRU mode

### Epoch report
- When LRU: skip `find_ttl_expired_l2` and `find_ttl_expired_l3a` scans
- Keep L1 pressure check (it's already reactive)

## Tests

1. `test_lru_no_ttl_fires` — LRU mode produces zero TTL migration events
2. `test_lru_objects_linger_in_l1` — objects stay in L1 until pressure (no proactive demotion)
3. `test_lru_higher_l1_occupancy` — LRU mode has higher L1 occupancy than TTL mode (objects stay longer)
4. `test_lru_eviction_on_pressure` — when L1 fills, LRU evicts oldest to L2
5. `test_lru_backward_compat` — TTL mode unchanged (default)
6. `test_lru_vs_ttl_hit_rate` — LRU should have equal or higher L1 hit rate than TTL

## Implementation sequence
1. Add `eviction_policy` to CacheConfig
2. Guard TTL_FIRE scheduling with policy check
3. Guard epoch TTL scans with policy check
4. Tests
5. Phase 3 report
6. Crosscheck, commit, push
