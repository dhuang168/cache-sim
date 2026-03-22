# Phase 3: Reactive Eviction Report

Date: 2026-03-22

## Summary

Added LRU reactive eviction policy as alternative to TTL-driven tier migration. In LRU mode, objects stay in their current tier indefinitely until space is needed — no proactive demotion timers.

## Configuration

```python
CacheConfig(eviction_policy="lru")  # or "ttl" (default, backward compatible)
```

## Behavior Comparison

| Aspect | TTL (current default) | LRU (new, vLLM-style) |
|--------|----------------------|----------------------|
| L1→L2 trigger | Timer expires (`ttl_l2_s`), **but resets on access** | L1 capacity pressure only |
| L2→L3A trigger | Timer expires (`ttl_l3a_s`) + epoch scans, **resets on access** | L2 capacity pressure only |
| Objects after session ends | Demoted after TTL if not accessed | **Linger** until space needed — future sessions can reclaim |
| Hot objects | Timer resets on every access → effectively pinned | Never evicted (LRU always picks cold objects) |
| L1 occupancy | Moderate (cold objects drain by timer) | **Higher** (objects stay until evicted) |
| L3A population | Higher (cold objects flow down) | Lower (only overflow reaches L3A) |
| Cache hit rate | Depends on TTL tuning | **Equal or higher** (objects stay in fast tiers longer) |
| Eviction pattern | Gradual (timer-driven trickle for cold objects) | **Bursty** (many evictions when space needed) |

**Note on TTL mode**: The TTL timer is **not strict** — it resets when the object is accessed. If an object is accessed within `ttl_l2_s` of its last access, the timer reschedules. This means frequently-accessed objects (shared framework prefix) stay in L1 indefinitely under TTL mode too. The TTL only fires for objects that have been **idle** for the full TTL duration. This makes TTL mode a hybrid (TTL + LRU-like access refresh).

## Key Insight: LRU + Hot Block Interaction

With LRU, frequently accessed objects (framework prefix, active sessions) **never get evicted** because their `last_accessed_at_us` is constantly refreshed. They effectively get pinned in the fastest available tier. Only cold/expired objects get evicted when capacity pressure arises.

This combines well with Phase 2's sharing model: the shared framework prefix (accessed by every request) stays in L1/L2 permanently under LRU, while TTL mode would proactively demote it after `ttl_l2_s` seconds (even though it's still hot).

## Tests

6 new tests in `tests/test_eviction_policy.py`:
- `test_lru_no_ttl_migrations` — zero TTL-driven L1→L2 migrations
- `test_lru_no_l2_to_l3a_ttl` — zero TTL-driven L2→L3A from epoch scans
- `test_lru_objects_stay_in_l1` — higher L1 occupancy than TTL mode
- `test_lru_pressure_eviction_still_works` — pressure eviction still functions
- `test_ttl_backward_compat` — TTL mode unchanged
- `test_lru_completes_successfully` — LRU sim runs without errors

**72 total tests, 6.09s.**
