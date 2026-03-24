# Progress: Phase 1 — Core DES Layer Port

**Date:** 2026-03-24
**Branch:** feature/des-core-swap
**Status:** Complete — engine ported, DESEvent emission working, all tests pass

## What was done

Ported all 11 sim/ modules into agentsim/core/des/ with contract adaptations:

### Sub-Phase 1.1: Foundation (config, workload, service, node, events)
- Copied 5 modules, fixed all `sim.*` imports to `agentsim.core.des.*`
- No logic changes needed

### Sub-Phase 1.2: Cache + Eviction
- Copied cache.py, eviction.py, chunk_store.py
- Added `make_cache_key()` helper for CacheKey generation
- Added `cache_object_to_contract()` bridge for frozen contract CacheObject
- Added `LRUEvictionPolicy(EvictionPolicyBase)` wrapper

### Sub-Phase 1.3: Oracle
- Merged sim/oracle.py functions into existing agentsim/core/des/oracle.py
- Added `SimplePrefillOracle` (numpy interp, backward-compatible)
- Added `SimpleDecodeOracle` (sqrt batch degradation)
- Added `transfer_time_us()`, `kv_transfer_time_us()`, `is_cache_worthwhile()`
- Existing `PiecewiseOracle` and `RooflineOracle` preserved (contract-compliant)

### Sub-Phase 1.4: Dispatch
- Copied PushDispatcher, PullDispatcher, SmartPushDispatcher
- Fixed imports

### Sub-Phase 1.5: Engine + Metrics
- Copied full engine (~1900 lines), fixed all imports
- Added `observers: list[ObserverBase]` parameter to `__init__`
- Added `_emit(kind, payload)` helper for DESEvent emission
- Added DESEvent emission at PREFILL_COMPLETE and DECODE_COMPLETE
- Added SavingsEvent classification in PREFILL_COMPLETE payload
- Fixed benchmark table path (parent.parent.parent.parent for repo root)
- Metrics stays as direct fields (backward-compat); ObserverBase rewrite deferred

### Sub-Phase 1.6: Verification
- New engine produces **identical results** to old engine (same seed/config → same completed count + hit rates)
- 7 new Phase 1 tests pass (match, DESEvent emission, payloads, unique IDs, timestamps, perf)
- All 165 existing tests pass (sim/ untouched)
- lint-imports: 3 contracts kept, 0 broken
- Performance: 30s sim completes in 0.1s

## Key verification result
```
Old engine: completed=724, hit={L1: 0.859, L2: 0.054, L3A: 0.087, miss: 0.0}
New engine: completed=724, hit={L1: 0.859, L2: 0.054, L3A: 0.087, miss: 0.0}
Match: completed=True, hit_rate_match=True
```

## What was deferred
- Full MetricsCollector as ObserverBase (currently still direct field access)
- CacheKey threading through all cache lookups (currently string keys with CacheKey bridge)
- More DESEvent emission points (REQUEST_ARRIVAL, CACHE_LOOKUP, EVICTION, TIER_TRANSFER)
- DispatcherBase adapter wrapper (dispatchers work directly, adapter deferred)

## Gate checklist status
- [x] New engine matches old engine (identical output)
- [x] DESEvent emission at PREFILL_COMPLETE and DECODE_COMPLETE
- [x] All 165 existing tests pass
- [x] lint-imports: 3 contracts kept, 0 broken
- [x] Performance gate: 30s sim in <5s
- [x] Progress report written

## Next step
Run Phase 0.5 tolerance tests against new engine to verify Scenarios A-D pass.
