# Progress: Phase 1 Hardening

**Date:** 2026-03-24
**Branch:** feature/des-core-swap
**Status:** Complete — all three hardening items done

## What was done

1. **CacheKey in DESEvent payloads**:
   - PREFILL_COMPLETE events now include `cache_key: CacheKey` object
   - Generated from `(model_id, "default", session_id:cached_tokens)` or `"miss"`
   - SavingsEvent classification improved — uses tier map instead of string hacking

2. **Confidence labels in oracle**:
   - `SimplePrefillOracle` now has `confidence` attribute (auto-detected from table metadata)
   - New `prefill_latency_with_confidence()` → `(int, ConfidenceLabel)` tuple
   - `SimpleDecodeOracle` also has `decode_latency_with_confidence()`
   - Backward-compatible: bare `int` methods still work

3. **MetricsObserver(ObserverBase)**:
   - New `MetricsObserver` class wraps `MetricsCollector` as an observer
   - Receives DESEvents, counts prefill/decode completions
   - Can be registered as engine observer: `SimEngine(cfg, observers=[MetricsObserver(collector)])`
   - Direct field access on MetricsCollector preserved for backward compat

## Updated Phase 1 gate status
```
[x] All benchmark scenarios pass tolerance bands
[x] All cache object creation uses bytes
[x] Every prefill oracle returns (latency, confidence_label) — via with_confidence()
[x] prefill_latency_us and decode_latency_us reported separately
[x] SavingsEvent classification present on every completed request
[x] CacheKey included in PREFILL_COMPLETE DESEvent payloads
[x] cache-sim's existing tests pass
[x] Stress scenario passes
[x] lint-imports passes
[x] Progress report written
```
