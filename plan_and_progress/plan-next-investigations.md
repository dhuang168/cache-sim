# Next Investigation Plan: Data Inconsistencies and New Experiments

## Inconsistencies and Unexplained Trends Found

### Issue 1: Q2 shows L2 nearly unused (0.0%) but Q7 shows L2 at 74.2%
- **Q2** (A100, 1W, peak=100): L2 hit rate = 0.0%, only 9 L2 worthwhile events out of 65K
- **Q7** (NPU, 4W, peak=15): L2 hit rate = 74.2%, 6,934 L2 worthwhile events
- **Why?** A100 has 80GB L1 (fits ~4 coding objects) and 1TB L2. NPU has 32GB L1 (fits ~1.5 objects) and 256GB DDR. The NPU's tiny L1 forces most objects to L2, where DDR's 200 GB/s bandwidth makes them worthwhile. But on A100, the TTL migration pushes objects L1→L2→L3A on a timer — objects pass through L2 without being accessed there.
- **Triage**: Is L2 truly useless on A100, or is TTL migration pushing objects out before they're re-accessed? Need to test with LRU eviction (no TTL) on A100.

### Issue 2: Q1 shows NO global-vs-local divergence at any scale (5 min)
- Original heavy_coding_report (20 min, peak=100): 99.8% global vs 68.6% local at 4W
- Q1 (5 min, peak=100): 99.8% both at all worker counts (1-8W)
- **Why?** 5 min isn't long enough for L3A to saturate and sessions to migrate enough. But Phase 0.5 Scenario D (8W, 5 min, peak=100) also showed 99.8% both — confirming this.
- **Triage**: Need to verify with longer sim or identify the exact saturation time where divergence begins.

### Issue 3: Q4 observer counts (10,604) vs engine counts (65,253) — 6:1 gap
- Engine sees 65,253 classified events. Anthropic observer only sees 10,604 PREFILL_COMPLETE events.
- The observer only receives DESEvents emitted after warmup, but the engine classifies all events including warmup. But even post-warmup, the engine should process ~55K events in the 4.5-min collection window.
- **Triage**: Possible that DESEvent emission only fires for a subset of prefill completions. Need to verify emission coverage.

### Issue 4: Q5 observability gap is 67.7% but is likely an artifact of baseline choice
- OpenAI baseline_ttft_us = 50s. Most cache hits have TTFT well below 50s × 0.35 = 17.5s threshold. But some legitimate hits have TTFT > 17.5s (L3A transfers can take 5-10s, plus queue wait).
- The gap isn't inherent to the protocol — it's a function of baseline calibration.
- **Triage**: Re-run Q5 with auto-calibrated baseline (p95 of cold-miss TTFT as baseline).

### Issue 5: Q8 shows global L3A SLOWER than local at controlled load
- GPU global TTFT p50=4.5s vs local 7.0s (global wins by 2.5s)
- NPU global TTFT p50=16.3s vs local 8.6s (global LOSES by 7.7s)
- **Why?** At controlled load, no session migration → local L3A has zero remote penalty. Global L3A adds 50ms per access. On NPU this is worse because the NPU's lower interconnect bandwidth amplifies the penalty.
- **But**: GPU global actually wins despite the penalty. This suggests GPU global has better cache affinity (pull dispatch routes to nodes with cached data). NPU may not benefit from this because its different worker topology (4 GPUs/worker vs 8) changes dispatch behavior.
- **Triage**: Investigate dispatch affinity differences between 4-GPU and 8-GPU worker topologies.

### Issue 6: Q9/Q11 chunk mode consistently underperforms — is our model too pessimistic?
- LMCache (Q9): 45.3% hit vs 99.9% object mode
- SGLang (Q11): 54.9% hit vs 99.9% object mode
- Phase 5 study: 60-76% hit in chunk mode
- But real LMCache reports "3-10x delay savings" and real SGLang shows 85-95% hit rates
- **Triage**: Our consecutive-lookup model may be too strict. Real systems may do partial reuse (CacheBlend) or have smarter chunk grouping. Need to model non-consecutive chunk reuse.

---

## Planned Investigations (7 items)

### I1: L2 utilization — TTL vs LRU eviction on A100 (triage Issue 1)
**Hypothesis**: TTL migration pushes A100 objects through L2 too fast to be re-accessed. LRU eviction (no TTL) should increase L2 hit rate.
**Experiment**: Run Q2 config with `eviction_policy="lru"` vs `"ttl"`. Compare L2 hit rate.
**Script**: `scripts/investigation_i1.py`

### I2: L3A saturation timeline — when does global-vs-local diverge? (triage Issue 2)
**Hypothesis**: Divergence starts when L3A fills past ~90% and sessions begin migrating.
**Experiment**: Run 4W heavy_coding at peak=100 for 2/5/10/15/20 min. Track L3A occupancy and hit rate delta over time.
**Script**: `scripts/investigation_i2.py`

### I3: DESEvent emission coverage audit (triage Issue 3)
**Hypothesis**: Not all prefill completions emit DESEvents — some code paths may skip emission.
**Experiment**: Count prefill completions in engine vs DESEvents emitted. Add a counter to verify 1:1 correspondence.
**Script**: `scripts/investigation_i3.py`

### I4: Auto-calibrated OpenAI baseline (triage Issue 4)
**Hypothesis**: Using cold-miss p95 TTFT as baseline will dramatically reduce the observability gap.
**Experiment**: Run Q5 with baseline = p95 of cold-miss TTFT (auto-detected from first 100 requests).
**Script**: `scripts/investigation_i4.py`

### I5: Dispatch affinity — 4 vs 8 GPUs per worker (triage Issue 5)
**Hypothesis**: 8 GPUs/worker gives better intra-worker L2 sharing, improving global L3A affinity. 4 GPUs/worker has less sharing → less affinity benefit → global penalty dominates.
**Experiment**: Run same workload with 4 vs 8 GPUs/worker, global L3A, track affinity dispatch rate and L1/L2 hit distribution.
**Script**: `scripts/investigation_i5.py`

### I6: Non-consecutive chunk reuse (CacheBlend model) (triage Issue 6)
**Hypothesis**: Allowing partial chunk reuse (skip gaps, recompute only missing chunks) will dramatically improve chunk-mode hit rates.
**Experiment**: Modify chunk lookup to count ALL cached chunks (not just consecutive from 0). Recompute fraction = missing_chunks / total_chunks. Compare hit rate and TTFT.
**Script**: `scripts/investigation_i6.py` (requires engine change)

### I7: Disaggregated P/D + stressed cache (new insight)
**Hypothesis**: Disaggregated mode's 14% TTFT improvement compounds with cache pressure. Under stressed L1 (500MB), the prefill speedup should have even larger impact because more requests hit the long-tail prefill times.
**Experiment**: Run disagg 3P:1D with stressed config (500MB L1) at peak=15. Compare TTFT improvement vs default config.
**Script**: `scripts/investigation_i7.py`
