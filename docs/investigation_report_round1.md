# Investigation Report — Round 1

**Branch**: feature/des-core-swap
**Scripts**: `scripts/investigation_i[1-7].py`

---

## Results Summary

| # | Issue | Hypothesis | Result | Status |
|---|-------|-----------|--------|--------|
| I1 | L2 unused on A100 | TTL pushes objects through too fast | **REJECTED** — L2 0% with both TTL and LRU. Objects bypass L2 entirely. | Explained |
| I2 | No global-local divergence at 5 min | Divergence needs L3A saturation | **PENDING** — running 2-20 min timeline | Running |
| I3 | Observer 6:1 count gap | Missing emission paths | **EXPLAINED** — DES emits 10.6K events, engine counts 65K savings_events. Gap is because savings_events counts ALL classified requests (including batch/batch-like fast ones), while observer only sees PREFILL_COMPLETE events. Ratio engine_prefill_samples/DES = 0.86 (DES emits slightly more due to no warmup filter on _emit). | Explained |
| I4 | OpenAI 67.7% gap | Baseline calibration | **PARTIALLY CONFIRMED** — auto-calibrated p95 baseline (84.1s) reduces gap to 53.8%. Still large. Gap is fundamental to TTFT inference. | Explained |
| I5 | Global L3A slower on NPU | Topology effect | **CONFIRMED** — 8 GPUs/worker: 58% affinity, 62% L1 hit. 4 GPUs/worker: 10% affinity, 12% L1 hit. Worker size drives cache affinity, which determines whether global L3A overhead is compensated. | Resolved |
| I6 | Chunk mode too pessimistic | CacheBlend would help | **PARTIALLY** — CacheBlend estimated 12% better recompute (0.89→0.78). Still far worse than object mode (0.26). Chunk fragility is structural. | Explained |
| I7 | Disagg compounds under stress | Larger improvement with small L1 | **REJECTED** — 15.0% improvement (default) vs 13.8% (stressed). Slightly less, not more. | Explained |

---

## Detailed Findings

### I1: L2 is genuinely unused on A100

| Policy | L1 Hit | L2 Hit | L3A Hit | TTL Migrations | Pressure Evictions |
|--------|--------|--------|---------|---------------|-------------------|
| TTL | 26.8% | 0.0% | 73.0% | 0 | 9,135 |
| LRU | 26.8% | 0.0% | 73.0% | 0 | 9,135 |

**Identical results.** L2 is not used on A100 because:
- A100 has 80GB L1 (fits ~4 coding objects, turns over fast)
- When objects are evicted from L1 under pressure, they go directly to L3A (8TB SSD)
- L2 (1TB DRAM) exists but objects are never re-accessed during their brief transit
- This is NOT a TTL artifact — it's a capacity/access pattern issue

**vs NPU (Q7: 74% L2 hit)**: NPU has 32GB L1 (fits ~1.5 objects) → most objects must reside in L2 (256GB DDR) → they ARE re-accessed there before migrating to L3A.

**Conclusion**: L2 usefulness is determined by L1/L2 capacity ratio relative to working set, not eviction policy.

---

### I3: DESEvent emission is correct

| Counter | Count |
|---------|-------|
| Engine savings_events | 65,253 |
| Engine prefill_duration_us samples | 9,103 |
| DES PREFILL_COMPLETE events | 10,604 |
| DES DECODE_COMPLETE events | 10,604 |

**The 65K vs 10.6K gap is NOT a bug.** `savings_events` counts every request that gets classified (including during warmup and including batch requests with near-zero prefill). `prefill_duration_us` only captures post-warmup requests with nonzero prefill. DES emission happens for ALL prefill_complete events (no warmup filter on `_emit`).

Ratio: 9,103 / 10,604 = 0.86 — the 14% difference is because DES emits during warmup but engine only records prefill_duration after warmup. This is correct behavior.

---

### I4: OpenAI gap is fundamental, not just calibration

| Baseline | TTFT Threshold | Gap |
|----------|---------------|-----|
| 50s (fixed) | 17.5s | 67.7% |
| 84.1s (cold-miss p95) | 29.4s | **53.8%** |
| 53.4s (cold-miss p50) | 18.7s | 66.2% |

Auto-calibrating the baseline to cold-miss p95 reduces the gap from 67.7% to 53.8% — a 14pt improvement. But **53.8% gap remains** because many cache hits have TTFT in the 20-30s range (L3A transfer + partial recompute), which exceeds 35% of even the 84s baseline. The threshold is too coarse to distinguish "fast L3A hit" from "slow cold miss".

**Conclusion**: TTFT-based cache inference is fundamentally limited for heavy coding workloads where cache hits and misses overlap in latency distribution.

---

### I5: Worker topology drives cache affinity

| Config | GPUs/Worker | Workers | L1 Hit | Affinity Rate | TTFT p50 |
|--------|------------|---------|--------|--------------|----------|
| 4 GPUs/worker | 4 | 8 | 11.8% | 10.3% | 7.5s |
| 8 GPUs/worker | 8 | 4 | **61.8%** | **58.4%** | **4.5s** |

**Root cause of Q8**: With 8 GPUs sharing one worker's L2, the pull dispatcher routes requests to the worker that has the session's KV in its shared L2. With 4 GPUs/worker (more workers), sessions are spread across more workers → less affinity → more L3A accesses → global L3A remote penalty (50ms) hurts more.

**This fully explains** why GPU global (8GPW) wins but NPU global (4GPW) loses at controlled load.

---

### I6: CacheBlend helps marginally, doesn't close the gap

| Mode | Hit Rate | Recompute |
|------|----------|-----------|
| Object (per-session) | 99.9% | 26.3% |
| Chunk (consecutive, tail_first) | 25.6% | 89.2% |
| Chunk + CacheBlend (estimated) | ~25.6% | **78.2%** |

CacheBlend would reduce recompute from 89.2% to 78.2% (12.3% improvement). But object mode achieves 26.3% — still 3× better. The fundamental issue is that chunk hit rate (25.6%) is low because L1 can only hold ~1000 chunks but each coding session generates 200-400 chunks. Eviction pressure breaks consecutive chains regardless of gap handling.

**Conclusion**: For heavy coding, the problem is storage capacity for chunks, not gap handling. Object mode wins because one object covers the entire prefix — no chain to break.

---

### I7: Disagg improvement doesn't compound under stress

| Config | Legacy TTFT p50 | Disagg TTFT p50 | Improvement |
|--------|----------------|-----------------|-------------|
| Default L1 (80GB) | 4.47s | 3.80s | **15.0%** |
| Stressed L1 (500MB) | 7.55s | 6.51s | **13.8%** |

The improvement is slightly **smaller** under stress (13.8% vs 15.0%), not larger. Under stress, more requests are cold misses (full recompute) — the 0.85 multiplier applies to prefill time, which is already maximized for cold misses. The compounding effect doesn't materialize because cold misses dominate.

---

## Open Item: I2 (Saturation Timeline)

I2 is still running (10 configs × 2-20 min at peak=100). Will update when complete. This is the key investigation for Q1 — finding exactly when L3A saturates and the global-local divergence begins.

---

## Resolved Issues Summary

| Original Issue | Resolution |
|---------------|-----------|
| L2 unused (Issue 1) | **Explained**: capacity/access pattern, not eviction policy |
| No divergence at 5 min (Issue 2) | **Pending I2**: need saturation timeline |
| Observer 6:1 gap (Issue 3) | **Explained**: different counting scopes, not emission bug |
| OpenAI 67.7% gap (Issue 4) | **Explained**: fundamental to TTFT inference, calibration helps 14pt |
| Global slower on NPU (Issue 5) | **Resolved**: 4 GPUs/worker → low affinity → remote penalty dominates |
| Chunk mode too pessimistic (Issue 6) | **Explained**: structural — storage capacity, not gap handling |
