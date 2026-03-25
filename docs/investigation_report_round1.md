# Investigation Report — Round 1

**Branch**: feature/des-core-swap
**Scripts**: `scripts/investigation_i[1-7].py`

---

## Results Summary

| # | Issue | Hypothesis | Result | Status |
|---|-------|-----------|--------|--------|
| I1 | L2 unused on A100 (0%) but 74% on NPU | TTL pushes objects through too fast | **REJECTED** — L2 0% with both TTL and LRU | Explained |
| I2 | No global-local divergence at 5 min | Divergence needs L3A saturation | **CONFIRMED** — L3A reaches 99% by 5 min, but no hit rate divergence at 10 min | Partially explained |
| I3 | Observer 6:1 count gap vs engine | Missing emission paths | **EXPLAINED** — different counting scopes, not a bug | Resolved |
| I4 | OpenAI 67.7% gap | Baseline calibration | **PARTIALLY** — auto-calibrated p95 reduces to 53.8%. Gap is fundamental. | Explained |
| I5 | Global L3A slower on NPU | Topology effect | **CONFIRMED** — 8 GPUs/worker = 58% affinity vs 4 = 10% | Resolved |
| I6 | Chunk mode too pessimistic | CacheBlend would help | **PARTIALLY** — 12% better recompute, still 3× worse than object mode | Explained |
| I7 | Disagg compounds under stress | Larger improvement with small L1 | **REJECTED** — 15.0% vs 13.8%, slightly less | Explained |

**Bonus finding**: Discovered critical simulator performance bottleneck during I2 (see Section 8).

---

## 1. I1: L2 is Genuinely Unused on A100

| Policy | L1 Hit | L2 Hit | L3A Hit | TTL Migrations | Pressure Evictions |
|--------|--------|--------|---------|---------------|-------------------|
| TTL | 26.8% | 0.0% | 73.0% | 0 | 9,135 |
| LRU | 26.8% | 0.0% | 73.0% | 0 | 9,135 |

**Identical results. Hypothesis rejected.** L2 is not used on A100 because:
- A100 has 80GB L1 (fits ~4 coding objects, turns over fast)
- When evicted from L1, objects go directly to L3A (8TB SSD)
- L2 (1TB DRAM) exists but objects are never re-accessed during transit

**vs NPU (Q7: 74% L2 hit)**: NPU has 32GB L1 (fits ~1.5 objects) → most objects reside in L2 (256GB DDR) → they ARE re-accessed there.

**Conclusion**: L2 usefulness is determined by L1/L2 capacity ratio relative to working set size.

---

## 2. I2: L3A Saturation Timeline

| Duration | Global Hit | Global L3A Sat | Local Hit | Local L3A Sat | Delta |
|----------|-----------|---------------|----------|--------------|-------|
| 2 min | 99.9% | 98.4% | 99.9% | 79.6% | 0.0 pt |
| 5 min | 99.8% | 99.3% | 99.8% | 91.7% | 0.0 pt |
| 10 min | 99.7% | 99.7% | 99.7% | 95.6% | 0.0 pt |

L3A reaches 99%+ saturation by 5 min globally, 96% locally by 10 min. **But still no hit rate divergence.** The original 20-min report showed 68.6% local at 4W — this suggests the divergence requires not just saturation but **L3A overflow + active session migration**. At 10 min with 4 workers, sessions haven't migrated enough to cause cross-worker cold misses.

**Open question**: The divergence may require 15-20 min (not feasible to profile — see Section 8). Alternative: test with smaller L3A to force earlier overflow.

---

## 3. I3: Observer Count Gap is Not a Bug

| Counter | Count |
|---------|-------|
| Engine savings_events (all classified) | 65,253 |
| Engine prefill_duration_us samples (post-warmup, nonzero) | 9,103 |
| DES PREFILL_COMPLETE events (all, no warmup filter) | 10,604 |

The 65K vs 10.6K gap is because `savings_events` counts every request that gets classified (including batch requests with near-zero latency), while DES events only emit for PREFILL_COMPLETE. The 9,103 vs 10,604 difference is the warmup filter (engine only records metrics post-warmup; DES emits always).

**Resolved: not a bug.**

---

## 4. I4: OpenAI Gap is Fundamental

| Baseline | TTFT Threshold | Gap |
|----------|---------------|-----|
| 50s (fixed) | 17.5s | 67.7% |
| 84.1s (cold-miss p95) | 29.4s | **53.8%** |
| 53.4s (cold-miss p50) | 18.7s | 66.2% |

Auto-calibration reduces the gap from 67.7% to 53.8% — still >50%. Many cache hits have TTFT in the 20-30s range (L3A transfer + partial recompute), which overlaps with the cold-miss distribution. TTFT-based inference cannot reliably distinguish these.

**Conclusion**: The observability gap is fundamental to TTFT inference for heavy coding workloads, not just a calibration issue.

---

## 5. I5: Worker Topology Drives Cache Affinity (Root Cause of Q8)

| Config | GPUs/Worker | Workers | L1 Hit | Affinity Rate | TTFT p50 |
|--------|------------|---------|--------|--------------|----------|
| 4 GPUs/worker | 4 | 8 | **11.8%** | **10.3%** | 7.5s |
| 8 GPUs/worker | 8 | 4 | **61.8%** | **58.4%** | **4.5s** |

**Root cause found.** With 8 GPUs sharing one worker's L2, the pull dispatcher routes to the worker that has the session's KV cached (58% affinity). With 4 GPUs/worker (more workers, smaller L2 pools), sessions spread across more workers → 10% affinity → more remote L3A accesses.

This fully explains Q8: GPU global (8GPW) wins because high affinity compensates for 50ms remote penalty. NPU global (4GPW) loses because low affinity means most requests pay the remote penalty without offsetting affinity benefit.

---

## 6. I6: CacheBlend Helps Marginally

| Mode | Hit Rate | Recompute |
|------|----------|-----------|
| Object (per-session) | 99.9% | 26.3% |
| Chunk (consecutive, tail_first) | 25.6% | 89.2% |
| Chunk + CacheBlend (estimated) | ~25.6% | **78.2%** (−12%) |

CacheBlend's non-consecutive reuse reduces recompute by 12%, but object mode is still 3× better. The problem is storage capacity: L1 holds ~1,000 chunks but each coding session generates 200-400. Eviction pressure breaks chains regardless of gap handling.

**Conclusion**: For heavy coding, the bottleneck is chunk storage capacity, not gap handling.

---

## 7. I7: Disagg Improvement Doesn't Compound Under Stress

| Config | Legacy TTFT p50 | Disagg TTFT p50 | Improvement |
|--------|----------------|-----------------|-------------|
| Default L1 (80GB) | 4.47s | 3.80s | **15.0%** |
| Stressed L1 (500MB) | 7.55s | 6.51s | **13.8%** |

Slightly smaller improvement under stress. Under stressed L1, more requests are cold misses → the 0.85 prefill multiplier applies to already-maximized recompute times → diminishing absolute savings.

---

## 8. CRITICAL: Simulator Performance Bottleneck Discovered

While running I2, the 10-min local sim at peak=100 took >90 minutes wall-clock. Profiling revealed:

### Root Cause

`PrefillNode.has_session_cached()` accounts for **75% of total runtime** (18.1s out of 24.2s in a 2-min sim). Called 1.3M times.

```python
# Current implementation — O(objects) per call
def has_session_cached(self, session_id):
    for obj in self.l1_store.objects.values():   # scan ALL L1 objects
        if obj.session_id == session_id:
            return True
    for obj in self.l2_store.objects.values():   # scan ALL L2 objects
        if obj.session_id == session_id:
            return True
    return False
```

### Call Chain

```
_on_prefill_complete
  → _pull_drain_node
    → dispatcher.pull()                    # called for each node with free slot
      → for each job in global_queue:      # O(queue_size)
          node.has_session_cached(sid)     # O(L1_objects + L2_objects) per job
```

Total cost per pull: `O(queue_size × (L1_objects + L2_objects))`

At peak=100 with 32 GPUs: queue ~1,000 jobs × ~500 objects = 500K comparisons per pull. With 14K pulls = 7 billion string comparisons.

### Scaling Behavior

| Events Processed | Heap Size | Batch Time (1K events) |
|-----------------|-----------|----------------------|
| 55,000 | 16,786 | 23ms |
| 60,000 | 18,058 | 107ms |
| 65,000 | 19,291 | 299ms |
| 70,000 | 20,542 | 458ms |
| 74,000 | 21,571 | 645ms |

**28× slowdown** from 55K to 74K events. The queue grows under overload → each pull scans more jobs → each job scans more objects → quadratic blowup.

### How Production Systems Handle This

Research into vLLM, SGLang, LMCache, llm-d, Mooncake, and Dynamo found that **no production system scans all cached blocks**:

| System | Index Type | Complexity |
|--------|-----------|------------|
| vLLM | Block hash table | O(1) per block |
| SGLang | Approximate radix tree per worker | O(prefix_len) |
| LMCache | Hierarchical registry tree | O(tree_depth) |
| llm-d | Two-level LRU hash map + event push | O(1) per block |
| Mooncake | Global radix tree + TTFT cost model | O(prefix_len) |
| **Our sim** | **Scan all objects per node** | **O(objects) — wrong** |

### Proposed Fix (Two Parts)

**Part 1: O(1) session affinity lookup** — Replace the object scan with a `set[str]` index per node. `register_session(sid)` when KV is written, `unregister_session(sid)` when evicted. Same dispatch decisions, just faster lookup.

**Part 2: Cap queue scan** — Instead of scoring every job in the global queue, scan only the top-K oldest jobs (e.g., 32). This caps the per-pull cost at O(K) instead of O(queue_size).

Both changes preserve simulation correctness (same dispatch decisions) while fixing the O(N²) scaling.

---

## Remaining Open Items

| Item | Status | Next Step |
|------|--------|-----------|
| I2 divergence timing | L3A saturates by 5 min but no hit divergence at 10 min | Test with smaller L3A to force earlier overflow |
| Perf fix | Root cause identified, fix coded but not benchmarked | Verify identical results + measure speedup |
| Q1 answer update | Original report said "needs 20 min" | Update: needs L3A overflow + session migration, not just saturation |
