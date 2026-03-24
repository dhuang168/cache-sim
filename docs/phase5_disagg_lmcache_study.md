# Phase 5: Disaggregated P/D + LMCache Study

**Workload**: heavy_coding (90% coding/agentic_coding) | **Model**: Llama3-70B FP16
**Clusters**: 4/8/12 workers × 8 GPUs | **Duration**: 10 min, 1 min warmup
**Dispatch**: pull | **Arrival**: peak scales with cluster (3/6/9)
**Git**: `49f075a` (Phase 2) + chunk store optimization

**Reproducing**: `scripts/disagg_lmcache_study.py`

---

## 1. Summary Table

### Global L3A

| Cluster | Mode | Completed | Hit% | Recompute | TTFT p50 | TTFT p95 | QW p95 | Prefill | Miss% |
|---------|------|-----------|------|-----------|----------|----------|--------|---------|-------|
| 4W (32 GPU) | Legacy | 7,201 | 99.7% | 23.8% | 11.0s | 37.6s | 0s | 13.7s | 0.3% |
| 4W (32 GPU) | **Disagg 3P:1D** | 7,201 | 99.7% | 23.8% | **9.8s** | **33.7s** | 0s | **12.2s** | 0.3% |
| 4W (32 GPU) | Disagg+LMC 3P:1D | 7,201 | 70.3% | 73.0% | 47.5s | 154.0s | 89.3s | 47.0s | 29.7% |
| 8W (64 GPU) | Legacy | 12,215 | 99.6% | 24.3% | 10.5s | 36.5s | 0s | 13.0s | 0.4% |
| 8W (64 GPU) | **Disagg 7P:1D** | 12,215 | 99.6% | 24.3% | **9.0s** | **31.4s** | 0s | **11.2s** | 0.4% |
| 8W (64 GPU) | Disagg+LMC 7P:1D | 12,215 | 74.9% | 69.5% | 35.3s | 144.3s | 38.8s | 44.9s | 25.1% |
| 12W (96 GPU) | Legacy | 19,010 | 99.8% | 24.3% | 10.4s | 36.8s | 0s | 13.0s | 0.2% |
| 12W (96 GPU) | **Disagg 11P:1D** | 19,010 | 99.8% | 24.3% | **8.9s** | **31.8s** | 0s | **11.2s** | 0.2% |
| 12W (96 GPU) | Disagg+LMC 11P:1D | 19,010 | 60.8% | 74.4% | 36.1s | 147.0s | 36.9s | 45.6s | 39.2% |

### Local L3A

| Cluster | Mode | Completed | Hit% | Recompute | TTFT p50 | TTFT p95 | QW p95 | Prefill | Miss% |
|---------|------|-----------|------|-----------|----------|----------|--------|---------|-------|
| 4W (32 GPU) | Legacy | 7,201 | 99.7% | 23.8% | 10.7s | 37.1s | 0s | 13.3s | 0.3% |
| 4W (32 GPU) | **Disagg 3P:1D** | 7,201 | 99.7% | 23.8% | **9.2s** | **32.2s** | 0s | **11.5s** | 0.3% |
| 4W (32 GPU) | Disagg+LMC 3P:1D | 7,201 | 69.3% | 76.9% | 49.3s | 168.2s | 101.0s | 51.0s | 30.6% |
| 8W (64 GPU) | Legacy | 12,215 | 99.6% | 24.3% | 10.3s | 36.5s | 0s | 13.0s | 0.4% |
| 8W (64 GPU) | **Disagg 7P:1D** | 12,215 | 99.6% | 24.3% | **8.8s** | **31.4s** | 0s | **11.1s** | 0.4% |
| 8W (64 GPU) | Disagg+LMC 7P:1D | 12,215 | 73.8% | 76.4% | 41.7s | 156.5s | 58.6s | 50.0s | 26.2% |
| 12W (96 GPU) | Legacy | 19,010 | 99.8% | 24.3% | 10.3s | 36.7s | 0s | 12.9s | 0.2% |
| 12W (96 GPU) | **Disagg 11P:1D** | 19,010 | 99.8% | 24.3% | **8.9s** | **31.8s** | 0s | **11.1s** | 0.2% |
| 12W (96 GPU) | Disagg+LMC 11P:1D | 19,010 | 72.5% | 78.2% | 43.9s | 163.8s | 63.8s | 52.4s | 27.5% |

---

## 2. Key Finding: Disaggregated P/D Delivers Consistent 14% TTFT Reduction

Across all cluster sizes and L3A modes, disaggregated P/D (without LMCache) produces a **consistent ~14% TTFT p50 reduction** from the 0.85 prefill latency multiplier:

| Cluster | Legacy TTFT p50 | Disagg TTFT p50 | Reduction |
|---------|----------------|-----------------|-----------|
| 4W global | 11.0s | 9.8s | **−11%** |
| 8W global | 10.5s | 9.0s | **−14%** |
| 12W global | 10.4s | 8.9s | **−14%** |
| 4W local | 10.7s | 9.2s | **−14%** |
| 8W local | 10.3s | 8.8s | **−14%** |
| 12W local | 10.3s | 8.9s | **−14%** |

The improvement is remarkably stable: no variance with cluster size or L3A mode. This is because:
- The 0.85 prefill multiplier directly reduces compute time by 15%
- At controlled load (slot util 14–22%), there's no queueing — TTFT ≈ prefill time
- Cache hit rates are identical (99.6–99.8%) — disagg doesn't affect caching at this load

### KV Transfer Overhead

| Cluster | Transfer mean | Transfer p95 | % of TTFT p50 |
|---------|-------------|-------------|---------------|
| 4W | 704ms | 1,667ms | 7.2% |
| 8W | 677ms | 1,630ms | 7.5% |
| 12W | 679ms | 1,647ms | 7.6% |

Transfer overhead is ~700ms (7% of TTFT), well worth the 14% prefill speedup. The net benefit is 7% after accounting for transfer.

---

## 3. Finding: Global vs Local L3A Shows No Difference at Controlled Load

Unlike the original heavy_coding_report (which used peak=100, causing severe overload), at controlled load the global vs local L3A distinction **vanishes**:

| Mode | Global Hit | Local Hit | Global TTFT p50 | Local TTFT p50 |
|------|-----------|----------|----------------|---------------|
| Legacy 4W | 99.7% | 99.7% | 11.0s | 10.7s |
| Legacy 8W | 99.6% | 99.6% | 10.5s | 10.3s |
| Legacy 12W | 99.8% | 99.8% | 10.4s | 10.3s |

At controlled load with pull dispatch, sessions don't migrate aggressively between workers, so local L3A rarely misses. The global vs local gap only appears under **heavy overload** where session migration forces cross-worker KV lookups.

---

## 4. Finding: LMCache Chunk Mode Underperforms for Heavy Coding Workloads

The most surprising result: **LMCache-style chunk dedup significantly degrades performance** for heavy coding workloads:

| Cluster | Object-mode Hit | Chunk-mode Hit | Object TTFT p50 | Chunk TTFT p50 |
|---------|----------------|---------------|-----------------|----------------|
| 4W global | 99.7% | 70.3% | 9.8s | 47.5s |
| 8W global | 99.6% | 74.9% | 9.0s | 35.3s |
| 12W global | 99.8% | 60.8% | 8.9s | 36.1s |

**Why chunk mode loses:**

1. **Consecutive lookup fragility**: The chunk-mode lookup requires *all* chunks from position 0 to be cached consecutively. A single evicted chunk in the middle breaks the entire cache hit. With heavy coding contexts (50–100K tokens = 195–390 chunks), the probability of at least one chunk being evicted is high.

2. **Recompute fraction explodes**: Object mode recomputes 24% of tokens (the delta between turns). Chunk mode recomputes 70–78% because a single chunk gap forces recomputation of everything after the gap.

3. **Dedup savings don't compensate**: 54% dedup ratio means shared prefix chunks are stored efficiently, but the total chunk count (500K–1.6M novel chunks) overwhelms L1 capacity, causing eviction cascades that break consecutive lookup chains.

4. **Object mode's PrefixTrie is more resilient**: A single monolithic object either exists or doesn't — no partial hit degradation. The trie returns the deepest matching prefix, and that single object covers the entire cached range.

### When Would Chunk Mode Win?

Chunk mode would outperform object mode when:
- **Many concurrent users with identical prefixes** and **small contexts** (high dedup ratio, few chunks per session — every chunk stays cached)
- **Chat/batch workloads** with short contexts (< 10K tokens = < 40 chunks — consecutive lookup almost always succeeds)
- **Very large L1** relative to working set (no eviction → no broken chains)

For heavy coding/agentic workloads with 50–100K token contexts, the object-mode approach is clearly superior.

---

## 5. Scaling Behavior

### Throughput Scales Linearly

| Cluster | Completed | Throughput |
|---------|-----------|------------|
| 4W (peak=3) | 7,201 | 13.3 req/s |
| 8W (peak=6) | 12,215 | 22.6 req/s |
| 12W (peak=9) | 19,010 | 35.2 req/s |

Throughput scales proportionally with cluster size and arrival rate (as expected at controlled load below saturation).

### Slot Utilization Stays Low

| Cluster | Legacy Util | Disagg Util | LMC Util |
|---------|------------|------------|----------|
| 4W | 18.1% | 21.6% | 67.8% |
| 8W | 14.7% | 14.4% | 57.5% |
| 12W | 15.4% | 14.5% | 57.9% |

Legacy and Disagg are well below saturation (15–22%). LMCache chunk mode shows 58–68% utilization because the high miss rate (30–40%) creates much longer prefills that occupy slots longer.

---

## 6. Chunk Store Optimization Note

During this study, the chunk store was optimized from O(n log n) per eviction (full sort of all chunks) to O(k) per eviction (two-bucket LRU with OrderedDict). This yielded a **~300× speedup** for the chunk mode simulation:

- Before: 40+ minutes for a single 10-min sim with 24 GPUs
- After: 6.8 seconds for the same config

The bottleneck was `evict_lru()` sorting the entire chunk dict on every insertion that triggered capacity pressure. The fix: maintain two LRU buckets (single-ref and multi-ref) as OrderedDicts with O(1) eviction from the front.

---

## 7. Eviction Strategy: LMCache LRU vs vLLM Tail-First

After investigating how production systems handle chunk eviction (LMCache, vLLM APC, SGLang RadixAttention), we found that LMCache has the **same consecutive-lookup fragility** as our simulator — plain LRU evicts chunks from the middle of prefix chains, orphaning everything after the gap.

vLLM APC solves this with **tail-first tie-breaking**: freed blocks are reversed so tail chunks (high index) enter the LRU queue first and get evicted first. This preserves shared prefixes at low indices.

We implemented both as a feature flag (`chunk_eviction="lru"` vs `"tail_first"`) and compared on the same 4W 3P:1D config:

| Metric | LMCache (LRU) | vLLM (tail-first) | Delta |
|--------|--------------|-------------------|-------|
| **Hit rate** | 70.3% | **76.8%** | +6.5pt |
| **Miss rate** | 29.7% | **23.2%** | −6.5pt |
| **Recompute** | 73.0% | **54.4%** | **−18.6pt** |
| **TTFT p50** | 47,506ms | **20,575ms** | **−57%** |
| **TTFT p95** | 153,954ms | **87,449ms** | **−43%** |
| **Dedup ratio** | 0.545 | **0.818** | +50% more dedup |

**Tail-first is dramatically better**: 57% lower TTFT p50, 82% dedup ratio. By evicting from the end of prefix chains, shared prefix chunks (indices 0–77 for coding, 0–117 for agentic_coding) stay cached. The consecutive lookup finds longer unbroken prefixes.

### How production systems handle this

| System | Eviction | Gap Handling |
|--------|----------|-------------|
| **LMCache** | Plain LRU, position-unaware | Breaks at first missing chunk (same gap problem) |
| **vLLM APC** | LRU + tail-first tie-breaking | Preserves prefixes by freeing from sequence end |
| **SGLang Radix** | Leaf-first LRU only | Gaps impossible by construction (tree structure) |

SGLang's radix tree is the cleanest — eviction only affects leaves (sequence ends), and the tree structure guarantees prefix integrity. vLLM's tail-first is a pragmatic approximation that works with block-level storage. LMCache relies on CacheBlend (non-contiguous reuse) for the RAG case but has this gap problem for prefix caching.

### Updated chunk mode comparison (4W, Disagg 3P:1D, global L3A, 10 min)

| Mode | Hit% | Recompute | TTFT p50 | TTFT p95 |
|------|------|-----------|----------|----------|
| Legacy (object, no chunk) | 99.7% | 23.8% | 11.0s | 37.6s |
| Disagg (object) | 99.7% | 23.8% | **9.8s** | **33.7s** |
| Disagg + LMCache LRU | 70.3% | 73.0% | 47.5s | 154.0s |
| **Disagg + vLLM tail-first** | **76.8%** | **54.4%** | **20.6s** | **87.4s** |

Tail-first dramatically closes the gap with object mode but still underperforms it for heavy coding. The remaining 23% miss rate comes from session-unique chunks (beyond the shared prefix) that grow rapidly and overflow L1 regardless of eviction order.

---

## 8. Recommendations

1. **Disaggregated P/D is recommended** for all cluster sizes. The 14% TTFT reduction is consistent and the 7% KV transfer overhead is acceptable at 50 GB/s RDMA.

2. **If using chunk dedup, use vLLM tail-first eviction**, not LMCache LRU. Tail-first is 57% faster TTFT and 50% better dedup ratio for heavy coding workloads. The `chunk_eviction="tail_first"` flag enables this.

3. **Object-mode per-session storage still outperforms chunk dedup** for heavy coding (99.7% vs 76.8% hit rate). The remaining gap is structural: session-unique context grows past shared prefix and overflows L1 regardless of eviction strategy.

4. **Global vs local L3A doesn't matter at controlled load.** The distinction only appears under severe overload with session migration.

5. **Chunk mode may be viable for chat/batch workloads** with short contexts (<10K tokens) where chunks stay cached. A follow-up study with a chat-heavy workload mix would be informative.

6. **CacheBlend (non-contiguous reuse)** would address the fundamental fragility — reuse cached chunks even with gaps by selectively recomputing only ~15% of tokens. This is a GPU kernel-level change, not just a caching policy.
