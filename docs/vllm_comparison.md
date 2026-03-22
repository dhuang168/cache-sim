# Comparison: PCS Cache Simulator vs vLLM

This document maps our simulator's design against vLLM's actual implementation to identify where we align, where we diverge, and what the implications are for simulation accuracy.

## 1. Cache Lookup and Prefix Matching

| Aspect | PCS Simulator | vLLM |
|--------|--------------|------|
| **Data structure** | Per-session `PrefixTrie` + shared prefix trie | Global hash table (SHA-256, parent-chained per block) |
| **Matching granularity** | Token count as proxy for prefix depth | Per-block (16 tokens/block), exact token-level matching |
| **Matching scope** | Session-local trie + shared system prefix trie | Global — any request can match any previously-cached block chain |
| **Cross-session sharing** | Only via `shared_system_prefix_tokens` mechanism | Fully implicit — identical token blocks share physical storage |

### Implications

**Our simulator under-counts cache sharing.** In vLLM, if two unrelated users submit identical 20k-token system prompts, they automatically share the same physical KV blocks. In our simulator, each session creates its own KV object — the only sharing is the `sp-{profile}` shared prefix entry, which avoids recomputation but doesn't share storage.

**Our prefix matching is approximate.** We use token count as a proxy for prefix depth (if the trie entry has ≥N tokens, it's a "match"). vLLM uses exact hash comparison — same tokens at the block level. Our approach is faster but can't distinguish different prompts of the same length.

**Recommendation**: For simulating cache hit rates, our model is a reasonable upper bound (prefix stability determines what fraction is "cached"). For simulating memory usage, we overestimate because we don't deduplicate identical blocks across sessions.

## 2. KV Cache Storage

| Aspect | PCS Simulator | vLLM |
|--------|--------------|------|
| **Allocation unit** | Per-session KV object (variable size, whole context) | Per-block (16 tokens, fixed size) |
| **Storage model** | `TierStore` with `capacity_bytes` and `block_size_bytes` | `BlockPool` with physical block array, ref counts |
| **Fragmentation** | Internal fragmentation from block-size rounding | Near-zero (<4%) — only last partial block |
| **Copy-on-write** | Not modeled | Supported (parallel sampling, shared blocks) |
| **Reference counting** | `ref_count` on `CacheObject` (shared prefix only) | `ref_cnt` per physical block (all sharing) |

### Implications

**Our allocation is coarser.** We store entire context KV as one object. vLLM stores it as a chain of 16-token blocks. This means:
- vLLM can share partial prefixes (first N blocks match, rest diverge)
- Our model is all-or-nothing: either the whole cached prefix matches or it doesn't
- vLLM's fragmentation is near-zero; ours depends on `block_size_bytes` configuration

**Recommendation**: Our model accurately captures the hit/miss dynamics (most value comes from the system prompt prefix which is all-or-nothing anyway). For memory capacity planning, vLLM's finer granularity means it can pack more useful data into the same space.

## 3. Cache Tiers

| Aspect | PCS Simulator | vLLM |
|--------|--------------|------|
| **L1 (GPU HBM)** | `TierStore` per GPU, explicit capacity | Primary KV cache in GPU memory, managed by `BlockPool` |
| **L2 (Host DRAM)** | `TierStore` per worker, shared by GPUs | Swap target for preempted requests (async DMA) |
| **L3 (SSD/External)** | `TierStore` per worker or global pool | Via Connector API (LMCache for Ceph/Redis) |
| **Tier movement** | TTL-driven + pressure-driven, explicit events | Scheduler-driven: preemption (swap/recompute), connector offloading |

### Key Differences

**TTL-driven tier migration vs scheduler-driven preemption:**
- Our simulator moves objects L1→L2 after `ttl_l2_s` expires, regardless of memory pressure. This models a proactive demotion policy.
- vLLM doesn't use TTLs. Blocks stay in GPU memory until evicted by LRU (when space is needed) or preempted (when the scheduler needs memory for a higher-priority request).
- vLLM V1 defaults to **recompute preemption** (discard KV, recompute later) rather than swapping to CPU, because swap overhead often exceeds recompute cost for shorter prefills.

**Our L2/L3 tiers are more structured.** We model explicit DRAM and SSD tiers with bandwidth/latency. vLLM's "L2" is CPU DRAM used only for swap, and "L3" requires an external connector. The tiered hierarchy is less rigid in vLLM.

**Recommendation**: Our TTL model is more conservative (proactive demotion). vLLM's reactive model keeps objects in fast storage longer. For capacity planning at the tier level, our model is reasonable. For latency modeling of eviction events, vLLM's reactive approach would produce different timing patterns.

## 4. Eviction Policy

| Aspect | PCS Simulator | vLLM |
|--------|--------------|------|
| **L1 eviction** | Watermark-based: evict when occupancy > threshold, choose oldest/lowest-ref | LRU among ref_cnt==0 blocks |
| **L2→L3 eviction** | TTL-driven hibernation | Scheduler preemption (recompute or swap) |
| **L3 cleanup** | LRU | N/A (external storage managed separately) |
| **Shared prefix protection** | Never evict shared before private | Preserve roots of longest chains (tie-breaker) |

### Implications

Both use LRU-family policies. The key difference is granularity:
- vLLM evicts **individual blocks** (16 tokens). An object's blocks can be partially evicted.
- Our simulator evicts **entire objects** (full session KV). It's all-or-nothing.

vLLM's block-level eviction is more efficient — it can evict just the tail blocks of a long context while keeping the shared prefix. Our simulator would evict the entire object.

## 5. Multi-Node Architecture

| Aspect | PCS Simulator | vLLM |
|--------|--------------|------|
| **Topology** | N workers × 8 GPUs, explicit L1/L2/L3 per worker | TP within node, PP across nodes |
| **Scheduler** | Push/pull dispatcher, per-request node assignment | Centralized scheduler (rank 0) |
| **KV cache coordination** | Dispatcher checks affinity (L1/L2 only) | Scheduler manages all block allocation/eviction |
| **Cross-node cache access** | Global L3A with 50ms remote latency | Not supported — KV cache is node-local in standard vLLM |
| **Session migration** | Happens when affinity lost (L3A not checked) | Not a concept — each request is independently scheduled |

### Key Differences

**vLLM has no "global L3A" concept.** Each node's GPU memory holds KV cache blocks independently. There is no mechanism for Node B to access Node A's cached blocks. The Connector API (LMCache) can offload to shared storage (Ceph), but this is external and latency is high.

**Our "session migration" problem doesn't exist in vLLM the same way.** In vLLM, each request is independently scheduled and gets its own prefix cache lookup against the local hash table. If the prefix is cached on this node, great. If not, it's a cache miss and the prefix is recomputed — there's no concept of "routing to the node that has the cache."

**Distributed prefix caching is an active research area.** Projects like `llm-d` and `KV-Cache-Wins` propose routing requests to nodes with cached prefixes (similar to our push dispatch with affinity). This is exactly what our simulator models.

**Recommendation**: Our simulator is modeling a **future architecture** that doesn't fully exist in production vLLM yet. The global L3A + affinity dispatch combination represents a design that systems like llm-d are working toward. This makes the simulator valuable for exploring design decisions before implementation.

## 6. Prefill Compute Model

| Aspect | PCS Simulator | vLLM |
|--------|--------------|------|
| **Prefill latency** | Piecewise-linear oracle from benchmark table | Actual GPU computation (no model) |
| **Batched prefill** | Not modeled (single-sequence) | Chunked prefill, batched with decode |
| **Decode** | `sqrt(batch_size)` degradation model | Continuous batching with PagedAttention |
| **Cache hit benefit** | Skip `oracle(cached_tokens)`, only compute `oracle(uncached_tokens)` | Skip prefill for matched blocks, only compute unmatched tail |

### Implications

**Our prefill model is single-sequence.** vLLM batches multiple prefills and interleaves with decode (chunked prefill). This means vLLM's actual prefill throughput is higher than our oracle suggests, because GPU utilization is better.

**Our cache hit benefit is correctly modeled.** Both systems skip computation for the cached prefix and only compute the new suffix. The savings are proportional to `cached_tokens / total_tokens`, which our prefix stability model captures.

## 7. Scheduler and Backpressure

| Aspect | PCS Simulator | vLLM |
|--------|--------------|------|
| **Admission** | Fixed slot count + queue + backpressure (drop) | Memory-based: admit if sufficient KV cache blocks available |
| **Preemption** | Not modeled (once admitted, runs to completion) | Preempt lower-priority requests to free blocks |
| **Queue model** | Explicit pending queue per node, FIFO | Waiting queue with priority, FCFS default |
| **Backpressure** | Drop request if queue full | Queue indefinitely (server-side), or reject via API |

### Key Difference

vLLM can **preempt** running requests — evict their KV cache and reschedule them later. Our simulator never preempts: once a request gets a slot, it runs to completion. This means our model doesn't capture the scenario where a high-priority request causes a running request to be evicted and requeued.

## Summary: Alignment and Gaps

### Where we align well

| Aspect | Alignment |
|--------|-----------|
| Cache hit/miss economics | ✅ Both skip prefill for cached prefix, compute for uncached suffix |
| Multi-tier storage | ✅ Both have GPU → CPU → external hierarchy |
| LRU eviction family | ✅ Both use LRU-based policies |
| Session model | ✅ Per-request KV cache, growing context per turn |
| Prefix stability | ✅ Our model captures the fraction of context that is reusable |

### Where we diverge

| Aspect | Gap | Impact |
|--------|-----|--------|
| **Block vs object granularity** | vLLM uses 16-token blocks; we use configurable block sizes | ✅ Phase 1: configurable block_size_tokens (16/256/4096). Boundary rounding modeled. |
| **Hash vs trie matching** | vLLM's hash is exact; our trie uses token count proxy | Partial gap — exact hash matching not yet implemented, but block boundary rounding captures the economic effect |
| **Cross-session block sharing** | vLLM shares physical blocks implicitly; we use ref-counted sharing tiers | ✅ Phase 2: three-tier sharing (framework/workspace/session) with ref counting, duplication tracking, bandwidth contention |
| **Preemption** | vLLM preempts running requests; we don't | We can't model priority-based scheduling dynamics |
| **Batched prefill** | vLLM batches prefill with decode; we model single-sequence | Our prefill latencies are conservative (slower than reality) |
| **Global L3A / distributed cache** | We model it; vLLM doesn't have it yet | Our model is forward-looking (llm-d direction) |
| **TTL vs reactive eviction** | We support both: `eviction_policy="ttl"` or `"lru"` | ✅ Phase 3: LRU mode matches vLLM's reactive approach |

### Recommendations for Improving Alignment

1. **Block-level allocation**: Replace whole-context `CacheObject` with block chains (16-token blocks). This is the highest-impact change for fidelity.
2. **Hash-based matching**: Replace PrefixTrie with hash-based block matching for exact prefix comparison.
3. **Implicit sharing**: Allow identical blocks across sessions to share physical storage (ref counting per block).
4. **Reactive eviction**: Replace TTL-driven tier movement with LRU-on-demand eviction (evict only when space is needed).
5. **Preemption model**: Allow the scheduler to preempt running requests to free KV blocks.
6. **Chunked/batched prefill**: Model GPU utilization from batching multiple prefills together.

## References

- [vLLM Automatic Prefix Caching Design Doc](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM PagedAttention Design](https://docs.vllm.ai/en/stable/design/paged_attention/)
- [vLLM KV Offloading Connector](https://vllm.ai/blog/kv-offloading-connector)
- [KV-Cache Wins: From Prefix Caching to Distributed Scheduling](https://llm-d.ai/blog/kvcache-wins-you-can-see)
- [PagedAttention Paper (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
- [vLLM v1 Core Source](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core)
