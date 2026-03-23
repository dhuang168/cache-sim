# Research: LMCache and KV Cache Management Systems

Date: 2026-03-22

## 1. What is LMCache?

### Origins and Authors

LMCache was created by **Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang** at the **University of Chicago**. Yuhan Liu (5th-year PhD, co-advised by Junchen Jiang and Shan Lu) leads the project. The team later founded **Tensormesh** to commercialize the work.

- **GitHub**: https://github.com/LMCache/LMCache (7.7k stars, v0.4.2 as of March 2026)
- **Paper**: arXiv:2510.09665 (tech report, anonymous submission format)
- **Docs**: https://docs.lmcache.ai
- **Blog**: https://blog.lmcache.ai

### What Problem It Solves

LMCache is a **KV cache management layer** that sits between LLM inference engines (vLLM, SGLang) and heterogeneous storage. It extracts KV caches generated during prefill, stores them across multiple storage tiers, and reuses them for subsequent requests sharing the same prefix (or even non-prefix text via CacheBlend). This reduces TTFT by avoiding redundant prefill computation.

Key claim: **3-10x delay savings and GPU cycle reduction** for multi-round QA and RAG workloads; up to **15x throughput improvement** in benchmarks.

### Architecture Overview

LMCache has two main components:

1. **LMCache Worker (Data Plane)**: Handles KV cache extraction, storage, retrieval, and transfer. Connects to the inference engine's model runner and scheduler via a modular connector API.

2. **LMCache Controller (Control Plane)**: Exposes programmable APIs for cache management:
   - `lookup(tokens)` — find which instances have a prefix match
   - `move()` — migrate KV cache across instances
   - `clear()` — remove KV cache from specific storage
   - `pin()` — anchor cache in GPU memory (prevent eviction)
   - `compress()` — apply compression
   - `query_ip()` — map instance IDs to addresses

---

## 2. LMCache's Caching Architecture

### Chunk-Based Storage (Not Prefix-Tree)

LMCache uses **fixed-size chunk-based** storage, NOT a prefix trie:

- **Default chunk size: 256 tokens** (configurable via `chunk_size` parameter)
- Token sequences are divided into aligned chunks
- Each chunk gets a **rolling hash** (configurable: `builtin` Python hash or `sha256`)
- The hash is stored in a `ChunkedTokenDatabase` mapping token-hashes to chunk locations
- `CacheEngineKey` = (chunk hash, model name, layer range, KV format)

**Key insight**: Chunks are position-independent via hashing. Identical text sequences hash identically regardless of position, enabling deduplication. This is different from our simulator's prefix trie approach.

### Storage Tiers

LMCache supports a **5-tier hierarchy**:

| Tier | Backend | Typical Latency | Default Capacity | Notes |
|------|---------|-----------------|-------------------|-------|
| GPU | Active working set | N/A | Managed by engine | Not directly managed by LMCache |
| CPU DRAM | `LocalCPUBackend` | <1ms | 5 GB (default) | Pinned memory, greedy pre-allocation |
| Local Disk | `LocalDiskBackend` | 1-10ms | Disabled by default | One file per chunk, async writes |
| Remote | `RemoteBackend` (Redis, Mooncake, InfiniStore, S3, Valkey, etc.) | 10-100ms | Unlimited | Persistent, cross-cluster sharing |
| P2P | `P2PBackend` (NIXL) | 5-50ms | Peer GPU capacity | Direct GPU-GPU via RDMA/NVLink |
| PD | `PDBackend` | Variable | N/A | Disaggregated prefill-decode transfers |

**Storage backends supported**: NFS, WEKA, GPU-Direct Storage (GDS), Mooncake Store, NIXL, S3, InfiniStore, Valkey (Redis fork). Four processor types: NVIDIA, AMD, Ascend, TPU.

### Eviction Policies

- **Primary policy: LRU** (configurable, also supports LFU, FIFO)
- No TTL-based tier migration (unlike our simulator)
- **Tier promotion on access**: accessing a chunk on a slow tier copies it to faster tiers
- **Weighted semaphores** rate-limit concurrent I/O per backend
- **Priority system**: `priority_limit` parameter can filter which requests get cached

### Prefix Matching

- `ChunkedTokenDatabase` tracks all cached token sequences
- On query arrival, LMCache checks for prefix matches at chunk boundaries and returns matched token count
- The scheduler adjusts prefill requirements accordingly (only computes uncached portion)
- **Partial hits at chunk boundaries**: if 600 tokens are cached but chunk_size=256, only 512 tokens (2 full chunks) are reused; remaining 88 tokens are recomputed

### Cross-Request Sharing

- **Chunk-level deduplication**: identical chunks hash identically, so multiple requests sharing a system prompt reuse the same cached chunks (reference counted)
- **CacheBlend** (advanced feature): enables reuse of **non-contiguous** cached chunks for RAG scenarios where retrieved documents appear in different orders. Selectively recomputes a small subset of critical tokens (~15% by default via `blend_recompute_ratios`). Published at EuroSys'25 (Best Paper).

---

## 3. Comparison with Our Simulator

### Mapping of Concepts

| Our Simulator | LMCache | Notes |
|---------------|---------|-------|
| L1 (HBM per-GPU) | GPU memory (managed by vLLM/SGLang) | LMCache doesn't directly manage GPU KV; it manages offloaded copies |
| L2 (DRAM per-worker) | CPU DRAM (`LocalCPUBackend`, 5 GB default) | Similar concept but different defaults |
| L3A (SSD per-worker or global) | Local Disk (`LocalDiskBackend`) + Remote storage | LMCache separates local disk from remote |
| TTL-based tier migration | No TTL — tier promotion on access, LRU eviction | **Major difference**: we push objects down on timer; LMCache pulls up on demand |
| Watermark-based L1 eviction | LRU eviction when capacity exceeded | We use high/low watermarks; LMCache uses simple LRU |
| Prefix trie (token-hash-based) | ChunkedTokenDatabase (rolling hash per chunk) | Both hash-based but different structures |
| Block-based allocation (configurable block size) | Fixed chunk size (default 256 tokens) | Similar concept; LMCache calls them "chunks" |
| Per-session KV objects | Per-chunk deduplication | LMCache deduplicates at chunk level; we store full per-session objects |
| Shared system prefix (single shared object) | Automatic chunk-level deduplication | LMCache's approach is more granular |
| Push/Pull dispatch for multi-node | Controller-based dispatch + P2P | LMCache has richer multi-node coordination |
| `is_cache_worthwhile()` break-even check | No explicit break-even check | We model whether restore is faster than recompute |
| Prefill/decode oracle (latency models) | No latency modeling (it's a real system) | We simulate; LMCache measures |

### What LMCache Has That We Don't

1. **CacheBlend / Non-prefix reuse**: LMCache can reuse cached chunks that are NOT prefixes (e.g., RAG document chunks in any order), selectively recomputing ~15% of tokens. Our simulator only models prefix-based cache hits.

2. **Compression (CacheGen)**: LMCache supports lossy KV cache compression (3.5-4.3x reduction) via `remote_serde=cachegen`. Reduces bandwidth requirements for remote storage. We don't model compression.

3. **P2P direct GPU-GPU transfer**: Via NIXL/RDMA, LMCache can transfer KV cache directly between GPUs without going through CPU. We model inter-node transfers but always through the storage tier hierarchy.

4. **Disaggregated prefill-decode (PD)**: First-class support for separating prefill and decode onto different GPUs, with KV cache transfer between them. Our simulator co-locates prefill and decode on the same node.

5. **Layer-wise pipelining**: LMCache loads KV cache layer-by-layer, overlapping compute and I/O. Only needs a single layer's buffer in GPU memory. We model bulk transfers.

6. **Dynamic offloading**: Three-pointer mechanism for progressive GPU-to-CPU offloading during inference, without blocking computation. We model instantaneous placement.

7. **Controller-based orchestration**: Centralized cache directory with lookup, move, pin, clear APIs. Our dispatcher is simpler (push affinity or pull queue).

8. **Multiple remote backends**: Redis, S3, Mooncake, InfiniStore, Valkey, WEKA, NFS. We model a single L3A SSD tier.

9. **Decode cache storage**: Option to cache decode-phase KV as well (`save_decode_cache`). We only cache prefill-generated KV.

### What Our Simulator Models That LMCache Doesn't Explicitly Expose

1. **TTL-based tier migration**: Our scheduled demotion model (L1 -> L2 after TTL, L2 -> L3A after TTL). LMCache uses demand-driven promotion + LRU eviction.

2. **Watermark-based eviction**: High/low watermark thresholds for L1 eviction. LMCache uses simple capacity-exceeded LRU.

3. **Break-even analysis**: `is_cache_worthwhile()` explicitly models when restoring from a tier is slower than recomputing from scratch. LMCache assumes cache hits are always beneficial.

4. **Diurnal workload patterns**: NHPP arrival process with sinusoidal modulation. LMCache serves whatever requests arrive.

5. **Prefill/decode latency modeling**: Oracle-based models for compute time as a function of token count. LMCache measures real latency.

6. **Per-session context growth**: Sessions accumulate context over multiple turns, with prefix stability decay. LMCache sees each request independently.

7. **Explicit GPU slot modeling**: Prefill and decode slot pools with queue backpressure. LMCache relies on the inference engine's scheduler.

8. **Savings class tracking**: WORTHWHILE vs BREAK_EVEN classification of cache hits. LMCache doesn't categorize hits this way.

---

## 4. Integration with Serving Frameworks

### vLLM Integration

LMCache integrates with vLLM via the **KV Cache Connector** interface (`KVConnectorBase_V1`):

- **Scheduler side**: `get_num_new_matched_tokens(query)` checks cache before scheduling prefill
- **Worker side**: `start_load_kv()` / `wait_for_layer_load()` for async retrieval; `save_kv_layer()` / `wait_for_save()` for offloading
- Communication between scheduler and worker via **ZeroMQ IPC sockets**
- Layerwise pipelining: loads next layer while computing current layer

**Performance**: CPU offloading achieves **400 Gbps** transfer bandwidth (vs vLLM native at 88 Gbps). 2.3-14x throughput improvement at QPS=1 with 10K-token inputs.

### SGLang Integration

- LMCache also integrates with SGLang via `SGLangGPUConnector`
- Tested with Qwen3-32B on 2x H100 (TP=2)
- Primarily supports KV cache offloading (not full P2P yet)

### Disaggregated Serving

LMCache has first-class PD disaggregation support:
- **Prefill node** generates KV cache -> stored via LMCache
- **Decode node** retrieves KV cache from LMCache -> starts decoding
- Transfer via NIXL (RDMA/NVLink) for GPU-direct, or through CPU/network
- Mean TTFT reduction: 1.53-1.84x; mean ITL reduction: 1.12-1.66x

### llm-d Integration

**llm-d** is a Kubernetes-native distributed inference framework (backed by Red Hat) that uses LMCache for KV cache management:
- **KV cache indexer** maintains a global near-real-time view of KV cache block locality across pods
- **Cache-aware routing** directs requests to pods that already hold relevant context
- Supports disaggregated prefill/decode with intelligent scheduling

---

## 5. Do We Need LMCache Compatibility?

### What "LMCache Compatible" Would Mean

To make the simulator model LMCache's behavior, we would need to:

1. **Replace TTL-based tier migration with demand-driven promotion**:
   - Remove scheduled L1->L2 and L2->L3A TTL timers
   - Instead: on cache miss at a fast tier, promote from slower tier to faster tier
   - Evict from each tier only when capacity is exceeded (LRU)

2. **Use chunk-based deduplication instead of per-session objects**:
   - Store KV at 256-token chunk granularity
   - Hash-based deduplication: identical chunks across sessions share storage
   - This would dramatically reduce storage requirements vs our current model

3. **Model CacheBlend for RAG workloads**:
   - Allow non-prefix cache hits with partial recomputation (~15% of tokens)
   - This is a significant feature for RAG-heavy workloads

4. **Add compression modeling**:
   - CacheGen-style 3.5-4.3x compression for remote/disk tiers
   - Trade-off: compression/decompression CPU cost vs bandwidth savings

5. **Match LMCache configuration parameters**:
   - `chunk_size: 256` (we already have configurable block sizes)
   - `cache_policy: LRU/LFU/FIFO` (we use watermark + TTL)
   - `max_local_cpu_size: 5.0` GB default (our L2 is much larger)
   - `blend_recompute_ratios: 0.15` for CacheBlend

### Priority Assessment

| Feature | Difficulty | Value | Priority |
|---------|-----------|-------|----------|
| LRU eviction mode (alternative to watermark+TTL) | Low | High | P1 — easy to add as alternative policy |
| Chunk-level deduplication | Medium | High | P1 — changes storage model fundamentally |
| Demand-driven tier promotion | Medium | Medium | P2 — different philosophy from TTL push |
| CacheBlend (non-prefix reuse) | High | Medium | P3 — only matters for RAG workloads |
| Compression modeling | Medium | Low | P3 — bandwidth savings, modest complexity |
| PD disaggregation | High | High | P2 — already partially modeled via multi-node |
| Layer-wise pipelining | Low | Low | P4 — implementation detail, not strategic |

### Recommendation

Rather than making the simulator "LMCache compatible" by replacing our current model, **add LMCache-style behaviors as configurable alternatives**:

- Add `eviction_policy: "lru" | "watermark_ttl"` config option
- Add `deduplication: "chunk" | "per_session"` config option
- Add `tier_migration: "ttl_push" | "demand_pull"` config option

This lets us compare the two philosophies (TTL-push vs demand-pull, per-session vs chunk-dedup) within the same simulator framework.

---

## 6. Other KV Cache Systems

### CacheGen (SIGCOMM 2024)

- **Authors**: Same UChicago group (Yuhan Liu et al.)
- **Focus**: KV cache **compression and streaming** for network transfer
- **Key idea**: Custom tensor encoder exploiting KV cache distributional properties
- **Results**: 3.5-4.3x KV cache size reduction, 3.2-3.7x total delay reduction
- **Now integrated into LMCache** as the `cachegen` serialization format (`remote_serde=cachegen`)

### CacheBlend (EuroSys 2025, Best Paper)

- **Authors**: Same UChicago group
- **Focus**: Reusing **non-prefix** KV caches for RAG
- **Key idea**: Pre-compute and store KV cache for each RAG document chunk independently. At query time, load cached chunks in any order and selectively recompute ~15% of critical tokens to restore cross-attention quality.
- **Results**: 2.2-3.3x TTFT reduction, 2.8-5x throughput increase vs full recompute
- **Now integrated into LMCache** via `enable_blending=true`

### SGLang RadixAttention

- **Authors**: LMSYS (UC Berkeley — Lianmin Zheng et al.)
- **Data structure**: **Radix tree** (trie-like) for KV cache management at **token level**
- **Key difference from vLLM**: Token-level matching (not block-level), automatic discovery of reuse opportunities
- **Eviction**: LRU on radix tree nodes + cache-aware scheduling
- **Performance**: 85-95% hit rate for few-shot learning (vs vLLM's 15-25%); 75-90% for multi-turn chat (vs vLLM's 10-20%). ~10% latency advantage in large multi-turn conversations.
- **Our simulator's prefix trie is most similar to this approach**, though we match at block boundaries rather than individual tokens.

### vLLM Automatic Prefix Caching (APC)

- **Data structure**: **Block-level hash table** (Merkle-tree-like hashing)
- **Block hash**: `hash(parent_hash, block_tokens, extra_hashes)` where extra_hashes include LoRA IDs, multimodal input hashes, cache salts
- **Eviction**: LRU among blocks with ref_count=0; tie-breaking prefers evicting the end of the longest prefix
- **Hash algorithms**: `builtin` (Python hash) or `sha256` (collision resistant)
- **Key limitation**: Block-level matching means partial blocks can't be reused (same as our simulator's block-boundary rounding)
- **No cross-instance sharing**: APC is single-instance only; LMCache adds cross-instance sharing on top

### Comparison Matrix

| Feature | Our Simulator | LMCache | SGLang RadixAttn | vLLM APC |
|---------|--------------|---------|-------------------|----------|
| **Matching granularity** | Block (configurable) | Chunk (256 tokens) | Token-level | Block (16 tokens typical) |
| **Data structure** | Prefix trie | Hash table (ChunkedTokenDatabase) | Radix tree | Block hash table |
| **Eviction** | Watermark + TTL | LRU/LFU/FIFO | LRU + cache-aware scheduling | LRU (ref_count=0) |
| **Tier migration** | TTL-push (scheduled) | Demand-pull (on access) | N/A (GPU only) | N/A (GPU only) |
| **Storage tiers** | HBM, DRAM, SSD | GPU, CPU, Disk, Remote, P2P | GPU only | GPU only |
| **Cross-session sharing** | Shared system prefix object | Chunk-level deduplication | Automatic via radix tree | Automatic via block hashing |
| **Non-prefix reuse** | No | Yes (CacheBlend) | No (prefix only) | No (prefix only) |
| **Cross-instance** | Multi-node dispatch | Controller + P2P | No | No |
| **Compression** | No | Yes (CacheGen) | No | No |
| **PD disaggregation** | No (co-located) | Yes (first-class) | No | No |
| **Latency modeling** | Oracle-based simulation | Real measurement | Real measurement | Real measurement |

---

## Key Takeaways for the Simulator

1. **LMCache's chunk-based deduplication is more storage-efficient** than our per-session object model. Two users with the same 20k system prefix store it once in LMCache but twice in our simulator. This is the single biggest architectural difference.

2. **TTL-push vs demand-pull is a fundamental design choice**, not a bug. Our TTL model represents a proactive tier management policy; LMCache's demand-pull is reactive. Both are valid — the question is which better matches production behavior.

3. **CacheBlend (non-prefix reuse) is a differentiator** for RAG workloads but irrelevant for our current coding/chat/agent profiles where requests are sequential within a session.

4. **The 256-token chunk size in LMCache** is comparable to our configurable block sizes (16-4096 tokens). At 256 tokens for 70B FP16, one chunk = 256 * 327,680 bytes = ~80 MB. Our default block size discussion should reference this.

5. **LMCache does NOT use TTLs or watermarks** — it's pure capacity-based LRU eviction with demand-driven promotion. If we want to model LMCache-like behavior, we'd disable TTL migration and switch to LRU-only eviction.

6. **Production deployments** (Google Cloud, GMI Cloud, CoreWeave) validate that multi-tier KV caching is a real production pattern, not just academic research. Our simulator's 3-tier model is architecturally aligned with reality.

---

## Sources

- [LMCache Paper (arXiv:2510.09665)](https://arxiv.org/abs/2510.09665)
- [LMCache GitHub](https://github.com/LMCache/LMCache)
- [LMCache Architecture Docs](https://docs.lmcache.ai/developer_guide/architecture.html)
- [LMCache Configuration Reference](https://docs.lmcache.ai/api_reference/configurations.html)
- [LMCache DeepWiki](https://deepwiki.com/LMCache/LMCache)
- [LMCache Blog: Turboboosting vLLM](https://blog.lmcache.ai/2024-09-17-release/)
- [LMCache Blog: CacheBlend](https://blog.lmcache.ai/2024-10-09-cacheblend/)
- [LMCache on Google Kubernetes Engine](https://cloud.google.com/blog/topics/developers-practitioners/boosting-llm-performance-with-tiered-kv-cache-on-google-kubernetes-engine)
- [CacheGen Paper (SIGCOMM 2024)](https://arxiv.org/abs/2310.07240)
- [CacheBlend Paper (EuroSys 2025)](https://arxiv.org/abs/2405.16444)
- [SGLang RadixAttention Blog (LMSYS)](https://lmsys.org/blog/2024-01-17-sglang/)
- [vLLM Automatic Prefix Caching Design](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM LMCache Examples](https://docs.vllm.ai/en/latest/examples/others/lmcache/)
- [SGLang vs vLLM Prefix Caching Comparison](https://medium.com/byte-sized-ai/prefix-caching-sglang-vs-vllm-token-level-radix-tree-vs-block-level-hashing-b99ece9977a1)
- [llm-d KV Cache Architecture](https://llm-d.ai/docs/architecture/Components/kv-cache)
- [llm-d Cache-Aware Routing](https://llm-d.ai/blog/kvcache-wins-you-can-see)
- [Samsung: Scaling AI Inference with KV Cache Offloading](https://download.semiconductor.samsung.com/resources/white-paper/scaling_ai_inference_with_kv_cache_offloading.pdf)
- [Multi-tier KV Cache Storage (Springer, 2025)](https://link.springer.com/article/10.1007/s40747-025-02200-4)
- [LMCache vLLM Sharing KV Cache Docs](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/sharing-kv-cache.html)
