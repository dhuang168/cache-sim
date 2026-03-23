# Research: Disaggregated Continuous Batching for LLM Serving

**Date:** 2026-03-22
**Purpose:** Inform cache simulator enhancements for modeling disaggregated prefill-decode architectures
**Status:** Research complete, ready for planning phase

---

## 1. What Is Disaggregated Continuous Batching?

### Standard Continuous Batching (Colocated)
In standard continuous batching (vLLM/Orca-style), prefill and decode run on the **same GPU**:
- New requests get prefilled immediately, preempting ongoing decodes ("prefill stall")
- Prefill is compute-bound (90-95% GPU utilization, 200-400 ops/byte)
- Decode is memory-bandwidth-bound (20-40% GPU utilization, 60-80 ops/byte)
- These two phases **interfere**: when a long prefill runs, all decode tokens stall; when a large decode batch occupies memory, prefill throughput drops

### Disaggregated Inference (Prefill-Decode Separation)
Disaggregated inference assigns prefill and decode to **separate GPU pools**:
1. **Prefill nodes** process input tokens, compute KV cache
2. **KV cache is transferred** from prefill node to decode node (via NVLink, RDMA, or network)
3. **Decode nodes** run continuous batching on decode-only workloads

This eliminates cross-phase interference entirely. Each phase can be independently scaled, parallelized, and even run on different hardware.

### Key Papers and Systems
| System | Venue | Key Contribution |
|--------|-------|-----------------|
| **DistServe** | OSDI 2024 | First to formalize goodput optimization via P/D disaggregation; 7.4x more requests at SLO |
| **Splitwise** | ISCA 2024 | Heterogeneous hardware (H100 prefill / A100 decode); 1.4x throughput at 20% lower cost |
| **TetriInfer (ShuffleInfer)** | arXiv 2024 | Chunked prefill + disaggregation + two-level scheduling; 97% TTFT improvement, 38% less resources |
| **Mooncake** | FAST 2025 (Best Paper) | KVCache-centric architecture; production system for Kimi; 59-498% capacity increase |
| **DeepSeek V3** | Production 2024-25 | MoE + disaggregated P/D + expert parallelism; 70KB/token KV (MLA compression) |
| **NVIDIA Dynamo** | GTC 2025 | Production framework with LLM-aware KV routing, hierarchical caching, up to 30x throughput |
| **LMCache** | arXiv 2025 | KV cache layer for cross-engine/cross-query reuse; up to 15x throughput improvement |

### Industry Adoption (as of late 2025)
Almost every production-grade LLM serving framework runs disaggregated inference: NVIDIA Dynamo, llm-d, Ray Serve LLM, SGLang, vLLM, LMCache, and Mooncake. The shift accelerated in 2025 as businesses moved from throughput optimization to latency SLO compliance.

---

## 2. Impact on Decode Utilization

### Why Disaggregation Improves Decode Fill Rates

**The core problem with colocated serving:**
- Decode is memory-bandwidth-bound and needs large batch sizes (64-256+) to approach compute saturation
- But prefill interruptions create "bubbles" -- decode tokens stall while a prefill runs
- With continuous batching, the scheduler must balance TTFT (wants immediate prefill) against TPOT (wants uninterrupted decode)
- Result: decode batches are undersized and frequently interrupted

**How disaggregation fixes this:**
1. **No prefill bubbles**: Decode nodes never run prefill, so decode batches run uninterrupted
2. **Predictable batch sizes**: The decode scheduler knows exactly how many sequences are active and can maintain optimal batch sizes
3. **Steady KV cache arrivals**: Prefill nodes stream completed KV caches to decode nodes at a predictable rate, allowing the decode scheduler to plan admissions
4. **Independent scaling**: Can allocate more decode nodes when decode is the bottleneck, more prefill nodes when prefill is the bottleneck

**Utilization improvements reported:**
- DistServe: 2.0-3.41x higher goodput on chatbot workloads; 4.48x higher goodput with 10.2x tighter SLO vs vLLM
- Splitwise: 1.4x throughput at 20% lower cost; or 2.35x throughput at same cost
- TetriInfer: 97% lower average TTFT, 47% lower JCT, using 38% fewer resources
- Mooncake: 59-498% more requests served within SLO (production Kimi deployment)

### Prefill-Decode Node Ratio
There is **no fixed optimal ratio** -- it depends on workload characteristics:
- Short prompts + long outputs: few prefill nodes, many decode nodes
- Long prompts + short outputs: many prefill nodes, few decode nodes
- DeepSeek V3 deployment: 3 prefill nodes : 9 decode nodes (1:3 ratio) with 8 H100 GPUs each
- Profile-driven search is the recommended approach (DistServe, HexGen-2)

### Heterogeneous Hardware Opportunity
Splitwise demonstrated that prefill can use high-compute GPUs (H100) while decode can use lower-cost GPUs (A100) since decode is memory-bandwidth-bound, not compute-bound. This reduces TCO by 20% while maintaining performance.

---

## 3. Impact on KV Cache Management

### 3.1 KV Cache Transfer Between Prefill and Decode Nodes

**Transfer is the critical new overhead** that does not exist in colocated serving.

**Transfer mechanisms (by speed):**
| Interconnect | Bandwidth | Typical Latency (2K tokens, 70B model) | Use Case |
|-------------|-----------|----------------------------------------|----------|
| NVLink (intra-node) | 600-900 GB/s | < 1 ms | P/D on same node |
| PCIe 5.0 x16 (8-way) | 64 GB/s per link, ~512 GB/s total | ~17.6 ms (OPT-175B) | Intra-node, different GPUs |
| RDMA (InfiniBand/RoCE) | 50-100 GB/s (400-800 Gbps) | 5-50 ms | Inter-node, same rack |
| Ethernet (TCP) | 12.5-50 GB/s (100-400 Gbps) | 10-100 ms | Cross-rack |

**Key finding from DistServe:** With proper placement, KV transfer overhead can be < 1 decode step:
- For OPT-175B at 2048 tokens: `2048 tokens * 4.5 MB/token / (64 GB/s * 8) = 17.6 ms`
- One decode step for OPT-175B on A100: 30-50 ms
- So transfer is pipelineable and does not dominate latency

**Key finding from Splitwise:** KV transfer overhead is < 7% of prefill computation time.

**Mooncake bandwidth numbers (production):**
- CPU RDMA (DRAM): ~47 GB/s
- GPU RDMA (VRAM via GPUDirect): ~15 GB/s
- Network: up to 800 Gbps (100 GB/s) per node with 8x400 Gbps RDMA NICs

### 3.2 KV Cache Size Per Token (Model-Dependent)

| Model | KV Size Per Token | Notes |
|-------|------------------|-------|
| LLaMA-70B (FP16, GQA) | ~327 KB | Standard attention, 8 KV heads |
| OPT-175B | ~4.5 MB | Full MHA, no GQA |
| DeepSeek-V3 (MLA) | ~70 KB | Multi-head Latent Attention compresses KV |
| LLaMA-8B (FP16) | ~65 KB | Smaller model |

**Transfer time examples for a 60K-token context (70B model, 327 KB/token = ~19.2 GB):**
- NVLink intra-node: ~21 ms
- RDMA (50 GB/s): ~384 ms
- PCIe 5.0 (64 GB/s): ~300 ms

For MLA-compressed models like DeepSeek-V3 (70 KB/token = ~4.1 GB for 60K tokens):
- RDMA (50 GB/s): ~82 ms
- This is a game-changer for disaggregation feasibility at long contexts

### 3.3 How Disaggregation Changes Caching Strategy

**New caching paradigm: KV cache as a shared distributed resource**

In colocated serving, KV cache lives on the GPU that processed the request. In disaggregated serving:

1. **Prefill nodes are stateless (or semi-stateless)**: After prefill, KV is transferred to decode node. Prefill node can discard it or keep it for prefix caching.

2. **Decode nodes are the primary KV cache holders**: They hold active decode sessions. When decode completes, the KV can be:
   - Kept in GPU memory (L1) for potential reuse
   - Offloaded to CPU DRAM (L2) for the decode node
   - Written to SSD (L3A) for long-term storage

3. **Distributed KV cache pool (Mooncake model)**:
   - Pool underutilized CPU DRAM and SSD across ALL nodes (prefill + decode)
   - A global scheduler (Conductor) tracks KV cache locations
   - When a new request arrives with a prefix match, route to the node holding the cached KV
   - Or transfer the cached KV to the chosen prefill node before prefill begins

4. **Two-phase cache flow**:
   - **Prefix cache reuse**: Before prefill, check if prefix KV exists in the distributed pool. If yes, load it to the prefill node (saves recomputation).
   - **KV transfer after prefill**: Ship completed KV from prefill node to decode node.
   - **Post-decode storage**: After decode completes, store KV for future prefix reuse.

**Mooncake's 4-step request workflow:**
1. **KVCache Reuse**: Load existing prefix KV from distributed pool to selected prefill instance
2. **Incremental Prefill**: Process only the non-cached portion of input
3. **KVCache Transfer**: Transfer complete KV layer-wise to designated decode instance
4. **Decoding**: Join request into continuous batching on decode instance

**Impact on prefix caching efficiency:**
- Mooncake: With 128K-token input, prefix caching reduces prefill time by **92%**
- The distributed cache pool means prefix KV can be reused even across different prefill nodes
- Cache-aware routing is critical: the scheduler must consider KV cache location when choosing prefill node

---

## 4. Oracle Config Implications for the Simulator

### 4.1 Prefill Latency Changes (Dedicated Prefill Nodes)

**What changes:**
- No interference from decode -- prefill gets 100% of GPU compute
- Prefill latency becomes **more predictable** (lower variance)
- Can use higher tensor parallelism (TP) for prefill since no memory reserved for decode KV cache
- Chunked prefill is still useful for scheduling granularity but not for decode protection

**Specific implications for the simulator:**
- The existing `PrefillOracle` table (tokens -> latency) remains valid as a base model
- But latency should be **lower and more consistent** than colocated estimates
- DistServe showed that tailoring TP for prefill separately reduces average prefill latency
- Suggested parameter: `prefill_latency_multiplier` (default 1.0 for colocated; ~0.7-0.9 for disaggregated, reflecting no interference)

### 4.2 Decode Latency Changes (Dedicated Decode Nodes)

**What changes:**
- No prefill bubbles -- decode batch runs uninterrupted
- Batch sizes are **predictable and higher** (can maintain near-optimal fill)
- Memory is fully dedicated to KV cache for active sequences + decode weights
- The sqrt batch degradation model may need adjustment: with stable batches, per-token latency is more predictable

**Specific implications for the simulator:**
- The existing `DecodeOracle` (sqrt batch degradation) is still directionally correct
- But effective batch sizes will be higher and more stable
- In colocated mode, effective decode batch fluctuates wildly (drops to 0 during prefill, then refills)
- In disaggregated mode, effective decode batch stays near the target fill level
- Suggested parameter: `decode_batch_fill_factor` (0.0-1.0; colocated ~0.4-0.6 average due to bubbles; disaggregated ~0.8-0.95)

### 4.3 New Parameters Needed

| Parameter | Type | Description | Typical Value |
|-----------|------|-------------|---------------|
| `disaggregated` | bool | Enable P/D disaggregation | `false` (default) |
| `n_prefill_nodes` | int | Dedicated prefill GPU count | Already exists |
| `n_decode_nodes` | int | Dedicated decode GPU count | **NEW** |
| `kv_transfer_bandwidth_bytes_per_s` | int | P->D KV transfer bandwidth | 50-100 GB/s (RDMA) |
| `kv_transfer_latency_floor_us` | int | Fixed overhead per transfer | 1000-5000 us |
| `prefill_latency_multiplier` | float | Prefill speedup from no interference | 0.7-1.0 |
| `decode_batch_fill_factor` | float | Average decode batch utilization | 0.4-0.95 |
| `kv_cache_pool_enabled` | bool | Distributed KV cache pool (Mooncake-style) | `false` |
| `prefill_node_cache_retention` | bool | Whether prefill nodes keep KV after transfer | `true` |

### 4.4 New Event Flow for Disaggregated Mode

Current flow:
```
ARRIVAL -> CACHE_LOOKUP -> [HIT|MISS] -> PREFILLING -> DECODE_QUEUED -> DECODING -> KV_WRITE -> COMPLETE
```

Disaggregated flow adds a KV transfer step:
```
ARRIVAL -> CACHE_LOOKUP -> [HIT|MISS] -> PREFILLING (on prefill node)
  -> KV_TRANSFER (prefill -> decode node) -> DECODE_QUEUED -> DECODING -> KV_WRITE -> COMPLETE
```

The KV_TRANSFER event models:
- Transfer time = `kv_size_bytes / kv_transfer_bandwidth + kv_transfer_latency_floor_us`
- During transfer, the decode slot is not yet occupied (it is reserved but not consuming compute)
- Transfer can be pipelined with ongoing decodes on the decode node

---

## 5. Production System Reports

### Mooncake (Moonshot AI / Kimi) -- FAST 2025 Best Paper
- **Scale**: Thousands of nodes, 100+ billion tokens/day
- **Architecture**: Fully disaggregated P/D with distributed KV cache pool (CPU DRAM + SSD across all nodes)
- **Network**: RDMA up to 8x400 Gbps per node
- **Prefix caching**: 92% prefill time reduction at 128K tokens
- **Throughput**: 115% more requests on A800, 107% more on H800 vs baseline
- **Capacity**: 59-498% more requests within SLO on real traces
- **Scheduler**: Conductor -- cache-aware routing that considers KV cache location, TTFT estimation, and load balancing simultaneously

### DeepSeek V3
- **Architecture**: Disaggregated P/D with MoE expert parallelism
- **KV efficiency**: MLA compresses KV to ~70 KB/token (vs ~327 KB for standard 70B)
- **Node ratio**: 3 prefill : 9 decode nodes (8 H100 each) in one reported deployment
- **Throughput**: 52.3K input tokens/s and 22.3K output tokens/s per node
- **Storage**: 3FS distributed file system over 400 Gbps RoCE for KV persistence

### NVIDIA Dynamo (GTC 2025)
- **Framework**: Open-source, production-ready disaggregated inference
- **KV management**: LLM-aware router with hash-based KV cache tracking, overlap scoring for cache-aware routing
- **Hierarchical caching**: GPU -> CPU DRAM -> SSD -> object storage (petabyte-scale)
- **Performance**: Up to 30x throughput improvement on DeepSeek-R1 with Blackwell
- **Topology**: Supports colocated P/D on same NVL72 rack for optimal KV transfer

### DistServe (UCSD, OSDI 2024)
- **Goodput**: 7.4x more requests served within SLO; 12.6x tighter SLO achievable
- **Chatbot workload**: 2.0-3.41x higher goodput vs vLLM
- **Code completion**: 3.2x higher goodput, 1.5x tighter SLO
- **KV transfer**: Placed P/D on same node to use high-speed NCCL; overhead < 1 decode step

### LMCache
- **Function**: Cross-engine KV cache reuse layer
- **Tiers**: GPU, CPU DRAM, local disk
- **Performance**: Up to 15x throughput improvement for multi-round QA and document analysis
- **Integration**: Works with vLLM and SGLang; supports both prefix reuse and P/D disaggregation

---

## 6. Implications for Cache Simulator Design

### What the simulator already models well:
- Multi-tier cache hierarchy (L1 HBM / L2 DRAM / L3A SSD) -- maps directly to Dynamo/Mooncake hierarchical caching
- Prefix-based cache lookup and reuse
- Separate prefill/decode slot pools
- TTL-based tier migration
- Multi-node dispatch with affinity

### What needs to change for disaggregated mode:
1. **Separate node pools**: Currently `n_prefill_nodes` handles both prefill and decode. Need separate prefill and decode node pools.
2. **KV transfer event**: New event type between prefill completion and decode queue admission, modeling transfer latency.
3. **Decode utilization model**: With disaggregation, decode batch fill should be higher and more stable. The decode oracle should reflect this.
4. **Prefill node caching**: Prefill nodes can retain KV for prefix reuse even after transferring to decode. This is a cache duplication that trades storage for compute.
5. **Distributed cache pool**: KV cache can be pooled across all nodes (Mooncake-style) rather than per-node. The simulator's L3A shared mode partially models this.
6. **Cache-aware scheduling**: Dispatch should consider KV cache location (which node has the prefix cached) -- the simulator already has affinity-based dispatch that partially models this.

### Priority ordering for implementation:
1. **KV_TRANSFER event** (highest impact, straightforward) -- add transfer latency between prefill and decode
2. **Decode batch fill model** -- parameterize the effective batch utilization
3. **Separate node pools** -- split existing nodes into prefill vs decode pools
4. **Prefill latency adjustment** -- multiplier for interference-free prefill
5. **Distributed KV cache pool** (stretch) -- global cache-aware routing

---

## Sources

- [DistServe: OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [DistServe paper (arxiv)](https://arxiv.org/abs/2401.09670)
- [DistServe blog: Throughput is Not All You Need](https://haoailab.com/blogs/distserve/)
- [Disaggregated Inference: 18 Months Later (Hao AI Lab)](https://haoailab.com/blogs/distserve-retro/)
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://www.cs.cmu.edu/~18742/papers/Patel2024.pdf)
- [TetriInfer/ShuffleInfer: Inference without Interference](https://arxiv.org/abs/2401.11181)
- [Mooncake: KVCache-centric Architecture (FAST 2025)](https://www.usenix.org/conference/fast25/presentation/qin)
- [Mooncake paper (arxiv)](https://arxiv.org/abs/2407.00079)
- [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- [DeepSeek V3 Technical Report](https://arxiv.org/html/2412.19437v1)
- [Deploying DeepSeek with PD Disaggregation (LMSYS)](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
- [NVIDIA Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA Dynamo 1.0 Production](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/)
- [LMCache paper](https://arxiv.org/abs/2510.09665)
- [vLLM Disaggregated Prefill docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [Prefill-decode disaggregation (BentoML Handbook)](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation)
- [FlowKV: Disaggregated Inference with Low-Latency KV Transfer](https://arxiv.org/html/2504.03775v1)
- [TraCT: CXL Shared Memory KV Cache](https://arxiv.org/html/2512.18194v1)
- [Cache-aware disaggregated inference (Together AI)](https://www.together.ai/blog/cache-aware-disaggregated-inference)
- [SPAD: Specialized Prefill and Decode Hardware](https://augustning.com/assets/papers/spad-arxiv-2025.pdf)
- [Perplexity blog: Disaggregated Prefill and Decode](https://www.perplexity.ai/hub/blog/disaggregated-prefill-and-decode)
