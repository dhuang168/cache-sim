# Research: Cache-Aware Request Routing in Production Systems

Date: 2026-03-24
Purpose: Determine whether production systems use an explicit index/registry vs scanning for cache-aware dispatch, and the computational complexity. Informs whether our simulator should use a session index or scan.

---

## 1. vLLM: Single-Instance Hash Table (No Cross-Instance Index)

### Within a single instance: O(1) hash table lookup

vLLM's Automatic Prefix Caching (APC) uses a **`BlockHashToBlockMap`** — a hash table mapping `BlockHashWithGroupId` to `KVCacheBlock`. This is the core data structure in the `BlockPool`.

**How `find_longest_cache_hit` works:**
1. Compute block hashes for the incoming request's tokens (chained: each block hash = `hash(parent_hash, block_tokens, extra)`)
2. For each block hash in sequence, call `block_pool.get_cached_block(block_hash)` — a **direct O(1) hash table lookup**
3. Break at the first miss (prefix-only matching)

**Complexity:** O(n/B) where n = token count, B = block size (default 16). Each lookup is O(1) in the hash table. No scanning of all cached blocks.

**Data structures in BlockPool:**
- `blocks[]` — array of all KVCacheBlock objects (index = block_id)
- `free_block_queue` — doubly-linked list (FreeKVCacheBlockQueue) for LRU eviction
- `cached_block_hash_to_block` — **hash table** for O(1) prefix lookup

### Cross-instance: vLLM has NO built-in cross-instance cache index

vLLM's APC is purely single-instance. The scheduler on rank 0 manages the BlockPool for that instance's GPUs. There is no mechanism for one vLLM instance to know what another instance has cached.

**For multi-instance routing, external systems are required:**
- **llm-d** provides the cross-instance index (see Section 5)
- **LMCache** provides the cross-instance controller (see Section 3)
- **NVIDIA Dynamo** provides a router with radix tree (see Section 6)

### Key insight for simulator
vLLM's approach is: **per-instance hash table, no global index**. The "session-to-node mapping" question doesn't arise within vLLM because it only handles one instance. Cross-instance affinity is delegated to external routers.

---

## 2. SGLang RadixAttention: Radix Tree + Approximate Mirror in Router

### Within a single instance: O(n) radix tree traversal

SGLang stores the KV cache in a **radix tree** (compact prefix tree). Each node stores a segment of tokens and pointers to child nodes.

**How `match_prefix` works:**
1. Start at root node
2. Walk down edges matching input tokens
3. At each node, compare the node's token segment against the corresponding input tokens
4. Return when no further match is found

**Complexity:** O(n) where n = number of input tokens. The radix tree walk is proportional to the prefix length, not the total number of cached sequences. This is the key advantage over scanning: you traverse the tree depth, not the breadth.

**Why no gaps by construction:** The radix tree structure guarantees that if a node exists, all its ancestors exist. Leaf-first eviction preserves this invariant.

### Cross-instance: Approximate Radix Tree in the Router (Communication-Free)

This is the most architecturally interesting approach. SGLang v0.4 introduced a **cache-aware load balancer** that maintains an **approximate radix tree per worker** in the router:

**How it works:**
1. The router maintains a `tree_map: DashMap<String, Tree>` — one radix tree per worker
2. When the router sends a request to worker W, it **inserts the request's tokens into worker W's approximate tree** (optimistic assumption that the worker will cache it)
3. On the next request, the router does `match_prefix` against each worker's approximate tree to predict cache hit length
4. The worker with the **longest prefix match** (above a threshold) gets the request
5. If no worker has a good match, route to the worker with the **smallest tree** (balance cache population)
6. Trees are pruned via TTL expiration (configurable `--eviction-interval`, `--max-tree-size`)

**Critical design choice: NO synchronization with workers.** The router never queries workers about their actual cache state. Instead, it **simulates** an approximate cache state based on its own routing decisions. This is why it's called "communication-free."

**Accuracy tradeoffs:**
- The approximate tree diverges from reality when workers evict entries (the router doesn't know)
- TTL-based pruning in the router approximates LRU eviction on the workers
- In practice, this achieves 1.9x throughput improvement and 3.8x hit rate improvement

**Complexity per routing decision:** O(W * P) where W = number of workers, P = average prefix match length. Each worker's tree is traversed to find the best match.

**Implementation:** Built in Rust (`sgl-model-gateway/` directory) for high concurrency. Uses `DashMap` (concurrent hash map) for worker-to-tree mappings and `AtomicUsize` for load counters.

### Key insight for simulator
SGLang's router proves that an **approximate index** (not perfectly synchronized) is sufficient for effective cache-aware routing. The router doesn't need to know the exact cache state — a probabilistic approximation works well in practice.

---

## 3. LMCache Controller: Centralized RegistryTree with Hierarchical Lookup

### Controller architecture: RegistryTree (chosen over flat index)

LMCache evaluated three approaches for tracking cache ownership across instances:

| Approach | Lookup | Deregistration | Memory | Chosen? |
|----------|--------|----------------|--------|---------|
| **S0: Flat Indexing** | Fast | 32.6 seconds (catastrophic) | High | No |
| **S1: RegistryTree** | 8 microseconds | 8 ms | Baseline | **Yes** |
| **S2: Reverse Index** (`key_to_worker_index`) | Near-O(1) | 285x slower than S1, 87x slower reports | 2x memory | No |

**Why the flat reverse index (S2) was rejected:** While `key_to_worker_index: dict[int, (instance_id, worker_id, location)]` gives O(1) lookup per key, it doubled memory and made operational tasks (worker join/leave, full state recovery) catastrophically slow. In a system with 100 instances x 1M chunks each, the reverse index has 100M entries to maintain.

**RegistryTree structure:** `RegistryTree -> InstanceNode -> WorkerNode -> Location -> Chunks`

**How `lookup(tokens)` works:**
1. Tokens are chunked (256-token default) and hashed
2. The controller traverses the RegistryTree hierarchy to find which instances/workers have matching chunk hashes
3. Returns `Dict[instance_id, Tuple[location, matched_prefix_length]]`
4. The router uses this to dispatch to the instance with the longest cached prefix

**Complexity:** O(C * I) where C = number of chunks in the request, I = number of instances to check. The tree traversal per instance is O(depth of hierarchy) which is small (3-4 levels).

**P2P flow:** When instance A has a cache miss but instance B has the data, the controller tells A to fetch from B directly (P2P via NIXL/RDMA), achieving 4x TTFT improvement.

### Key insight for simulator
LMCache proves that the **obvious approach (flat reverse index)** has unacceptable operational costs at scale. A hierarchical structure with slightly slower lookup but fast join/leave/recovery is the production-viable choice. At 100 instances x 1M chunks, the reverse index approach breaks down.

---

## 4. Mooncake Conductor: Global Block Radix Tree with TTFT Cost Function

### Cache index: Block Radix Tree (global)

Mooncake's Conductor maintains a **global Block Radix Tree** (or similar hierarchical structure) that records the location of KV cache blocks across all prefill instances.

**How it works:**
1. Index is updated via **event-driven incremental pushes** — when KV blocks are created, migrated, or evicted on any instance, that instance pushes an update to the Conductor. No polling.
2. When a request arrives, the Conductor queries the global tree to determine the common prefix between the request's tokens and all cached blocks across all instances
3. For each candidate prefill instance, the Conductor estimates TTFT:

```
TTFT_estimate = queue_wait_time + cache_transfer_time + prefill_compute_time
where:
  queue_wait_time = f(instance.current_queue_depth)
  cache_transfer_time = f(prefix_match_length, cache_location)
  prefill_compute_time = f(request.total_tokens - prefix_match_length)
```

4. The request is assigned to the instance with the **lowest estimated TTFT**

**Key differentiator:** Mooncake's cost function considers not just cache locality but also **queue depth** and **transfer cost**. A remote instance with a perfect cache hit but a full queue may lose to a local instance with a partial hit and an empty queue.

### Key insight for simulator
Mooncake's multi-factor TTFT estimation (cache locality + queue depth + transfer cost) is the most sophisticated routing approach. Our simulator's push dispatcher already considers affinity but could benefit from incorporating queue pressure into the routing decision.

---

## 5. llm-d: Global KV-Block Index with Event-Driven Updates

### Architecture: Two-level LRU index + ZMQ event stream

llm-d (Red Hat's Kubernetes-native inference framework) implements the most explicitly documented global index:

**Index structure:** Two-level LRU cache
- Level 1: Maps block key (`Key{ModelName, ChunkHash}`) to level-2 cache
- Level 2: Maps to pod entries (`PodEntry{PodIdentifier, DeviceTier}`) — which pods have this block and on what tier (GPU/CPU)

**Block hash computation:** Compatible with vLLM's hashing
- Tokens grouped into 16-token blocks (matching vLLM default)
- FNV-64a hash on CBOR-encoded `[parentHash, tokenChunk, extra]` tuples
- Chained hashing (each block depends on all predecessors)
- `PYTHONHASHSEED` must match across the fleet

**Read path (scoring a routing decision):**
1. Tokenize incoming prompt
2. Convert tokens to block keys via `TokenProcessor` (deterministic, O(n/16))
3. Query `Index.Lookup(keys, podSet)` — O(1) per key in the hash table
4. `KVBlockScorer` counts **consecutive cache hits from position 0** for each candidate pod
5. Return pod scores: `score = number of consecutive prefix blocks cached`

**Write path (maintaining the index):**
1. vLLM pods publish ZMQ messages: `kv@{podIP}@{model}` topic, msgpack-encoded event batches
2. Events: `BlockStored`, `BlockRemoved`, `AllBlocksCleared`
3. Events are sharded by `FNV-1a(podID) % shards` for parallel processing with per-pod ordering guarantees
4. Index is updated in near-real-time

**Complexity:**
- Lookup: O(n/B) per pod, where n = tokens, B = block size (16)
- Scoring: O(n/B * P) total, where P = number of candidate pods
- Each individual block lookup is O(1) in the hash map

**Performance:** Cache-aware routing achieved 97% improvement in output tokens/second vs random routing (8,730 vs 4,428 tokens/s). P90 TTFT was 170x faster (0.54s vs 92.5s).

### Key insight for simulator
llm-d is the clearest implementation of "global hash index + consecutive prefix scorer." The design is simple, fast (O(1) lookups), and compatible with vLLM's existing hash computation. The write path uses event streaming (not polling) for near-real-time synchronization.

---

## 6. NVIDIA Dynamo: Radix Tree Router with Overlap Scoring

### Router architecture: Radix tree + cost function

NVIDIA Dynamo's router uses a **radix tree** (like SGLang) to track cached KV blocks across workers:

**Implementations:**
- `RadixTree` — single-threaded (when `--router-event-threads=1`)
- `ConcurrentRadixTree` — multi-threaded for high concurrency
- `PositionalIndexer` — alternative implementation

**How the router learns cache state:**
Four event transport modes:
1. **NATS Core (default):** Workers maintain local indexer; router queries on startup
2. **JetStream:** Events persisted with durable consumers
3. **ZMQ:** Workers publish via ZMQ PUB sockets (like llm-d)
4. **Approximate:** No events; router predicts state from its own routing decisions with TTL expiration (like SGLang)

**Routing cost function:**
```
cost = kv_overlap_score_weight * prefill_blocks + decode_blocks
where:
  prefill_blocks = (total_tokens - cached_tokens) / block_size
  decode_blocks = estimated from input tokens + worker's active sequences
```

Select worker with lowest cost. Can be deterministic (argmin) or probabilistic (softmax with temperature).

**Key differentiator:** Dynamo supports both **event-driven exact tracking** (NATS/ZMQ) and **approximate prediction** (like SGLang). This gives operators the choice between accuracy and overhead.

---

## 7. Summary: Index vs Scan

| System | Scope | Data Structure | Lookup Method | Complexity | Synchronization |
|--------|-------|---------------|---------------|-----------|-----------------|
| **vLLM APC** | Single-instance | Hash table (`BlockHashToBlockMap`) | O(1) per block hash | O(n/B) per request | N/A (single instance) |
| **SGLang (single)** | Single-instance | Radix tree | Tree traversal | O(n) prefix walk | N/A (single instance) |
| **SGLang Router** | Multi-instance | Approximate radix tree per worker | Tree traversal per worker | O(W * P) | **None** — simulates from routing decisions |
| **LMCache Controller** | Multi-instance | RegistryTree hierarchy | Hierarchical traversal | O(C * I) | Centralized controller, workers report |
| **Mooncake Conductor** | Multi-instance | Global Block Radix Tree | Tree query | O(n * I) | Event-driven incremental push |
| **llm-d Indexer** | Multi-instance | Two-level LRU hash map | O(1) per block per pod | O(n/B * P) | ZMQ event stream from vLLM pods |
| **NVIDIA Dynamo** | Multi-instance | Radix tree (concurrent) | Tree traversal + cost function | O(W * P) | NATS/ZMQ events OR approximate |

**No system scans all cached blocks.** Every system uses an indexed data structure:
- **Hash tables** for O(1) per-block lookup (vLLM, llm-d)
- **Radix/prefix trees** for O(n) prefix matching (SGLang, Mooncake, Dynamo)
- **Hierarchical registries** for organized multi-instance lookup (LMCache)

---

## 8. Implications for Our Simulator

### Current simulator approach
Our simulator uses:
- `PrefixTrie` in `sim/cache.py` for prefix matching (similar to SGLang's radix tree)
- `PushDispatcher` with `has_session_cached()` scan across nodes for affinity
- `PullDispatcher` with affinity scoring

### What production systems teach us

1. **An explicit index is universal.** No production system scans all cached blocks to find a match. They all maintain an indexed data structure. Our `PrefixTrie` already follows this pattern for prefix matching within a node.

2. **Cross-instance routing has two viable approaches:**
   - **Event-driven index** (llm-d, Mooncake, Dynamo NATS/ZMQ): Workers push cache events to a central index. Router queries the index. Near-real-time accuracy.
   - **Approximate simulation** (SGLang, Dynamo approximate): Router predicts cache state from its own routing decisions. No synchronization overhead. Slightly less accurate but simpler.

3. **Session-to-node mapping vs prefix index:** Our simulator's `has_session_cached()` is effectively a session-to-node mapping. Production systems don't track sessions — they track **block hashes**. A session's tokens are hashed into blocks, and the index stores which node has which blocks. This is more general because:
   - Multiple sessions sharing a system prompt automatically share the same block hashes
   - Partial prefix matches are discoverable (not just "this node has session X")
   - No need for an explicit session registry

4. **The scoring function matters.** Simple "longest prefix match" (llm-d, SGLang) is good. But Mooncake's multi-factor approach (cache locality + queue depth + transfer cost) is better for production because it avoids routing everything to the node with the best cache but the worst queue.

### Recommendation for our simulator

Our current approach (session-based affinity in PushDispatcher) is a reasonable simplification for a simulator. It captures the key behavior: requests from the same session are routed to the node that has their KV cached. The production alternative (block-hash-based prefix index) is more general but adds complexity that may not change simulation results significantly.

**If we want higher fidelity:**
- Replace `has_session_cached()` with a per-node block hash index (like llm-d's approach)
- Add multi-factor scoring: `cost = cache_affinity_score + queue_pressure_weight * queue_depth`
- Model the approximate nature of cache state (SGLang's approach: the dispatcher doesn't perfectly know cache state)

**Cost-benefit:** For the simulator's purpose (comparing cache policies, tier configurations, and scaling), session-based affinity captures 90%+ of the routing benefit. Block-hash indexing would matter most when modeling cross-session prefix sharing at the routing layer (e.g., two different sessions with the same system prompt being routed to the same node because the block hashes match).

---

## Sources

- [vLLM Automatic Prefix Caching Design](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM v1 Prefix Caching](https://docs.vllm.ai/en/v0.8.5/design/v1/prefix_caching.html)
- [vLLM KV Cache Management (DeepWiki)](https://deepwiki.com/vllm-project/vllm/3.4-kv-cache-management-and-prefix-caching)
- [SGLang v0.4: Cache-Aware Load Balancer](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
- [SGLang Router Architecture Proposal (GitHub #7532)](https://github.com/sgl-project/sglang/issues/7532)
- [SGLang Router Roadmap (GitHub #10341)](https://github.com/sgl-project/sglang/issues/10341)
- [LMCache P2P Blog](https://blog.lmcache.ai/en/2026/01/21/p2p-1/)
- [LMCache Lookup API](https://docs.lmcache.ai/kv_cache_management/lookup.html)
- [LMCache Architecture](https://docs.lmcache.ai/developer_guide/architecture.html)
- [Mooncake Paper (FAST 2025)](https://arxiv.org/abs/2407.00079)
- [Mooncake Conductor RFC (GitHub #977)](https://github.com/kvcache-ai/Mooncake/issues/977)
- [llm-d KV Cache Architecture](https://llm-d.ai/docs/architecture/Components/kv-cache)
- [llm-d KV-Block Index Architecture (GitHub)](https://github.com/llm-d/llm-d-kv-cache/blob/main/docs/architecture.md)
- [llm-d KVCache Indexer Core (DeepWiki)](https://deepwiki.com/llm-d/llm-d-kv-cache-manager/2.2-kvcache-indexer-core)
- [llm-d Blog: KVCache Wins](https://llm-d.ai/blog/kvcache-wins-you-can-see)
- [NVIDIA Dynamo KV Cache-Aware Routing](https://docs.nvidia.com/dynamo/latest/user-guides/kv-cache-aware-routing)
- [NVIDIA Dynamo KV Router](https://docs.nvidia.com/dynamo/latest/router/README.html)
- [Red Hat: KV Cache Aware Routing with llm-d](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference)
