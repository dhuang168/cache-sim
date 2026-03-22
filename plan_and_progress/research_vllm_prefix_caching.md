# Research: vLLM Prefix Caching Architecture

Date: 2026-03-22

## Key Findings

### Hash-based block matching (not trie)
- KV cache partitioned into fixed-size blocks (default 16 tokens)
- SHA-256 parent-chained hashing: each block hash encodes entire causal history
- Single hash match on block N guarantees blocks 0-N are identical
- Hash inputs: parent hash + token IDs + extras (LoRA, multimodal, cache salt)

### Block management (PagedAttention)
- Physical blocks stored non-contiguously (paged)
- Block table per request maps logical→physical blocks
- Copy-on-write for shared blocks (parallel sampling)
- <4% memory waste (vs 60-80% with pre-allocation)

### Eviction: 3-tier LRU
- Never evict ref_cnt > 0 (active)
- LRU among ref_cnt == 0 (doubly-linked list, O(1) access)
- Tie-break: evict end of longest chain (preserve shared roots)
- Blocks linger after request completes — future requests can reclaim

### Multi-tier: GPU → CPU → Disk
- GPU HBM: primary KV cache
- CPU DRAM: swap target for preempted requests (async DMA)
- External storage: via Connector API (LMCache for Ceph/Redis)
- V1 default: recompute (not swap) — swap overhead > recompute for short prefills

### Multi-node
- TP: KV cache sharded by attention head ownership (centralized scheduler)
- PP: Each stage holds KV for its layers
- Scheduler is centralized (rank 0) — makes all block allocation/eviction decisions

### Shared prompts
- Implicit: same token blocks → same hash → shared physical blocks
- 100 requests with same system prompt share one physical copy
- Cache salt for multi-tenant isolation

## Sources
- vLLM APC Design Doc, DeepWiki, PagedAttention paper
- vLLM v1 source: kv_cache_manager.py, block_pool.py, kv_cache_utils.py
- GitHub RFCs: #2614, #16016, #16144, #33526, #36311
