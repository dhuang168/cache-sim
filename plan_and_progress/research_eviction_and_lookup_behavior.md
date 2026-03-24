# Research: Eviction and Lookup Behavior in Production KV Cache Systems

Date: 2026-03-23

## 1. LMCache: Chunk Eviction and Cache Lookup

### Eviction: Plain LRU, NOT position-aware

Source: `lmcache/storage_backend/evictor/lru_evictor.py` (dev branch)

LMCache's LRU evictor uses a Python `OrderedDict`. On get, the accessed key is moved to the end (`cache_dict.move_to_end(key)`). On put, when capacity is exceeded, it iterates from the front (oldest) and evicts until enough space is freed. There is **no position awareness** -- any chunk can be evicted regardless of its position in a prefix chain. The evictor has no knowledge of token positions, prefix structure, or chunk ordering. It simply evicts the least-recently-used chunk by access time.

Key code:
```python
# lru_evictor.py update_on_put()
while cache_size + self.current_cache_size > self.MAX_CACHE_SIZE:
    evict_key = next(iter_cache_dict)  # front of OrderedDict = LRU
    evict_cache_size = self.get_size(cache_dict[evict_key])
    self.current_cache_size -= evict_cache_size
    evict_keys.append(evict_key)
```

### Cache Lookup: Requires consecutive prefix chunks from position 0

Source: `lmcache/v1/cache_engine.py` (dev branch)

LMCache's lookup returns an integer: `matched_prefix_length`. The lookup iterates through chunks sequentially and **breaks at the first missing chunk**:

```python
# In the non-layerwise path:
hit_chunks, block_mapping = self.storage_manager.batched_contains(keys, search_range, pin)
for idx, (start, end, key) in enumerate(chunk_info_list):
    if idx < hit_chunks:
        res = end
        continue
    return res  # stops at first gap
```

In the retrieve path (actual KV loading), the same sequential-break behavior:
```python
# _process_tokens_internal (non-async path, prefix-only mode):
memory_obj = memory_obj_map.get(key)
if memory_obj is None:
    # returned chunks are expected to be contiguous.
    # break at the first missing chunk.
    break
```

**Implication**: If LRU evicts a chunk at position N, all chunks after position N become unreachable even though they still exist in storage. This creates "orphaned" chunks that waste storage until they are eventually evicted themselves.

### CacheBlend mode (non-contiguous)

LMCache also has a `_process_tokens_internal` method that handles non-contiguous chunks, used when `enable_blending` is True. In this mode, it can retrieve chunks from multiple storage locations and handles gaps by tracking `ret_mask` and `last_failed_block_start`. But this is the CacheBlend path, not the default prefix-caching path.

## 2. CacheBlend: Non-Contiguous Partial Cache Hits

Source: arXiv:2405.16444 (EuroSys'25 Best Paper)

### Core mechanism
CacheBlend enables reuse of **non-prefix KV caches**. Unlike standard prefix caching (which requires consecutive chunks from position 0), CacheBlend can load cached KV for arbitrary text segments in the input.

### Selective recomputation
After loading non-contiguous cached KV chunks, CacheBlend identifies tokens with the **highest KV deviation** (HKVD) -- tokens whose precomputed KV values differ most from what full attention would produce given the surrounding context. It then selectively recomputes KV for only these high-deviation tokens.

- Typical recomputation fraction: **5-18%** of tokens (usually ~10-20%)
- Less than 15% recomputation typically preserves output quality
- The key insight: attention is sparse, so most tokens' KV values are insensitive to surrounding context

### Positional encoding handling
For non-prefix chunks, positional encoding is corrected via RoPE rotation matrices. Since RoPE attention scores depend only on relative position (not absolute position), the K vectors can be multiplied by a rotation matrix to adjust from the cached position to the correct position in the new input.

### Performance impact
In RAG workloads: ~3x TTFT speedup compared to prefix-only caching, because it can reuse cached KV for retrieved document chunks regardless of their position in the prompt.

## 3. vLLM Automatic Prefix Caching (APC): Position-Aware Eviction

### Eviction: LRU with position-aware tie-breaking (tail-first)

Source: `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/single_type_kv_cache_manager.py`

vLLM uses a `FreeKVCacheBlockQueue` (doubly-linked list) with this eviction order:
1. Only evict blocks with `ref_cnt == 0` (not in active use)
2. Among ref_cnt=0 blocks: **LRU** (least recently used first)
3. Tie-breaking: blocks with **more hash tokens** (deeper in the prefix chain / tail blocks) are evicted first

This is achieved by **freeing blocks in reverse order** when a request completes:
```python
# single_type_kv_cache_manager.py free()
# Free blocks in reverse order so that the tail blocks are freed first.
ordered_blocks = reversed(req_blocks)
self.block_pool.free_blocks(ordered_blocks)
```

Since `free_blocks` appends to the free queue, and the free queue is FIFO (LRU), the tail blocks end up at the front of the free queue and get evicted first. This preserves prefix chains: shared prefixes (near position 0) survive longer than tail blocks (unique to individual requests).

### Cache Lookup: Requires consecutive prefix from position 0

Source: `vllm/v1/core/single_type_kv_cache_manager.py` `FullAttentionManager.find_longest_cache_hit()`

vLLM's prefix matching iterates through block hashes sequentially and **breaks at the first miss**:
```python
for block_hash in itertools.islice(block_hashes, max_num_blocks):
    if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
        for computed, cached in zip(computed_blocks, cached_block):
            computed.append(cached)
    else:
        break  # stops at first gap
```

**Critical**: Block hashes are **chained** -- each block's hash depends on all preceding tokens: `hash(prefix_tokens, tokens_in_this_block)`. So even if a block's content exists in cache, if any preceding block was evicted, the hash chain is broken and the block cannot be found. This means evicting a middle block effectively invalidates all subsequent blocks.

### The RFC acknowledged the gap problem

The original vLLM APC RFC (#2614) explicitly acknowledged that eviction "may randomly free a block in the middle of a prefix," and noted that handling non-contiguous attention (pattern like "ooooxxxxxooooxxxoooxxxooooxxx") would require a special attention kernel. This was deferred to Phase 2 and has NOT been implemented as of 2026.

## 4. SGLang RadixAttention: Leaf-First LRU Eviction

### Eviction: LRU on leaf nodes only, recursive promotion

Source: `python/sglang/srt/mem_cache/radix_cache.py`

SGLang uses a radix tree where each node stores a segment of tokens and corresponding KV cache indices. Eviction is **strictly leaf-first**:

```python
def evict(self, params):
    leaves = list(self.evictable_leaves)
    eviction_heap = [(self.eviction_strategy.get_priority(node), node) for node in leaves]
    heapq.heapify(eviction_heap)

    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)
        # If parent becomes a leaf (no children, not locked), add to heap
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))
```

Key properties:
- **Only leaf nodes are evictable** (tracked in `evictable_leaves` set)
- **lock_ref > 0** protects nodes from eviction (active requests increment lock_ref on all ancestor nodes)
- When a leaf is deleted, its parent may become a new leaf and get added to the eviction heap
- This means eviction naturally works from the **end of sequences inward** toward shared prefixes
- Supports both LRU and SLRU (Segmented LRU) strategies

### Cache Lookup: Longest prefix match from root

SGLang's `match_prefix` walks the radix tree from root, following edges that match the input tokens. It returns the longest matching prefix and the tree node where the match ended. Since the radix tree structure inherently encodes prefix relationships, there are **no gaps by construction** -- if a node exists, all its ancestors exist. Evicting a leaf never creates orphaned interior nodes.

### Why radix tree avoids the orphan problem

In SGLang's design, the structure prevents gaps:
- Evicting a leaf removes the suffix of a sequence
- The parent (prefix) remains intact
- New requests matching the prefix still get a cache hit
- The orphan problem (chunks in storage but unreachable due to broken prefix chain) cannot occur

## Summary: Implications for Our Simulator

| System | Eviction | Lookup | Gap Handling |
|--------|----------|--------|--------------|
| LMCache (prefix mode) | Plain LRU, position-unaware | Consecutive prefix from pos 0, breaks at first miss | Orphaned chunks waste storage |
| LMCache (CacheBlend mode) | Same LRU | Non-contiguous allowed | Selective recompute for 10-20% of tokens |
| vLLM APC | LRU with tail-first tie-breaking | Consecutive prefix from pos 0, breaks at first miss | Hash chains break on middle eviction |
| SGLang RadixAttention | Leaf-first LRU (only leaves evictable) | Longest prefix from root, no gaps possible | Gaps impossible by construction |

### Key takeaways for simulator design:
1. **Our simulator should model prefix-chain-aware eviction** -- vLLM and SGLang both prefer evicting tail blocks to preserve shared prefixes. Plain LRU (like LMCache) creates orphaned chunks.
2. **Lookup should break at first gap** -- this is universal in prefix-caching mode. Non-contiguous reuse (CacheBlend) is a separate feature requiring selective recomputation.
3. **SGLang's radix tree is the gold standard** for avoiding orphaned cache entries. Our trie-based approach aligns with this.
4. **vLLM's reverse-order free is clever** -- by freeing tail blocks first, they naturally appear at the front of the LRU queue and get evicted first, preserving prefix chains without needing tree structure.
