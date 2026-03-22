# Phase 1: Block-Level Allocation — Plan

Date: 2026-03-22

## Goal

Replace whole-context `CacheObject` with block chains. Support configurable block sizes to model different production systems:

| System | Block Size | Matching | Notes |
|--------|-----------|----------|-------|
| vLLM | 16 tokens | Hash per block, parent-chained | Fine granularity, GPU-optimized |
| OpenAI | 256 tokens | Prefix blocks | Coarser, API-level caching breakpoints |
| Anthropic | Message boundary | Static cache at message breakpoints | Variable-size blocks aligned to message boundaries |
| Non-GPU (CPU/SSD) | 4096 tokens | Page-aligned | OS page size for CPU/SSD tiers |

## Design

### New: `KVBlock` dataclass

```python
@dataclass
class KVBlock:
    block_id: int
    session_id: str
    token_start: int      # first token index in this block
    token_count: int      # tokens in this block (≤ block_size, last block may be partial)
    size_bytes: int       # actual KV bytes for this block's tokens
    tier: Tier
    created_at_us: int
    last_accessed_at_us: int
    ref_count: int        # >1 when shared across sessions (Phase 2)
    parent_block_id: int | None  # for hash chain (Phase 2)
```

### New: `BlockChain` — replaces `CacheObject`

A `BlockChain` is a sequence of `KVBlock`s representing one session's cached KV:

```python
@dataclass
class BlockChain:
    session_id: str
    cache_key: str
    blocks: list[KVBlock]   # ordered list of blocks
    total_tokens: int
    total_size_bytes: int
```

### Config: `CacheConfig` (new section or extension)

```python
@dataclass
class CacheConfig:
    block_size_tokens: int = 16     # tokens per block (vLLM=16, OpenAI=256, page=4096)
    block_alignment: str = "fixed"  # "fixed" (uniform), "message" (Anthropic-style)
```

Add to `SimConfig` as `cache: CacheConfig`.

### TierStore changes

Currently stores `CacheObject` by key. Change to store `KVBlock` by block_id:
- `objects: dict[str, CacheObject]` → `blocks: dict[int, KVBlock]`
- `chains: dict[str, BlockChain]` → index from cache_key to block chain
- `can_fit(size)` → works on block granularity
- Eviction operates on individual blocks, not whole chains

### Engine changes

**`_place_kv_object` → `_place_kv_blocks`**:
- Compute number of blocks: `ceil(total_tokens / block_size_tokens)`
- Create `KVBlock` for each, with `size_bytes = kv_size_bytes(block_tokens, model)`
- Place blocks into tier (L1 → L2 → L3A fallthrough)
- Partial placement: first N blocks in L1, rest in L2 (if L1 fills mid-chain)

**Cache lookup**:
- Find longest matching block chain prefix (by token count, same as current)
- Return number of matched blocks × block_size = cached_tokens
- Unmatched blocks = uncached (need recompute)

**Eviction**:
- Evict individual blocks (tail of chain first, preserve prefix)
- Block-level LRU within each tier

### Backward compatibility

- `block_size_tokens=0` or absent → legacy mode (whole-context objects, current behavior)
- All existing tests pass unchanged in legacy mode
- New tests use explicit block sizes

## Tests

### Unit tests (`tests/test_blocks.py`)
1. `test_block_count_16_tokens` — 100 tokens at block_size=16 → 7 blocks (6 full + 1 partial)
2. `test_block_count_256_tokens` — 1000 tokens at block_size=256 → 4 blocks
3. `test_block_count_4096_tokens` — 50000 tokens at block_size=4096 → 13 blocks
4. `test_block_size_bytes` — each block's size_bytes = kv_size_bytes(block_tokens, model)
5. `test_partial_last_block` — last block has fewer tokens, correct size

### Integration tests
6. `test_block_eviction_preserves_prefix` — evict tail blocks, verify prefix blocks remain
7. `test_16_token_blocks_sim` — run sim with block_size=16, verify hit rate reasonable
8. `test_256_token_blocks_sim` — run sim with block_size=256, compare hit rate vs 16
9. `test_4096_token_blocks_sim` — run sim with block_size=4096, compare
10. `test_block_size_affects_fragmentation` — smaller blocks = less fragmentation

### Extreme condition tests
11. `test_single_token_block` — block_size=1, verify correct but slow
12. `test_block_larger_than_context` — block_size=1000000, behaves like legacy mode

## Implementation Sequence

1. Add `CacheConfig` to config.py
2. Create `KVBlock` and `BlockChain` in cache.py
3. Update `TierStore` to support block-level storage
4. Update engine `_place_kv_object` → `_place_kv_blocks`
5. Update cache lookup for block chains
6. Update eviction for block-level granularity
7. Add tests
8. Verify all 35 existing tests pass (legacy mode)
9. Create `docs/phase1_block_allocation_report.md`
10. Crosscheck, commit, push

## What This Does NOT Include (deferred to Phase 2)
- Cross-session block sharing (ref_count > 1)
- Hash-based matching (parent-chained SHA-256)
- Implicit deduplication
