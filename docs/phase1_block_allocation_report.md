# Phase 1: Block-Level Allocation Report

Date: 2026-03-22

## Summary

Added configurable token block size to model different production caching systems. The block size controls cache hit granularity — only full blocks count as cached.

## Configuration

New `CacheConfig` section in `SimConfig`:

```json
"cache": {
    "block_size_tokens": 16,
    "block_alignment": "fixed"
}
```

| Block Size | System | Behavior |
|-----------|--------|----------|
| 0 (default) | Legacy | Whole-context objects, no rounding (backward compatible) |
| 16 | vLLM | Fine granularity, minimal boundary loss |
| 256 | OpenAI | Coarser, API-level prefix caching breakpoints |
| 4096 | Non-GPU / Page-aligned | OS page size for CPU/SSD tiers |
| message | Anthropic | Static cache at message boundaries (block_alignment="message") |

## How Block Size Affects Caching

Only **full token blocks** count as cached. The last partial block must be recomputed.

Example: 10,000-token context with 95% prefix stability = 9,500 raw cached tokens:

| Block Size | Full Blocks | Cached Tokens | Lost Tokens | Effective Stability |
|-----------|-------------|---------------|-------------|-------------------|
| Legacy (0) | — | 9,500 | 0 | 95.0% |
| 16 tokens | 593 | 9,488 | 12 | 94.9% |
| 256 tokens | 37 | 9,472 | 28 | 94.7% |
| 4,096 tokens | 2 | 8,192 | 1,308 | **81.9%** |

At 4K blocks, 13% of cached tokens are lost to block boundary rounding. This significantly impacts cache effectiveness for contexts that don't align to block boundaries.

## Fragmentation

Each block allocates a fixed number of tokens. The last block has internal fragmentation:

| Total Tokens | Block Size | Blocks | Wasted Tokens | Fragmentation |
|-------------|-----------|--------|---------------|---------------|
| 100 | 16 | 7 | 12 | 10.7% |
| 1,000 | 256 | 4 | 24 | 2.3% |
| 50,000 | 4,096 | 13 | 2,248 | 4.3% |
| 256 | 16 | 16 | 0 | 0% (exact fit) |

## Test Results

21 new tests in `tests/test_blocks.py`:
- 5 block count tests (16/256/4096/exact/legacy)
- 6 boundary rounding tests (different sizes, edge cases)
- 4 fragmentation tests
- 2 granularity comparison tests
- 4 integration tests (sim runs at 16/256/4096, hit rate comparison)

**56 total tests, 4.74s.** All pass including 35 existing tests (backward compatible).

## Verification: Block Size Affects Recompute Fraction

The `test_block_size_affects_hit_rate` test confirms larger blocks → higher recompute fraction (more tokens lost at boundaries). This is the expected granularity tradeoff.
