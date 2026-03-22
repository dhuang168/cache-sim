# Experiment A: Block Size Comparison

**Config**: heavy_coding, 1 worker (8 GPUs), 5 min, peak=5, ~3500 events

## Results

### Block size × sharing × eviction policy

| Block Size | Sharing | Hit Rate | Recompute Frac | L1 Occ | Memory Saved |
|-----------|---------|----------|---------------|--------|-------------|
| Legacy (0) | Off | 99.8% | 0.264 | 37% | — |
| Legacy (0) | On | 99.8% | 0.264 | 40% | 16,122 GB |
| 16 tokens | Off | 99.8% | 0.264 | 37% | — |
| 16 tokens | On | 99.8% | 0.264 | 41% | 16,127 GB |
| 256 tokens | Off | 99.8% | 0.268 | 38% | — |
| 256 tokens | On | 99.8% | 0.268 | 38% | 16,119 GB |
| **4096 tokens** | Off | **85.6%** | **0.388** | 38% | — |
| **4096 tokens** | On | **85.6%** | **0.388** | 41% | 15,953 GB |

TTL and LRU eviction produced **identical results** at this load level (5 min, low pressure).

## Findings

1. **Block size 0/16/256**: Identical hit rate (99.8%) and recompute fraction (0.264-0.268). Confirms Phase 1 conclusion: fine vs moderate granularity is irrelevant for coding workloads.

2. **Block size 4096**: Hit rate drops to 85.6%, recompute fraction jumps to 0.388. The 4K-token block boundary loses significant cached tokens for contexts that don't align. **4K blocks are too coarse for coding workloads.**

3. **Sharing**: Memory saved is ~16 TB across all block sizes. This is cumulative over 3,499 events — each shared prefix hit saves ~4.6 GB (the framework prefix). Sharing doesn't affect hit rate (expected — it's a memory optimization, not a hit rate optimization at single worker).

4. **TTL vs LRU**: Identical at this load. The 5-min sim with low arrival rate doesn't create enough pressure to differentiate the eviction policies.

5. **L1 occupancy slightly higher with sharing** (40-41% vs 37-38%). Shared prefix objects are accessed frequently, keeping them in L1 longer.
