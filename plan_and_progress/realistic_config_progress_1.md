# Realistic Config — Progress #1

Date: 2026-03-21

## Changes

### L3A capacity semantics
- Config `tiers[2].capacity_bytes` now means **per-worker SSD capacity**
- Global L3A: `capacity × n_workers` (pool all SSDs)
- Local L3A: `capacity` per worker (own SSD only)
- Previously: config = total, local = total/N. Now: config = per-worker, global = per-worker × N.

### Default config updated to realistic hardware
- L1: 80 GB per GPU (unchanged)
- L2: 1 TB per worker (was 4 TB — realistic host DRAM for KV cache)
- L3A: 8 TB per worker (was 20 TB — realistic NVMe per server)

### Single-worker L3A penalty fix
- Global L3A remote penalty only applies when `n_workers > 1` (was `n_nodes > 1`)
- At 1 worker, global and local are identical — no spurious penalty

### Results
| Workers | GPUs | Global L3A Total | Local L3A/Worker | Global Hit | Local Hit |
|---------|------|-----------------|-----------------|-----------|----------|
| 1 | 8 | 8 TB | 8 TB | 94.6% | 94.6% |
| 2 | 16 | 16 TB | 8 TB | 100% | 97.3% |
| 4 | 32 | 32 TB | 8 TB | 100% | 97.3% |
| 8 | 64 | 64 TB | 8 TB | 100% | 97.3% |

At 1 worker: identical (as expected). At 2+: global hits 100% because pooled capacity is massive.

### Crosscheck
All plan items delivered. 24/24 tests pass. Plots regenerated.
