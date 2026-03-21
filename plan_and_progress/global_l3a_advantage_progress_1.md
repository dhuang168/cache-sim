# Global L3A Advantage — Progress #1

Date: 2026-03-21

## Bug Fixed

Local L3A was getting full `l3a_cfg.capacity_bytes` per worker instead of `capacity / n_workers`. This gave local mode N× more total capacity than global, masking the capacity pooling advantage.

## Results: Global L3A wins at 2+ workers with heavy coding workload

### Heavy coding mix (coding=40%, agentic_coding=35%)

| Workers | Global Hit Rate | Local Hit Rate | Winner | Gap |
|---------|----------------|----------------|--------|-----|
| 1 (same) | 92.3% | 93.8% | Local +1.5% | Latency penalty with no pooling benefit |
| 2 | 91.9% | 76.6% | **Global +15.3%** | Local 25GB/worker too small |
| 4 | 92.0% | 51.8% | **Global +40.2%** | Local 12.5GB/worker → catastrophic misses |

### Default mix (coding=20%, agentic_coding=15%)

| Workers | Global Hit Rate | Local Hit Rate |
|---------|----------------|----------------|
| 2 | 94.0% | 76.1% |
| 4 | 94.0% | 48.2% |

### Sustaining QPS
| Workers | Global QPS | Local QPS |
|---------|-----------|-----------|
| 2 | 155 | 136 |
| 4 | 309 | 178 |
| 8 | 485 | 485 |

### Why global wins
- Coding KV objects are 8-16 GB each
- At 4 workers, local L3A = 12.5GB per worker — can't hold even 1 coding object
- Global pools 50GB across all workers — holds 3-6 coding objects
- Cold miss costs ~17s (60k recompute) vs 50ms global penalty
- Net: avoiding cold misses far outweighs the per-hit penalty

### Validated with metrics
- Local 4W: slot_util=74%, prefill_p95=17,000ms (full recompute)
- Global 4W: slot_util=46%, prefill_p95=12,873ms (cache hits)
- The slot utilization difference confirms cold misses consume more GPU time

## Crosscheck: all plan items delivered
No gaps. 24/24 tests pass.
