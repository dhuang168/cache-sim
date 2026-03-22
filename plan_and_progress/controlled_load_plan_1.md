# Controlled Load Tests — Plan

Date: 2026-03-22

## Motivation
The 20-min analysis had 693/s arrival vs 78/s throughput — 88.7% dropped. Queue metrics were meaningless noise. Need controlled load tests to see the real impact.

## Test (a): Reduced arrival rate (75/s)
- Set arrival rate to ~75/s (near throughput capacity of 78/s)
- 4 workers × 8 GPUs, 20 min
- Global vs local L3A
- **Expected**: Global has shorter queue, better TTFT, fewer drops

## Test (b): Fixed request count (90K requests, 25 min limit)
- Generate exactly ~90K requests (adjust arrival rate × duration)
- Compare: how many complete in 25 min? What's the completion time?
- **Expected**: Global completes faster (cache hits = shorter prefill = slots free sooner)

## Implementation
- Script: `scripts/controlled_load_test.py`
- For (a): scale arrival_rate_peak down so effective rate ≈ 75/s
- For (b): can't easily fix request count in DES, but can set arrival rate × duration to target ~90K and compare completion counts
