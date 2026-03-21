# Global L3A Advantage — Plan #1: Heavy Coding Workload

Date: 2026-03-21

## Hypothesis

With 70-80% coding+agentic_coding workload, global L3A should outperform local L3A because:
- Coding KV objects are 8-16 GB each
- Per-worker local L3A (50GB) holds only 3-6 coding objects → overflows → cold misses
- Global L3A pools capacity → fewer cold evictions
- A cold miss costs ~17s (60k-token recompute) vs 50ms global L3A penalty
- Net: avoiding even a few cold misses saves more than the penalty on all hits

## Approach

1. Create heavy-coding profile_mix: coding=0.40, agentic_coding=0.35, chat=0.10, batch=0.10, agent=0.05
2. Run global vs local at stressed config with 1-8 workers
3. Verify traffic mix
4. Compare: hit rate, cold misses, TTFT, prefill_duration, cold_evictions_per_epoch
5. If global wins at some worker counts, identify the crossover point
6. Save findings, plots, commit

## Expected outcome

- At 1 worker: same (only 1 worker, no cross-worker benefit)
- At 2+ workers with heavy coding: global L3A should have higher hit rate because pooled capacity prevents cold evictions that local L3A can't avoid
- The advantage should grow with more workers (more fragmentation in local mode)
