# Report V2 — Plan: 20-Min Sim Throughout

Date: 2026-03-21

## Report structure
1. Global vs Local at 20 min (the headline result)
2. Deep dive: 1 worker (8 GPUs) at 20 min — occupancy, TTFT, queue, hit rates
3. Deep dive: 4 workers (32 GPUs) at 20 min — same metrics, global vs local
4. Session migration analysis
5. Summary

## Sims to run
- 1 worker, 20 min (single baseline)
- 4 workers, 20 min, global L3A
- 4 workers, 20 min, local L3A
- Sustaining QPS sweep at 20 min (1/2/4/8 workers, global vs local)
- TTL sweep at 20 min (4 workers, global vs local)

## Custom script
Write `scripts/heavy_coding_analysis.py` that runs all needed sims and generates plots in `plots/heavy_coding/`.
