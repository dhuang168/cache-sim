# Realistic Coding Workload — Progress #1

Date: 2026-03-21

## Changes

### New/updated workload profiles
- **coding** profile updated: 20k system prefix (was 2k), 8k input/turn (was 800), 95→80% stability (was 85→40%)
- **agentic_coding** profile added: 30k system prefix, 15k input/turn, 5-20k context growth, 95→85% stability
- **profile_mix** updated: chat=0.30, coding=0.20, batch=0.20, agent=0.15, agentic_coding=0.15
- Legacy v1 profiles preserved in `configs/legacy_v1.json`
- Config bumped to `run_id: default-v2`

### KV size impact
- 60k-token KV = 18.3 GB (70B FP16)
- L1 (80GB) fits only ~4 such objects → real eviction pressure for coding workloads
- L2 (4TB) fits ~223 objects → adequate for warm cache

### Test results
- 22/22 tests pass with v2 config
- No test changes needed — all backward compatible

### Plot observations (v2 workload)
- Sustaining QPS at 8 nodes: 96 (was 122 with v1) — heavier requests
- TTFT phase transition at 8 nodes still present (~345ms vs ~700ms at 4 nodes)
- Hit rate remains 100% in 10s stressed sim — L2+L3A absorb everything at this scale
- Global vs local L3A gap still not visible at 50GB L3A — would need longer sims or smaller L3A

### Crosscheck: all plan items delivered
No gaps.
