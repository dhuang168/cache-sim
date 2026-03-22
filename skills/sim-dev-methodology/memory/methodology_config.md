---
name: Config consistency — dependency map
description: When changing any parameter, verify all dependent parameters still produce valid results
type: feedback
---

Changing parameter A can silently invalidate parameter B. Walk the dependency map before running:
- Workload sizes → oracle/model range (no clamping)
- Arrival rates → system not severely overloaded (>50% drops = meaningless metrics)
- Profile mix → traffic proportions match config
- Topology (workers/GPUs) → capacity semantics (per-worker × N)
- Sim duration → long enough for steady state
- Diurnal parameters → sim_start_time in active period

**Why:** Oracle clamped at 32K tokens when workloads grew to 60K — cold misses appeared cheaper than cache hits, completely inverting conclusions. No error, just silently wrong results.

**How to apply:** Before any sim after config change: list what changed → check what depends on it → run sanity check (metrics in plausible ranges? values clamped? system in meaningful operating regime?).
