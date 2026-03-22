---
name: Experiment design — controlled load, seeds, metrics
description: Design experiments that produce reliable, interpretable results
type: feedback
---

Experiment design principles:
- **Controlled load**: Set arrival rate below capacity to avoid drops. If >50% dropped, queue metrics have survivorship bias.
- **One variable**: Hold everything constant except the variable under test. Same seed.
- **Multiple durations**: Short sims show transient behavior. Steady state may require 10-20 min.
- **Seed sensitivity**: At capacity edges, run 3+ seeds. If results vary wildly, the system is at an unstable operating point.
- **Always report**: hit rate AND completed requests AND drop rate AND slot utilization. Hit rate alone is misleading if most requests are dropped.

**Why:** At peak=5, results varied from 21% to 85% drops across seeds — the system was at the capacity edge. Without multi-seed testing, any single result looks definitive but is actually random.

**How to apply:** Start with low load (no drops), increase to find the cliff, then characterize the cliff with multiple seeds. Report the operating regime (underloaded/balanced/overloaded) alongside every metric.
