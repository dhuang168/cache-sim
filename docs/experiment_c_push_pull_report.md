# Experiment C: Push vs Pull Dispatch

**Config**: heavy_coding, 4 workers (32 GPUs), 5 min, peak=3, global L3A, LRU, sharing enabled

## Results

| Metric | Push | Pull | Pull Advantage |
|--------|------|------|---------------|
| Hit rate | 100% | 100% | Same |
| Completed requests | 1,902 | **2,118** | **+11.4% more** |
| Dropped | **10%** | **0%** | Zero drops |
| Queue wait mean | 1,533ms | **0ms** | **No queueing** |
| Queue wait p95 | 1,579ms | **0ms** | — |
| TTFT mean | 12,240ms | **10,573ms** | **14% faster** |
| TTFT p95 | 75,169ms | **52,080ms** | **31% faster** |
| Affinity | 20% | **46%** | **2.3× more affinity** |
| Contention events | 952 | 1,002 | Similar |
| Load balance (std) | 65 | 60 | Slightly better |

## Findings

1. **Pull dispatch is superior across all metrics.** Zero drops (vs 10% push), 14% faster mean TTFT, 31% faster p95 TTFT, 11% more completed requests.

2. **Pull achieves 2.3× higher affinity** (46% vs 20%). Because idle nodes pull from the global queue and prefer jobs whose session KV is in their local cache, they naturally achieve better affinity than push's "send to least loaded" approach.

3. **Pull has zero queueing.** Nodes only pull when they have a free slot → requests never wait in a per-node queue. Push commits requests to specific nodes that may have full queues → 1.5s mean queue wait.

4. **Why pull is better**: Push assigns requests before knowing if the target node has capacity. If the chosen node is busy, the request queues (or drops if queue full). Pull inverts this — nodes advertise availability by pulling, so requests only go to nodes that are ready.

5. **Contention is similar** (952 vs 1002). Both modes access global L3A at similar rates — the difference is in dispatch efficiency, not cache access patterns.

6. **Load balance slightly better with pull** (std 60 vs 65). Nodes self-select based on availability → natural load balancing.

## 20-Minute Results (with per-SSD contention fix)

At 20 min with global L3A and corrected contention model, the algorithms converge:

| Algorithm | Completed | QW Mean | TTFT Mean | Contention |
|-----------|-----------|---------|-----------|------------|
| push | 23,781 | 2.8s | 25.6s | 493 |
| push_smart | 23,784 | 2.9s | 25.7s | 577 |
| **pull** | **23,893** | **2.6s** | **25.3s** | **295** |

Pull still edges out push (+0.5% completed, -10% contention, -8% queue wait), but the gap is much smaller at 20 min than at 5 min. Smart push is comparable to basic push — L3A-aware routing doesn't help much when global L3A hit rate is already 99.8%.

## Recommendation

**Pull dispatch is preferred** — it consistently outperforms push at all durations with no downside. The advantage is largest at 5 min (14% faster TTFT, zero drops) and narrows at 20 min (2% better) as global L3A absorbs most of the cache benefit regardless of dispatch strategy.

Smart push doesn't significantly improve over basic push because the bottleneck is compute time (14-17s per coding request), not dispatch quality.
