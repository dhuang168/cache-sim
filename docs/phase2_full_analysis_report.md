# Full Analysis: Multi-Node Cache with Block Sharing

**Config**: `configs/heavy_coding.json` | **Model**: Llama3-70B FP16 | **Features**: LRU eviction, three-tier sharing, per-SSD contention
**Controlled load**: `peak=3` (~75 req/s arrival) | **Duration**: 20 min | **Workers**: 4 × 8 GPUs unless noted

## 1. Single Worker Baseline (8 GPUs, 20 min)

```
Hit rate:       99.8%
Completed:      12,553 / 24,875 (50% dropped — single worker saturated)
TTFT:           mean=62.6s  p95=157.2s
Queue wait:     mean=45.6s  p95=121.7s
Prefill:        mean=18.9s  p95=55.9s
Slot utilization: 79%
L1=44%  L2=99%  L3A=92%
Sharing saved:  50.7 TB
```

L2 is saturated (99%). L3A at 92% and rising. With only 8 GPUs (256 prefill slots), the system can't keep up with the coding workload — 50% of requests dropped.

## 2. Node Scaling: Global vs Local L3A

| Workers | GPUs | Global Hit | Global Comp | Global QW | Local Hit | Local Comp | Local QW |
|---------|------|-----------|-------------|-----------|----------|------------|----------|
| 1 | 8 | 99.8% | 12,553 | 45.6s | 99.8% | 12,553 | 45.6s |
| 2 | 16 | 99.8% | 3,129 | 157s | 89.3% | 13,457 | 50.3s |
| **4** | **32** | **99.8%** | **23,781** | **2.8s** | 78.0% | 14,892 | 48.6s |

### Key observations

**At 1 worker**: Identical (same physical SSD, no cross-worker access).

**At 2 workers**: Global L3A has high contention — only 2 SSDs to share, `per_ssd_readers = concurrent/2`. This creates 23K contention events, inflating transfer times → 87% drops, only 3.1K completed. Local wins at this scale.

**At 4 workers**: **Global L3A wins decisively.** Contention drops to 493 events (4 SSDs distribute the load). 99.8% hit rate + low contention = 4% drops, 23.8K completed. Local's 78% hit rate causes 40% drops.

**The crossover is at 3-4 workers**: below that, per-SSD contention on global L3A hurts more than the cold misses on local. Above that, the contention is distributed across enough SSDs to be manageable.

## 3. Dispatch Algorithms (4 workers, Global L3A)

| Algorithm | Completed | Queue Wait | TTFT Mean | Contention | Affinity |
|-----------|-----------|-----------|-----------|------------|---------|
| push | 23,781 | 2.8s | 25.6s | 493 | 4% |
| push_smart | 23,784 | 2.9s | 25.7s | 577 | 4% |
| **pull** | **23,893** | **2.6s** | **25.3s** | **295** | 3% |

Pull edges out push by ~0.5% in completed requests and 8% in queue wait. Smart push doesn't significantly improve over basic push — at 99.8% global L3A hit rate, the dispatch strategy has minimal impact on cache performance. The bottleneck is prefill compute time (18.9s mean), not dispatch quality.

## 4. Eviction Policy: TTL vs LRU

| Policy | Completed | Queue Wait | TTFT | L1 Occ | L2 Occ | L1→L2 Events |
|--------|-----------|-----------|------|--------|--------|--------------|
| TTL | 23,807 | 2.8s | 25.6s | 24% | 58% | 26,006 |
| LRU | 23,781 | 2.8s | 25.6s | 24% | 58% | 25,600 |

**No practical difference** at this load. Both policies produce identical hit rates, throughput, and tier occupancy. The TTL mode's access-based refresh means hot objects stay in fast tiers anyway, converging with LRU behavior.

## 5. Sharing: On vs Off

| Sharing | Completed | Queue Wait | TTFT | Memory Saved |
|---------|-----------|-----------|------|-------------|
| Off | 23,810 | 2.7s | 25.5s | — |
| On | 23,781 | 2.8s | 25.6s | **127 TB** |

Sharing doesn't affect hit rate or throughput — it's a **memory efficiency** optimization. The 127 TB saved represents the cumulative deduplication of framework (20K tokens) and workspace (5K tokens) prefixes across sessions. At 327 KB/token for 70B FP16, this is significant for capacity planning.

## 6. Load Sensitivity (4 workers, Global L3A)

| Peak | Arrival | Throughput | Dropped | Slot Util | QW Mean | TTFT Mean |
|------|---------|-----------|---------|-----------|---------|-----------|
| 1 | 7.0/s | 7.0/s | **0%** | 15% | 19ms | 21.3s |
| 2 | 14.0/s | 14.0/s | **0%** | 32% | 24ms | 22.5s |
| 3 | 20.9/s | 20.0/s | 4% | 48% | 2.8s | 25.6s |
| 4 | 27.6/s | 24.4/s | 12% | 55% | 10.3s | 30.7s |
| 5* | 34.8/s | 5.4/s | 85% | 84% | 142s | 206s |

*Peak=5 is seed-dependent: 85% drops at seed=42, 46% at seed=123, 21% at seed=456. At this arrival rate, the system is at the edge of capacity — whether it collapses depends on the timing of agentic_coding bursts.

**Max theoretical throughput** = 1024 slots / 18.9s mean prefill = **54 req/s**. The system operates linearly up to ~20 req/s (peak=3, 48% utilization), starts degrading at ~28 req/s (peak=4, 55% utilization, 12% drops), and becomes unstable at ~35 req/s (peak=5).

**Why 48% utilization at peak=3**: Arrival rate (20.9/s) is 39% of max throughput (54/s). The observed 48% is higher because slot utilization includes queue wait time, not just prefill compute. The 4% drops are from bursty arrivals that momentarily exceed per-node capacity.

## 7. Sharing Impact on Eviction Pressure

| Metric | No Sharing | Sharing | Impact |
|--------|----------|---------|--------|
| L1→L2 evictions | 17,978 | **25,600** | +42% more |
| Cold evictions | 10,943 | **14,370** | +31% more |
| L3A contention | 516 | 493 | -4% less |
| Memory saved | — | **127 TB** | Cumulative deduplication |

**Surprise**: Sharing increases eviction pressure. Shared prefix objects (framework 20K, workspace 5K tokens) stay pinned in L1 (high ref_count, constantly accessed by new sessions). They consume L1 capacity → less room for session-unique KV → more session objects evicted to L2/L3A.

This is the **sharing memory paradox**: sharing saves total memory (127 TB fewer bytes stored) but the pinned shared objects create more churn for non-shared objects. The net effect on throughput is neutral (identical hit rates and completion counts) because the evicted session objects land in L2/L3A and are still accessible.

## Summary

| Finding | Evidence |
|---------|----------|
| **Global L3A wins at 4+ workers** | 99.8% hit, 4% drops, 60% more throughput than local |
| **2 workers: local wins** (contention too high on 2 SSDs) | Global: 87% drops, local: 46% drops |
| **Pull dispatch preferred** | +0.5% throughput, -8% queue wait, -40% contention |
| **TTL ≈ LRU** at controlled load | Identical throughput and TTFT |
| **Sharing saves 127 TB** | Memory optimization, no hit rate impact |
| **Capacity cliff at peak=4-5** | 12% drops at peak=4, seed-dependent collapse at peak=5 |
| **Sharing paradox** | Saves 127TB but increases L1 evictions by 42% (pinned shared objects consume L1) |
| **Prefill compute is the bottleneck** | Mean 18.9s, slot utilization drives drops |

## Configuration Reference

```json
{
  "service": {"n_prefill_nodes": 32, "n_gpus_per_worker": 8, "dispatch_algorithm": "pull"},
  "cache": {
    "eviction_policy": "lru",
    "sharing": {"enabled": true, "tiers": [
      {"name": "framework", "tokens": 20000, "sharing_group_size": 1000},
      {"name": "workspace", "tokens": 5000, "sharing_group_size": 10}
    ]}
  }
}
```
