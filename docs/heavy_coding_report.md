# Heavy Coding Workload Report

**Config**: `configs/heavy_coding.json`
**Date**: 2026-03-21
**Model**: Llama3-70B FP16 on A100-80GB

## Workload Profile

90% of traffic is coding workloads (coding + agentic_coding), reflecting a deployment heavily weighted toward AI coding assistants like Claude Code and Cursor.

| Profile | Mix Weight | System Prefix | Input/Turn | Context Growth | Session Duration |
|---------|-----------|---------------|-----------|----------------|-----------------|
| coding | 45% | 20,000 tokens | 8,000 tokens | 2-10k/turn | 1 hour |
| agentic_coding | 45% | 30,000 tokens | 15,000 tokens | 5-20k/turn | 30 min |
| chat | 5% | 2,048 tokens | 150 tokens | 100-500/turn | 10 min |
| batch | 3% | 2,048 tokens | 1,200 tokens | none | 1s |
| agent | 2% | 2,048 tokens | 400 tokens | 300-1.5k/turn | 30 min |

**KV object sizes** (70B FP16 at 327,680 bytes/token):
- Coding session (26k tokens avg): ~8.2 GB
- Agentic coding session (50k tokens avg): ~15.6 GB
- Chat session (2.6k tokens avg): ~0.8 GB

## Hardware Configuration

| Tier | Capacity | Bandwidth | Block Size | Scope |
|------|----------|-----------|-----------|-------|
| L1 (HBM) | 80 GB | 3 TB/s | 5 KB | Per-GPU |
| L2 (DRAM) | 1 TB | 64 GB/s | 32 MB | Per-worker (shared by 8 GPUs) |
| L3A (SSD) | 8 TB | 7 GB/s | 256 MB | Per-worker SSD; global mode pools all workers |

**Worker topology**: 8 GPUs per worker. Workers = 1, 2, 4, or 8 (8-64 GPUs total).

**Service**: 32 prefill slots/GPU, 256 decode slots shared, 128 prefill queue max/GPU.

## Stressed Scenario Results (single worker, 8 GPUs)

Small L1 (500MB) and L2 (10GB) per worker force multi-tier activity. L3A = 50GB/worker.

```
Sharing factor:  1.882
Hit rates:       L1=0.0%  L2=0.02%  L3A=98.6%  miss=1.4%
Tier saturation: L1=0.0%  L2=92.5%  L3A=85.1%
Cold evictions:  329
```

Key observations:
- L1 is too small (500MB) for any coding KV object (8-16 GB) → 0% L1 hits
- L2 (10GB) holds 1-2 coding objects but quickly fills → most objects go to L3A
- L3A absorbs 98.6% of lookups
- 1.4% cold miss rate — L3A pressure causes some evictions

### Plot: Tier Occupancy Over Time
![Tier Occupancy](../plots/heavy_coding/tier_occupancy.png)

L2 saturates at ~93% immediately. L3A rises to 85% as coding sessions accumulate.

### Plot: TTFT Distribution
![TTFT Distribution](../plots/heavy_coding/ttft_distribution.png)

L3A hits dominate. Cold misses show a long tail from recomputing large coding contexts.

### Plot: Savings Class Distribution
![Hit Rate Pie](../plots/heavy_coding/hit_rate_pie.png)

L3A WORTHWHILE is the dominant class — SSD restore is faster than recomputing 60k tokens.

### Plot: Queue Depth
![Queue Depth](../plots/heavy_coding/queue_depth.png)

Prefill queue saturated at max (128) — coding workloads are compute-intensive.

### Plot: Recompute Fraction
![Recompute Fraction](../plots/heavy_coding/recompute_fraction.png)

Most requests recompute only 5-20% of context (high prefix stability from coding sessions).

## Sensitivity Analysis

### L1 Capacity Sensitivity
![L1 Sensitivity](../plots/heavy_coding/l1_sensitivity.png)

L1 hit rate only appears at 40-80 GB — below that, coding KV objects (8-16 GB) don't fit.

### L2 Capacity Sensitivity
![L2 Sensitivity](../plots/heavy_coding/l2_sensitivity.png)

L2 hit rate improves with capacity, but even at 50 GB, L3A remains the primary cache tier for coding workloads.

## Multi-Node Scaling: Global vs Local L3A

### Node Scaling (1-8 workers × 8 GPUs)
![Node Scaling](../plots/heavy_coding/node_scaling.png)

| Workers | GPUs | Global L3A | Global Hit | Local L3A/worker | Local Hit |
|---------|------|-----------|-----------|-----------------|----------|
| 1 | 8 | 50 GB | 94.6% | 50 GB | 94.6% |
| 2 | 16 | 100 GB | 99.1% | 50 GB | 99.0% |
| 4 | 32 | 200 GB | 99.9% | 50 GB | 98.7% |
| 8 | 64 | 400 GB | 99.9% | 50 GB | 98.7% |

At this stressed L3A size (50GB/worker), global L3A has a small advantage at 4+ workers (99.9% vs 98.7%). The gap is modest because 50GB/worker is large enough to hold several coding objects.

**L3A Latency Sensitivity** (panel f, 16 GPUs): TTFT is stable across 0-100ms remote latency, indicating the 50ms penalty is negligible compared to prefill compute time (~17s for coding contexts).

### TTL Sensitivity (2 workers × 8 GPUs)
![TTL Sensitivity](../plots/heavy_coding/ttl_sensitivity.png)

Both global and local maintain >97% hit rate across all L2 TTL values. TTL has minimal impact — capacity is the bottleneck, not object lifetime.

### Sustaining QPS at SLA (p95 queue wait < 500ms)
![Sustaining QPS](../plots/heavy_coding/sustaining_qps.png)

| Workers | GPUs | Global QPS | Local QPS |
|---------|------|-----------|-----------|
| 1 | 8 | 33 | 33 |
| 2 | 16 | 67 | 67 |
| 4 | 32 | 136 | 140 |
| 8 | 64 | 278 | 278 |

Sustaining QPS scales roughly linearly with workers. Global and local L3A perform similarly at this scale — the 50ms L3A penalty is small relative to the ~17s prefill compute for coding workloads.

## Key Findings

1. **Coding workloads produce massive KV objects** (8-16 GB), making L1 (80GB) a thin cache that holds only 4-10 objects. L3A becomes the primary cache tier.

2. **98.6% of lookups hit L3A** in the stressed config — the system is effectively an SSD-backed cache with HBM as a small buffer.

3. **Cache prefix stability is high** (95% initial, 80-85% final for coding profiles). Most of each request is a cache hit even on the first turn due to the 20-30k shared system prefix.

4. **Global L3A advantage is modest** at the stressed config's 50GB/worker — both modes achieve >98% hit rate. The advantage would be more pronounced with smaller per-worker SSD or higher concurrent session counts.

5. **Sustaining QPS is 33/worker** for heavy coding workloads — limited by the ~17s prefill compute time for large coding contexts, not by cache performance.

6. **50ms global L3A latency is negligible** relative to the 15-17s coding prefill compute. The latency sensitivity plot shows <100ms TTFT variation across 0-100ms remote latency.

## Reproduction

```bash
# Generate all plots
python scripts/sanity_plots.py --config configs/heavy_coding.json --outdir plots/heavy_coding

# Run programmatically
from sim.config import SimConfig
from sim.engine import SimEngine
config = SimConfig.from_json("configs/heavy_coding.json")
config.sim_duration_s = 60.0
config.warmup_s = 5.0
config.sim_start_time_s = 36000.0
metrics = SimEngine(config).run()
print(metrics.report())
```
