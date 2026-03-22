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

**Worker topology**: 8 GPUs per worker (H100 DGX-class).

**Service**: 32 prefill slots/GPU, 256 decode slots shared, 128 prefill queue max/GPU.

## Single Worker Results (8 GPUs, realistic hardware)

```
Sharing factor:  1.865
Hit rates:       L1=98.2%  L2=1.7%  L3A=0.0%  miss=0.08%
Tier saturation: L1=91.8%  L2=154.1%  L3A=0.0%
L1->L2 pressure:   7.4/s
Cold evictions:    0
```

Key observations:
- **98.2% L1 hits** — 80GB L1 holds ~4-10 coding KV objects. Most lookups find cached KV in HBM.
- **L2 acts as overflow** — only 1.7% of lookups fall through to L2 (DRAM).
- **L3A unused** — with 80GB L1 + 1TB L2 per worker, objects rarely reach L3A in a 60s sim.
- **0.08% cold miss** — near-perfect cache performance.
- **L1 saturation 91.8%** — L1 is well-utilized but not overflowing excessively.

### Plot: Tier Occupancy Over Time
![Tier Occupancy](../plots/heavy_coding/tier_occupancy.png)

L1 fills to ~92% and stays there. L2 absorbs overflow. L3A remains empty.

### Plot: TTFT Distribution
![TTFT Distribution](../plots/heavy_coding/ttft_distribution.png)

L1 hits dominate with low TTFT. The small number of cold misses show a long tail (~17s for full 60k-token recompute).

### Plot: Savings Class Distribution
![Hit Rate Pie](../plots/heavy_coding/hit_rate_pie.png)

L1 hits dominate — HBM is fast enough to hold the working set for a 60s sim.

### Plot: Queue Depth
![Queue Depth](../plots/heavy_coding/queue_depth.png)

Prefill queue pressure from coding workloads' long compute times.

### Plot: Recompute Fraction
![Recompute Fraction](../plots/heavy_coding/recompute_fraction.png)

Most requests recompute only 5-20% of context (high prefix stability from coding sessions).

## Sensitivity Analysis

### L1 Capacity Sensitivity
![L1 Sensitivity](../plots/heavy_coding/l1_sensitivity.png)

L1 hit rate appears only above ~10 GB. Below that, coding KV objects (8-16 GB) don't fit. At 80 GB, L1 absorbs most of the working set.

### L2 Capacity Sensitivity
![L2 Sensitivity](../plots/heavy_coding/l2_sensitivity.png)

L2 hit rate vs capacity sweep. With realistic L1 (80GB), L2 is mainly used for overflow — less sensitive than in stressed configs.

## Multi-Node Scaling: Global vs Local L3A

### Node Scaling (1-8 workers × 8 GPUs)
![Node Scaling](../plots/heavy_coding/node_scaling.png)

| Workers | GPUs | Global L3A Total | Local L3A/Worker | Global Hit | Local Hit |
|---------|------|-----------------|-----------------|-----------|----------|
| 1 | 8 | 8 TB | 8 TB | 99.9% | 99.9% |
| 2 | 16 | 16 TB | 8 TB | 99.9% | 99.9% |
| 4 | 32 | 32 TB | 8 TB | 99.9% | 99.9% |
| 8 | 64 | 64 TB | 8 TB | 99.9% | 99.9% |

**At realistic hardware scale, global and local L3A are identical.** With 8TB SSD per worker, even local L3A has more than enough capacity for the workload in a 60s sim. The global L3A advantage only appears when per-worker SSD is small relative to total KV working set — see the stressed config analysis below.

**L3A Latency Sensitivity** (panel f): TTFT is completely flat across 0-100ms remote latency because L3A is unused — all hits come from L1/L2.

### TTL Sensitivity (2 workers × 8 GPUs)
![TTL Sensitivity](../plots/heavy_coding/ttl_sensitivity.png)

Hit rate is 99.9% regardless of TTL. Shorter L2 TTLs increase queue wait (objects move to L3A sooner → slower restore), but the effect is small.

### Sustaining QPS at SLA (p95 queue wait < 500ms)
![Sustaining QPS](../plots/heavy_coding/sustaining_qps.png)

| Workers | GPUs | Global QPS | Local QPS |
|---------|------|-----------|-----------|
| 1 | 8 | 44 | 44 |
| 2 | 16 | 79 | 79 |
| 4 | 32 | 171 | 171 |
| 8 | 64 | 332 | 332 |

Scales linearly. Global and local identical — cache is never the bottleneck at realistic hardware scale. **QPS is limited by prefill compute (~14-17s for large coding contexts), not by cache capacity.**

## When Does Global L3A Matter?

The global L3A advantage appears under **constrained SSD capacity** — when per-worker SSD is too small to hold the active coding sessions' KV objects. In prior stressed-config analysis (50GB L3A/worker):

| Workers | Global Hit | Local Hit | Gap |
|---------|-----------|----------|-----|
| 1 | 94.6% | 94.6% | 0% |
| 2 | 99.1% | 99.0% | +0.1% |
| 4 | 99.9% | 98.7% | +1.2% |

The gap grows with more workers because local L3A fragments capacity. At extreme scales or with smaller SSDs, the gap would be larger (see `plan_and_progress/global_l3a_advantage_progress_1.md` for 40+ percentage point gaps with 12.5GB/worker).

## Key Findings

1. **Realistic hardware absorbs coding workloads well.** 80GB L1 + 1TB L2 + 8TB L3A per worker provides 99.9% hit rate. Cache capacity is not the bottleneck.

2. **Prefill compute is the bottleneck.** 60k-token coding contexts require ~14-17s of GPU compute. This limits sustaining QPS to ~44/worker regardless of cache configuration.

3. **Global vs local L3A difference is negligible at realistic scale.** 8TB/worker SSD never overflows in normal operation. The advantage only appears with constrained SSD or extreme concurrent session counts.

4. **L1 is the critical tier.** 98.2% of lookups hit L1 (HBM). L1 capacity directly determines cache effectiveness — below 10GB, coding KV objects can't fit and performance degrades sharply.

5. **High prefix stability** (95% initial, 80-85% final) means most tokens are served from cache. The 20-30k shared system prefix is reused across all turns in a session.

## Reproduction

```bash
# Generate all plots with heavy coding config
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
