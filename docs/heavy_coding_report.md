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

**Worker topology**: 8 GPUs per worker (H100 DGX-class). Service: 32 prefill slots/GPU, 256 decode slots shared.

**L3A capacity semantics**: Config value = per-worker SSD. Global mode pools all workers' SSDs (e.g., 4 workers × 8 TB = 32 TB shared). Local mode: each worker uses its own 8 TB SSD.

---

## Key Finding: Global L3A Is Essential for Multi-Worker Deployments

At production-relevant durations (5-20 min), **global L3A dramatically outperforms local** due to session migration between workers:

| Duration | Global L3A Hit Rate | Local L3A Hit Rate | Gap |
|----------|--------------------|--------------------|-----|
| 1 min | 99.91% | 99.91% | 0% |
| 5 min | 99.81% | **55.23%** | **+44.6%** |
| 10 min | 99.73% | **36.74%** | **+63.0%** |
| 20 min | 99.77% | **26.56%** | **+73.2%** |

*(4 workers × 8 GPUs, realistic hardware, heavy coding workload)*

### Why the Gap Appears

1. **L1/L2 saturate within minutes.** L1 (80 GB/GPU) fills in ~1 min, L2 (1 TB/worker) fills by 5 min. Objects overflow to L3A.
2. **Affinity is lost once objects reach L3A.** The push dispatcher's affinity check only looks at L1/L2. Once objects are evicted to L3A, subsequent requests lose affinity and get dispatched to any available worker.
3. **97% of dispatches are non-affinity** at steady state — sessions constantly migrate between workers.
4. **Local L3A can't serve migrated sessions.** Worker 1's SSD doesn't have KV objects created on Worker 0. Each cold miss costs 14-17s (full 60k-token recompute).
5. **Global L3A finds everything.** The pooled SSD storage is accessible from any worker → 99.8% hit rate maintained.

### Why 1-Min Sims Show No Difference

At 1 min, L1 (80 GB/GPU) and L2 (1 TB/worker) haven't filled yet. Objects stay in L1/L2, affinity checks find them, and sessions don't migrate. **The 60s diagnostic plots below reflect this early-stage behavior and should not be used to compare global vs local L3A.**

---

## Single Worker Snapshot (60s sim, 8 GPUs)

The following plots use a 60s simulation — sufficient for single-worker diagnostics but too short for multi-worker global vs local comparison.

```
Sharing factor:  1.866
Hit rates:       L1=98.2%  L2=0.83%  L3A=0.88%  miss=0.08%
Tier saturation: L1=91.8%  L2=75.5%  L3A=9.9%
```

- **98.2% L1 hits** — 80 GB L1 holds ~4-10 coding KV objects. At 60s, the working set fits in HBM.
- **L2/L3A** absorb overflow (0.83% and 0.88% respectively).
- **0.08% cold miss** — near-perfect in the short term.

### Tier Occupancy Over Time
![Tier Occupancy](../plots/heavy_coding/tier_occupancy.png)

L1 fills to ~92%. L2 and L3A absorb overflow. Note: this is the first 60s — at 5+ min, L2 hits 100% and L3A becomes critical.

### TTFT Distribution
![TTFT Distribution](../plots/heavy_coding/ttft_distribution.png)

L1 hits dominate with low TTFT. Cold misses (rare at 60s) show ~17s tail from full 60k-token recompute.

### Savings Class Distribution
![Hit Rate Pie](../plots/heavy_coding/hit_rate_pie.png)

### Queue Depth
![Queue Depth](../plots/heavy_coding/queue_depth.png)

Prefill queue pressured by coding workloads' 14-17s compute times.

### Recompute Fraction
![Recompute Fraction](../plots/heavy_coding/recompute_fraction.png)

Mean recompute fraction = 31%. This is driven by 8-15k new input tokens per turn (user message + tool output), not by prefix instability. Even with 95% prefix stability, the new tokens create 25-35% recompute. See analysis below.

## Capacity Sensitivity (60s sim)

### L1 Capacity Sensitivity
![L1 Sensitivity](../plots/heavy_coding/l1_sensitivity.png)

L1 hit rate appears only above ~10 GB. Below that, coding KV objects (8-16 GB) don't fit. At 80 GB, L1 absorbs most of the 60s working set.

### L2 Capacity Sensitivity
![L2 Sensitivity](../plots/heavy_coding/l2_sensitivity.png)

With realistic L1 (80 GB), L2 mainly handles overflow at 60s. At longer durations, L2 sensitivity would increase as L1 saturates.

---

## Multi-Node Scaling Plots (60s sim)

**Important caveat**: These plots use a 60s simulation. At this duration, L1/L2 absorb most objects, making global and local L3A appear identical. **The true global vs local difference requires 5+ min sims** (see key finding above).

### Node Scaling (1-8 workers × 8 GPUs)
![Node Scaling](../plots/heavy_coding/node_scaling.png)

At 60s, both global and local show ~99.9% hit rate across all worker counts. This reflects L1/L2 absorbing the short-sim working set, not the steady-state behavior.

### TTL Sensitivity (2 workers × 8 GPUs)
![TTL Sensitivity](../plots/heavy_coding/ttl_sensitivity.png)

At 60s, TTL has minimal impact — L1/L2 hold everything. At production durations, shorter L2 TTL would push objects to L3A sooner, amplifying the global vs local difference.

### Sustaining QPS at SLA (p95 queue wait < 500ms)
![Sustaining QPS](../plots/heavy_coding/sustaining_qps.png)

| Workers | GPUs | Global QPS | Local QPS |
|---------|------|-----------|-----------|
| 1 | 8 | 39 | 39 |
| 2 | 16 | 79 | 79 |
| 4 | 32 | 163 | 163 |
| 8 | 64 | 316 | 316 |

At 60s, sustaining QPS scales linearly with workers. Global and local are identical because L3A is barely used. **At production durations, local L3A's 27% hit rate would severely reduce sustaining QPS** due to the 14-17s cold-miss penalty on 73% of requests.

### L3A Latency Sensitivity (2 workers, 60s)

TTFT is flat across 0-100ms remote latency at 60s because L3A is barely used. At production durations where L3A is the primary cache tier, the 50ms remote latency would add ~50ms to every L3A-served request — acceptable given the 14-17s prefill compute time.

---

## Detailed Analysis

### Tier Saturation Over Time (4 workers, realistic hardware)

| Duration | L1 Sat. | L2 Sat. | L3A Sat. | L2→L3A | Cold Evictions | Miss Rate |
|----------|---------|---------|----------|--------|---------------|-----------|
| 1 min | 59% | 75% | 100% | 0 | 0 | 0.09% |
| 5 min | 72% | 100% | 100% | 0 | 25 | 0.19% |
| 10 min | 62% | 100% | 100% | 688 | 1,537 | 0.27% |
| 20 min | 33% | 100% | 99% | 1,409 | 20,613 | 0.23% |

L2 saturates by 5 min. L3A is saturated from the start (objects skip full L2). By 20 min, L3A churns objects at 93× its capacity (740 TB placed through 8 TB/worker).

### Recompute Fraction Analysis

The recompute fraction (mean=31%) may seem high given 98% cache hit rate. These measure different things:

- **Cache hit rate**: was a cached KV object found? (binary)
- **Recompute fraction**: what fraction of tokens must be recomputed? (per request)

```
Total context: ~30k tokens (stable prefix)
Cached tokens: 30k × 0.95 = 28.5k
Uncached from instability: 1.5k
New input tokens (always uncached): 8-15k
Recompute fraction: 9.5k / 38k = 25% (coding) to 16.5k / 46.5k = 35% (agentic_coding)
```

This is realistic — in Claude Code, each turn adds 8-15k new tokens out of a 40-60k total prompt.

### Session Migration Mechanism

The push dispatcher uses cache-affinity routing:
1. Check if session's KV is in any node's L1 or L2
2. If affinity node found and not overloaded → route to it
3. Otherwise → route to least-loaded node

Once objects are evicted from L1/L2 to L3A (happens within minutes), the affinity check returns no match. 97% of dispatches become non-affinity, scattering sessions across workers. With local L3A, each scattered request is a cold miss.

**Potential mitigations**:
- **Global L3A** (recommended): pools all workers' SSDs, making any session's KV accessible from any worker
- **L3A-aware affinity**: extend dispatch affinity check to include L3A (adds cross-worker lookup overhead)
- **Session pinning**: pin sessions to a worker regardless of load (sacrifices load balancing)

---

## Key Findings

1. **Global L3A is essential for multi-worker coding deployments.** At 20 min, global=99.8% vs local=26.6% hit rate. The 73-point gap is caused by session migration after L1/L2 saturate — local L3A can't serve requests that land on a different worker.

2. **Prefill compute is the throughput bottleneck.** 60k-token coding contexts require 14-17s of GPU compute. Sustaining QPS = ~39/worker, limited by compute not cache.

3. **Cache tiers saturate within minutes.** L2 fills at 5 min, L3A from the start. At 20 min, L3A churns objects at 93× capacity. Short sims (60s) are misleading — they show L1 absorbing everything.

4. **L1 is critical but temporary.** 80 GB HBM absorbs 98% of lookups in the first minute, but saturates quickly. The long-term working set far exceeds L1 capacity.

5. **Recompute fraction (~31%) is structural.** Driven by 8-15k new input tokens per turn, not prefix instability. Even with perfect prefix caching, each coding turn recomputes 25-35% of tokens.

6. **50ms global L3A latency is acceptable.** At 14-17s prefill compute, the 50ms remote SSD penalty adds <0.3% overhead. The benefit (73pt hit rate improvement) far outweighs the cost.

## Reproduction

```bash
# Generate all plots (60s snapshots)
python scripts/sanity_plots.py --config configs/heavy_coding.json --outdir plots/heavy_coding

# Run longer sim for global vs local comparison
from sim.config import SimConfig
from sim.engine import SimEngine
config = SimConfig.from_json("configs/heavy_coding.json")
config.sim_duration_s = 1200.0  # 20 min
config.warmup_s = 10.0
config.sim_start_time_s = 36000.0
config.service.n_prefill_nodes = 32  # 4 workers × 8 GPUs
config.service.n_gpus_per_worker = 8
config.service.l3a_shared = True  # or False for comparison
metrics = SimEngine(config).run()
print(metrics.report())
```
