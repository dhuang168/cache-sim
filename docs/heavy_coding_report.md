# Heavy Coding Workload Report

**Config**: `configs/heavy_coding.json` | **Simulation**: 20 minutes | **Model**: Llama3-70B FP16 on A100-80GB

## Workload Profile

90% coding workloads reflecting AI coding assistant deployments (Claude Code, Cursor).

| Profile | Mix | System Prefix | Input/Turn | Context Growth | Session Duration |
|---------|-----|---------------|-----------|----------------|-----------------|
| coding | 45% | 20k tokens | 8k tokens | 2-10k/turn | 1 hour |
| agentic_coding | 45% | 30k tokens | 15k tokens | 5-20k/turn | 30 min |
| chat | 5% | 2k tokens | 150 tokens | 100-500/turn | 10 min |
| batch | 3% | 2k tokens | 1.2k tokens | none | 1s |
| agent | 2% | 2k tokens | 400 tokens | 300-1.5k/turn | 30 min |

**KV object sizes** (70B FP16): coding ~8.2 GB, agentic_coding ~15.6 GB, chat ~0.8 GB.

## Hardware

| Tier | Capacity | Bandwidth | Scope |
|------|----------|-----------|-------|
| L1 (HBM) | 80 GB | 3 TB/s | Per-GPU |
| L2 (DRAM) | 1 TB | 64 GB/s | Per-worker (8 GPUs share) |
| L3A (SSD) | 8 TB | 7 GB/s | Per-worker; global mode pools all workers |

Worker topology: 8 GPUs/worker. Global L3A: 4 workers × 8 TB = 32 TB pooled. Local L3A: 8 TB/worker.

---

## 1. Global vs Local L3A

The central finding: **global L3A is essential for multi-worker coding deployments.**

### Hit Rate Over Time (4 workers × 8 GPUs, 20 min)

| Duration | Global L3A | Local L3A | Gap |
|----------|-----------|----------|-----|
| 1 min | 99.91% | 99.91% | 0% |
| 5 min | 99.81% | **55.23%** | **+44.6%** |
| 10 min | 99.73% | **36.74%** | **+63.0%** |
| 20 min | 99.77% | **26.56%** | **+73.2%** |

### Timeline: L2/L3A Occupancy, Queue Depth, Cold Evictions
![Global vs Local Timeline](../plots/heavy_coding/global_vs_local_timeline.png)

Both global and local L3A reach 100% occupancy quickly. The difference: with local L3A, cold evictions spike as migrated sessions can't find their KV → queue depth grows from cold-miss recomputes.

### Node Scaling (1-4 workers, 20 min)
![Node Scaling 20min](../plots/heavy_coding/node_scaling_20min.png)

| Workers | GPUs | Global Hit | Local Hit | Local Misses |
|---------|------|-----------|----------|-------------|
| 1 | 8 | 99.8% | 99.8% | 1,921 |
| 2 | 16 | 99.8% | **49.3%** | 418,173 |
| 4 | 32 | 99.8% | **26.6%** | 606,005 |

At 1 worker: identical (no cross-worker migration). At 2+ workers: local L3A collapses because sessions migrate and can't find KV on the new worker.

**Note on queue wait p95**: The node scaling plot shows queue wait ~80s for global and ~70s for local. This is **not** meaningful — the system is so overloaded (arrival 693/s vs throughput 78/s) that **88.7% of requests are dropped** before entering the queue. Both queues are permanently full (~4000 depth). The 10s difference is noise on equally-saturated queues. Queue wait is only meaningful when the system is not in severe overload.

### TTL Sensitivity (4 workers, 20 min)
![TTL Sensitivity 20min](../plots/heavy_coding/ttl_sensitivity_20min.png)

| L2 TTL | Global Hit | Local Hit |
|--------|-----------|----------|
| 10s | 99.8% | 26.5% |
| 30s | 99.8% | 33.8% |
| 60s | 99.8% | 26.9% |
| 120s | 99.8% | 27.1% |
| 300s | 99.8% | 26.6% |

**TTL has minimal impact** on either mode. Global L3A maintains 99.8% regardless. Local L3A stays at ~27% regardless — the bottleneck is cross-worker accessibility, not object lifetime.

---

## 2. Single Worker Deep Dive (8 GPUs, 20 min)

![Single Worker 20min](../plots/heavy_coding/single_worker_20min.png)

With a single worker, there's no session migration — all objects stay local. **Hit rate: 99.8%.**

Key observations:
- **L1 occupancy fluctuates** (33-60%) — coding objects are large (8-16 GB) relative to 80 GB L1, so a few objects fill it
- **L2 saturates at 100%** within minutes — 1 TB can't hold the growing working set
- **L3A becomes the primary cache** — most lookups eventually find objects in SSD
- **Slot utilization near 100%** — coding prefills take 14-17s, keeping slots busy
- **Recompute fraction ~31%** — each turn adds 8-15k new tokens (user message + tool output), even with 95% prefix stability
- **TTFT**: L1 hits ~100ms, L3A hits ~1-5s, cold misses ~17s

---

## 3. Multi-Worker Deep Dive (4 workers × 8 GPUs, 20 min)

![Multi Worker 20min](../plots/heavy_coding/multi_worker_20min.png)

Global (green) vs Local (orange, dashed) — same hardware, different L3A mode.

Key observations:
- **L1/L2/L3A occupancy identical** — both modes use the same per-worker storage, both saturated (~100% L2/L3A)
- **Queue depth identical** — both modes are equally saturated (~4000 depth, near max). The system is throughput-limited regardless of cache mode
- **Slot utilization identical** — both near 100%. Every slot is busy whether doing a cache-hit prefill or a cold-miss recompute
- **The difference is invisible in infrastructure metrics.** Global and local look identical in occupancy, queue depth, and utilization. The 73pt hit rate gap shows up only in **what each slot computes**: global slots do partial recompute (cache hit, fast), local slots do full recompute (cold miss, 14-17s). Same slot utilization, vastly different useful throughput.

**Why don't the per-epoch plots show the difference?** Because per-epoch infrastructure metrics (occupancy, queue depth, slot utilization) measure *resource consumption*, not *request-level performance*. Both modes saturate all slots and queues equally. The difference shows up in **TTFT by cache hit type**:

| Component | Global TTFT (mean) | Global Count | Local TTFT (mean) | Local Count |
|-----------|-------------------|-------------|-------------------|-------------|
| L1 hit | 4.8s | 10,779 | 6.4s | 19,881 |
| L3A hit | 31.6s | 23,363 | 33.3s | 5,332 |
| Cold miss | 16.6s | 21 | **52.8s** | **7,195** |

*(5-min sim, 4 workers)*

Key observations:
- **Local cold misses take 52.8s** average TTFT (vs global's 16.6s for the rare cold miss). The 52.8s includes both the 17s recompute AND the cascading queue wait from other cold misses blocking slots.
- **Local has 7,195 cold misses** vs global's 21 — a 340× increase.
- **Local has more L1 hits** (19,881 vs 10,779) because when sessions re-land on the same worker after a cold miss, the freshly recomputed KV goes into L1 — but this is wasted work since the KV already existed in another worker's L3A.
- **The `prefill_duration` metric (compute only, no queue wait)** is similar for both modes (~7.5s) because it doesn't capture the cascading queue delay. The real impact shows in TTFT, which includes queue wait.

Dispatch stats at 20 min:
- Global: 878 affinity, 824,278 non-affinity (0.1% affinity)
- Local: 194,950 affinity, 630,206 non-affinity (24% affinity)

---

## 4. Session Migration

### Why Sessions Migrate

The push dispatcher routes requests to nodes with cache affinity (session's KV in L1/L2). But:
1. L1 (80 GB/GPU) and L2 (1 TB/worker) saturate within minutes
2. Objects are evicted to L3A via TTL or pressure
3. Once in L3A, the affinity check **no longer detects them** (it only checks L1/L2)
4. The session loses affinity → next request dispatched to any available node
5. At steady state, **97-99% of dispatches are non-affinity**

### Impact

Each migrated request with local L3A is a **cold miss** (14-17s full recompute) because the new worker's SSD doesn't have the KV. With 97% non-affinity dispatch, this means 97% of requests recompute from scratch.

### Mitigations

| Approach | Benefit | Cost |
|----------|---------|------|
| **Global L3A** | 99.8% hit rate, any worker finds any KV | 50ms remote latency (negligible vs 14s compute) |
| **L3A-aware affinity** | Route to worker whose SSD has the KV | Cross-worker L3A lookup at dispatch time |
| **Session pinning** | Avoid migration entirely | Sacrifices load balancing |

---

## 5. Summary

| Finding | Evidence |
|---------|----------|
| **Global L3A is essential** for multi-worker deployments | 99.8% vs 26.6% hit rate at 20 min (73pt gap) |
| **Session migration is the mechanism** | 97% non-affinity dispatch at steady state |
| **Infrastructure metrics don't show the gap** | Queue depth, slot utilization, occupancy identical. Both drop 88.7% of requests. The gap is in useful work per completed request. |
| **System is severely overloaded** | Arrival 693/s vs throughput 78/s. 88.7% of requests dropped. Queue wait p95 (~70-80s) is noise on permanently-full queues. |
| **Prefill compute is the throughput bottleneck** | Mean 7.5s/request, 1024 slots → max ~137/s theoretical, actual 78/s |
| **L1 is critical but temporary** | 98% L1 hits at 1 min, saturates by 5 min |
| **50ms global L3A latency is negligible** | <0.3% overhead on 7.5s mean prefill |
| **TTL has no impact** on global vs local | Both modes insensitive to L2 TTL (10-300s) |
| **Recompute fraction ~31% is structural** | Driven by 8-15k new input tokens/turn, not instability |

## Reproduction

```bash
# Generate all 20-min analysis plots
python scripts/heavy_coding_analysis.py

# Quick comparison (programmatic)
from sim.config import SimConfig
from sim.engine import SimEngine
config = SimConfig.from_json("configs/heavy_coding.json")
config.sim_duration_s = 1200.0  # 20 min
config.warmup_s = 10.0
config.sim_start_time_s = 36000.0
config.service.n_prefill_nodes = 32  # 4 workers × 8 GPUs
config.service.n_gpus_per_worker = 8
config.service.l3a_shared = True  # or False for local
metrics = SimEngine(config).run()
print(metrics.report())
```
