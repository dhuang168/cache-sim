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

**Prefill compute cost** scales quadratically with token count (O(n²) attention):

| Tokens | Prefill Latency |
|--------|----------------|
| 10,000 | 2.2s |
| 30,000 | 14.9s |
| 50,000 | 37.0s |
| 65,000 | 54.4s |
| 100,000 | 120.7s |

A cache hit on a 95k-token agentic_coding context saves **98s** (13s hit vs 111s cold miss — 88% reduction).

## Hardware

| Tier | Capacity | Bandwidth | Scope |
|------|----------|-----------|-------|
| L1 (HBM) | 80 GB | 3 TB/s | Per-GPU |
| L2 (DRAM) | 1 TB | 64 GB/s | Per-worker (8 GPUs share) |
| L3A (SSD) | 8 TB | 7 GB/s | Per-worker; global mode pools all workers |

Worker topology: 8 GPUs/worker. Global L3A: N workers × 8 TB = pooled. Local L3A: 8 TB/worker.

---

## 1. Global vs Local L3A — The Central Finding

**Global L3A is essential for multi-worker coding deployments.**

### Hit Rate Over Time (4 workers × 8 GPUs, 20 min)

![Global vs Local Timeline](../plots/heavy_coding/global_vs_local_timeline.png)

| Metric | Global L3A | Local L3A |
|--------|-----------|----------|
| Hit rate | **99.8%** | **68.6%** |
| Cold misses | 1,921 | **259,179** |
| Cold evictions | 13,849 | 3,492 |

Global maintains 99.8% throughout. Local degrades to 68.6% as sessions migrate between workers and can't find their KV on the new worker's SSD.

### Node Scaling (1-4 workers, 20 min)

![Node Scaling 20min](../plots/heavy_coding/node_scaling_20min.png)

| Workers | GPUs | Global Hit | Global QW p95 | Local Hit | Local QW p95 |
|---------|------|-----------|--------------|----------|-------------|
| 1 | 8 | 99.8% | 115s | 99.8% | 115s |
| 2 | 16 | 99.8% | 113s | **86.5%** | **250s** |
| 4 | 32 | 99.8% | 116s | **68.6%** | **394s** |

At 1 worker: identical (no migration). At 2+ workers: local queue wait **explodes** (250-394s p95) because cold misses take 37-120s each, blocking slots and cascading into queue buildup. Global maintains steady 115s queue wait.

### TTL Sensitivity (4 workers, 20 min)

![TTL Sensitivity 20min](../plots/heavy_coding/ttl_sensitivity_20min.png)

| L2 TTL | Global Hit | Global QW p95 | Local Hit | Local QW p95 |
|--------|-----------|--------------|----------|-------------|
| 10s | 99.8% | 115s | 69.5% | 357s |
| 60s | 99.8% | 115s | 69.9% | 379s |
| 300s | 99.8% | 116s | 68.6% | 394s |

TTL has no impact on either mode. The bottleneck is cross-worker L3A accessibility, not object lifetime.

### Controlled Load Tests (4 workers, 20 min)

At reduced arrival rates where the system isn't in severe overload:

**Test (a): ~75 req/s arrival (peak=3)**

| Metric | Global | Local | Global Advantage |
|--------|--------|-------|-----------------|
| Hit rate | 99.8% | 76.7% | +23pt |
| Dropped | 4% | **40%** | 10× fewer drops |
| Queue wait mean | 2.7s | **50.8s** | **19× shorter** |
| Queue wait p95 | 19s | **244s** | **13× shorter** |
| TTFT mean | 25s | **80s** | **3.1× faster** |
| TTFT p95 | 72s | **314s** | **4.4× faster** |

**Test (b): ~125 req/s arrival (peak=5) — completion count**

| Metric | Global | Local | Advantage |
|--------|--------|-------|-----------|
| Completed | 32,987 | 17,946 | **83.8% more** |
| Dropped | 20% | **57%** | |
| TTFT mean | 42s | **94s** | **2.2× faster** |

Global completes **84% more requests** in the same 20 minutes.

---

## 2. Single Worker Deep Dive (8 GPUs, 20 min)

![Single Worker 20min](../plots/heavy_coding/single_worker_20min.png)

With a single worker, no session migration — all objects stay local. **Hit rate: 99.8%.**

```
Tier saturation: L1=61%  L2=100%  L3A=100%
Prefill:  mean=13.4s  median=8.4s  p95=45.4s
Queue wait: mean=51s  median=45s  p95=115s
Slot utilization: 100%
```

- **L2 and L3A saturated at 100%** — the 47 TB working set vastly exceeds storage
- **Prefill p95 = 45s** — large agentic_coding contexts (50-100k tokens) take 37-120s
- **Slot utilization 100%** — all 256 slots (32/GPU × 8) are always busy
- **Recompute fraction ~31%** — 8-15k new tokens per turn, even with 95% prefix stability

---

## 3. Multi-Worker Deep Dive (4 workers × 8 GPUs, 20 min)

![Multi Worker 20min](../plots/heavy_coding/multi_worker_20min.png)

Global (green solid) vs Local (orange dashed) — same hardware, different L3A mode.

Key observations:
- **L1/L2/L3A occupancy identical** — both saturated, same per-worker storage
- **Queue depth diverges** — local queue depth grows higher as cold misses (37-120s each) block slots, cascading into longer waits for all subsequent requests
- **Slot utilization both 100%** — but global slots do useful work (cache hits), local slots waste time on full recomputes
- **Cold evictions**: global has more (13,849 vs 3,492) — global L3A churns objects across workers; local has fewer evictions but the evicted objects cause catastrophic cold misses

Dispatch stats:
- Global: 890 affinity (0.1%), 824,266 non-affinity — near-total session migration
- Local: 539,089 affinity (65%), 286,067 non-affinity — higher affinity because cold misses recompute KV locally, creating temporary L1/L2 affinity

---

## 4. Session Migration

### Mechanism

1. **L1/L2 saturate within minutes.** L2 (1 TB/worker) fills at ~5 min.
2. **Objects evict to L3A.** Via TTL or pressure.
3. **Affinity lost.** Dispatch affinity only checks L1/L2. Once in L3A, no affinity signal.
4. **Session scatters.** 97-99% of dispatches are non-affinity at steady state.
5. **Local L3A = cold miss.** New worker's SSD doesn't have the KV. Full recompute: 37-120s.
6. **Global L3A = cache hit.** Pooled SSD has the KV. Transfer + partial compute: 5-15s.

### Cost Per Request

| Context Size | Cache Hit (transfer + partial) | Cold Miss (full recompute) | Savings |
|-------------|-------------------------------|---------------------------|---------|
| 30k coding | 4.5s | 14.9s | 70% |
| 50k agentic | 9.5s | 37.0s | 74% |
| 80k agentic | 13.1s | 111.2s | **88%** |
| 100k agentic | 16.8s | 120.7s | **86%** |

### Mitigations

| Approach | Benefit | Cost |
|----------|---------|------|
| **Global L3A** (recommended) | 99.8% hit rate from any worker | 50ms + transfer time per L3A access |
| **L3A-aware affinity** | Route to worker whose SSD has KV | Cross-worker L3A lookup at dispatch |
| **Session pinning** | Avoid migration entirely | Sacrifices load balancing |

---

## 5. Summary

| Finding | Evidence |
|---------|----------|
| **Global L3A is essential** | 99.8% vs 68.6% hit rate at 20 min; 84% more completed requests |
| **Session migration is the cause** | 97% non-affinity dispatch; local L3A can't serve migrated sessions |
| **Queue wait explodes without global L3A** | Global 2.7s vs local 50.8s mean queue wait (19×) at controlled load |
| **Cold miss cost is massive** | 37-120s full recompute vs 5-15s cache hit (70-88% savings) |
| **Infrastructure metrics mask the gap** | Occupancy, slot utilization identical — the difference is in TTFT and throughput |
| **TTL has no impact** | Hit rate insensitive to L2 TTL (10-300s) for both modes |
| **Recompute fraction ~31% is structural** | 8-15k new input tokens per turn, not prefix instability |

## Reproduction

```bash
# Generate all 20-min analysis plots
python scripts/heavy_coding_analysis.py

# Run controlled load test
python -c "
from sim.config import SimConfig
from sim.engine import SimEngine
import copy
cfg = SimConfig.from_json('configs/heavy_coding.json')
cfg.sim_duration_s = 1200.0; cfg.warmup_s = 10.0; cfg.sim_start_time_s = 36000.0
cfg.service.n_prefill_nodes = 32; cfg.service.n_gpus_per_worker = 8
for p in cfg.profiles: p.arrival_rate_peak = 3  # controlled load
for shared in [True, False]:
    c = copy.deepcopy(cfg)
    c.service.l3a_shared = shared
    c.service.l3a_remote_latency_us = 50000 if shared else 0
    m = SimEngine(c).run()
    label = 'Global' if shared else 'Local'
    total = sum(m.savings_events.values())
    miss = m.savings_events.get('COLD_MISS', 0)
    print(f'{label}: hit={1-miss/total:.1%} completed={len(m.queue_wait_us)}')
"
```
