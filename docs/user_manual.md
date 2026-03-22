# User Manual

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/dhuang168/cache-sim.git
cd cache-sim
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `polars`, `marisa-trie`, `matplotlib` (for plots).

## Running the Simulator

### 1. Basic Run

```python
from sim.config import SimConfig
from sim.engine import SimEngine

config = SimConfig.from_json("configs/default.json")
config.sim_duration_s = 60.0
config.warmup_s = 5.0
config.sim_start_time_s = 36000.0  # 10 AM — ensures all profiles have nonzero arrival rate
metrics = SimEngine(config).run()
report = metrics.report()
print(report)
```

**Important**: Set `sim_start_time_s` to a business-hours offset (e.g., 36000 = 10 AM). The default (0 = midnight) produces zero arrival rate for most profiles due to the diurnal model.

### 2. Running Tests

```bash
# All 27 tests
pytest tests/ -v

# Just invariant tests (4 tests)
pytest tests/test_invariants.py -v

# Just multi-node tests (14 tests)
pytest tests/test_multinode.py -v

# With performance benchmark
pytest tests/ -v --benchmark-only
```

### 3. Generating Plots

```bash
# Default config — 10 diagnostic plots
python scripts/sanity_plots.py

# Custom config with output directory
python scripts/sanity_plots.py --config configs/heavy_coding.json --outdir plots/heavy_coding

# 20-min heavy coding analysis (5 plots)
python scripts/heavy_coding_analysis.py
```

**sanity_plots.py** produces 10 PNGs: tier occupancy, TTFT distribution, hit rate pie, queue depth, recompute fraction, L1 sensitivity, L2 sensitivity, node scaling (global vs local L3A), TTL sensitivity, sustaining QPS.

**heavy_coding_analysis.py** produces 5 PNGs at 20-min duration: global vs local timeline, single worker deep dive, multi-worker deep dive, node scaling, TTL sensitivity.

### 4. Parameter Sweep

```bash
python scripts/sweep.py --config configs/default.json --output results/sweep.json
```

---

## Configuration Reference

Three configs are provided:

| Config | Purpose |
|--------|---------|
| `configs/default.json` | v2 reference — realistic hardware, mixed workload |
| `configs/heavy_coding.json` | 90% coding workload for cache analysis |
| `configs/legacy_v1.json` | Original v1 with small token counts |

### Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_id` | string | — | Identifier for this run |
| `seed` | int | — | Random seed for reproducibility |
| `sim_duration_s` | float | — | Total simulation time in seconds |
| `warmup_s` | float | — | Warmup period (metrics not collected) |
| `epoch_report_interval_s` | float | — | Sampling interval for time-series metrics |
| `sim_start_time_s` | float | 0.0 | Wall-clock offset for diurnal rate. 0=midnight, 32400=9AM, 36000=10AM. **Set to 36000 for short sims** to ensure all profiles have nonzero arrival rates. |
| `ttl_l2_s` | float | — | TTL before L1 objects migrate to L2 |
| `ttl_l3a_s` | float | — | TTL before L2 objects hibernate to L3A |
| `eviction_hbm_threshold` | float | — | L1 occupancy threshold triggering eviction (0-1) |
| `eviction_ram_threshold` | float | — | L2 occupancy threshold triggering eviction (0-1) |

### Tier Configuration (`tiers[]`)

Order must be [L1, L2, L3A]. Capacity semantics depend on tier and topology:

| Field | L1 | L2 | L3A |
|-------|----|----|-----|
| `capacity_bytes` | Per-GPU | Per-worker | Per-worker SSD |
| `bandwidth_bytes_per_s` | GPU HBM bandwidth | Host DRAM bandwidth | SSD bandwidth |
| `latency_floor_us` | HBM latency | DRAM latency | SSD latency |
| `block_size_bytes` | 5 KB typical | 32 MB typical | 256 MB typical |

**L3A capacity note**: The config value is per-worker SSD capacity. In global mode, total pool = `per_worker × n_workers`. In local mode, each worker uses its own SSD.

### Model Configuration (`model`)

| Field | Description |
|-------|-------------|
| `model_id` | Model name |
| `n_layers` | Transformer layers (80 for 70B) |
| `n_kv_heads` | KV attention heads (8 for GQA) |
| `head_dim` | Dimension per head (128) |
| `bytes_per_element` | 2 for FP16, 1 for FP8 |

KV size: `2 × n_layers × n_kv_heads × head_dim × bytes_per_element × token_count`

For 70B FP16: 327,680 bytes/token. A 50k-token context = 15.6 GB.

### Workload Profiles (`profiles[]`)

| Field | Description |
|-------|-------------|
| `name` | Profile identifier |
| `arrival_rate_peak` | Base new-session rate (sessions/s). Actual rate = `peak × profile_mix[name] × diurnal_factor`. |
| `diurnal_peak_trough_ratio` | Peak/trough ratio in 24h cycle. 2.0 = peak is 2× trough. |
| `iat_mean_s` | Mean inter-arrival time within a session |
| `iat_dist` | `"exponential"` or `"lognormal"` |
| `input_len_mean_tokens` | Mean input length per request (new tokens each turn) |
| `input_len_sigma_tokens` | Spread of input length |
| `output_len_pareto_alpha` | Pareto shape for output length |
| `output_len_pareto_xmin` | Pareto minimum for output length |
| `context_growth_min_tokens` | Min context growth per turn |
| `context_growth_max_tokens` | Max context growth per turn |
| `prefix_stability_initial` | Fraction of context that is stable prefix (turn 1) |
| `prefix_stability_final` | Fraction of context that is stable prefix (last turn) |
| `session_duration_mean_s` | Mean session lifetime |
| `session_duration_dist` | `"lognormal"` or `"weibull"` |
| `shared_system_prefix_tokens` | System prompt tokens shared across all sessions of this profile |

### Profile Mix (`profile_mix`)

Weights that **scale each profile's arrival rate**. Must sum to 1.0.

```json
"profile_mix": {
    "chat": 0.30,
    "coding": 0.20,
    "batch": 0.20,
    "agent": 0.15,
    "agentic_coding": 0.15
}
```

The effective new-session rate for each profile is `arrival_rate_peak × profile_mix[name] × diurnal_factor(t)`. With all profiles at `arrival_rate_peak=100` and mix weights summing to 1.0, the total peak session rate ≈ 100 sessions/s.

### Service Configuration (`service`)

| Field | Default | Scope | Description |
|-------|---------|-------|-------------|
| `n_prefill_slots` | 32 | Per-GPU | Concurrent prefill GPU slots |
| `n_decode_slots` | 256 | Shared | Concurrent decode GPU slots |
| `prefill_queue_max` | 128 | Per-GPU | Max prefill queue before backpressure (requests dropped) |
| `decode_queue_max` | 512 | Shared | Max decode queue |
| `n_prefill_nodes` | 1 | Cluster | Total GPU count. Must be divisible by `n_gpus_per_worker`. |
| `n_gpus_per_worker` | 8 | — | GPUs per server. GPUs on same worker share L2 (DRAM) and L3A (SSD). |
| `dispatch_algorithm` | `"push"` | — | `"push"` (affinity routing) or `"pull"` (global queue) |
| `inter_node_latency_us` | 5 | — | Base NVLink latency between GPUs |
| `inter_node_bandwidth_bytes_per_s` | 100 GB/s | — | Cross-node bandwidth |
| `l3a_shared` | `true` | — | `true` = global L3A (pooled SSDs), `false` = local (own SSD only) |
| `l3a_remote_latency_us` | 50,000 | — | Extra latency for global L3A access (only when `l3a_shared=true` and >1 worker) |

**Worker topology**: `n_prefill_nodes / n_gpus_per_worker` = number of workers. Example: 32 nodes / 8 per worker = 4 workers. Each worker has 8 GPUs sharing 1 TB DRAM and 8 TB SSD.

---

## Common Recipes

### Run heavy coding workload analysis

```python
config = SimConfig.from_json("configs/heavy_coding.json")
config.sim_duration_s = 1200.0  # 20 min (needed for steady state)
config.warmup_s = 10.0
config.sim_start_time_s = 36000.0
config.service.n_prefill_nodes = 32  # 4 workers × 8 GPUs
config.service.n_gpus_per_worker = 8
config.service.l3a_shared = True  # global L3A
metrics = SimEngine(config).run()
```

### Compare global vs local L3A (20-min, 4 workers)

```python
import copy
base = SimConfig.from_json("configs/heavy_coding.json")
base.sim_duration_s = 1200.0
base.warmup_s = 10.0
base.sim_start_time_s = 36000.0
base.service.n_prefill_nodes = 32
base.service.n_gpus_per_worker = 8

for shared in [True, False]:
    c = copy.deepcopy(base)
    c.service.l3a_shared = shared
    c.service.l3a_remote_latency_us = 50_000 if shared else 0
    m = SimEngine(c).run()
    total = sum(m.savings_events.values())
    miss = m.savings_events.get("COLD_MISS", 0)
    label = "Global" if shared else "Local"
    print(f"{label}: hit={1-miss/total:.1%} miss={miss}")
```

### Controlled load test (reduced arrival rate)

```python
config = SimConfig.from_json("configs/heavy_coding.json")
config.sim_duration_s = 1200.0
config.warmup_s = 10.0
config.sim_start_time_s = 36000.0
config.service.n_prefill_nodes = 32
config.service.n_gpus_per_worker = 8
for p in config.profiles:
    p.arrival_rate_peak = 3  # reduce from 100 to avoid severe overload
metrics = SimEngine(config).run()
```

### Find sustaining QPS at SLA

```python
from sim.analysis import find_sustaining_qps

config = SimConfig.from_json("configs/heavy_coding.json")
config.sim_duration_s = 300.0  # 5 min
config.warmup_s = 10.0
config.sim_start_time_s = 36000.0
config.service.n_prefill_nodes = 32
config.service.n_gpus_per_worker = 8

mult, report = find_sustaining_qps(
    config, sla_queue_wait_p95_ms=500.0,
    rate_range=(0.1, 5.0), max_iterations=8,
)
base_qps = sum(p.arrival_rate_peak * config.profile_mix.get(p.name, 0)
               for p in config.profiles)
print(f"Sustaining QPS: {mult * base_qps:.0f}")
```

### Access raw metrics

```python
metrics = SimEngine(config).run()

# TTFT by source (microseconds)
for src in ["L1_hit", "L2_hit", "L3A_hit", "cold_miss"]:
    data = metrics.ttft_us.get(src, [])
    if data:
        import numpy as np
        print(f"{src}: n={len(data)} mean={np.mean(data)/1000:.0f}ms p95={np.percentile(data,95)/1000:.0f}ms")

# Tier occupancy time-series (%)
l2_occ = metrics.tier_occupancy_pct["L2"]

# Queue wait and prefill duration
print(f"Queue wait p95: {np.percentile(metrics.queue_wait_us, 95)/1000:.0f}ms")
print(f"Prefill duration mean: {np.mean(metrics.prefill_duration_us)/1000:.0f}ms")
print(f"Slot utilization: {np.mean(metrics.slot_utilization_pct):.0f}%")

# Dispatch stats (multi-node)
print(f"Affinity dispatches: {metrics.affinity_dispatches}")
print(f"Non-affinity: {metrics.non_affinity_dispatches}")
print(f"Cross-node transfers: {metrics.cross_node_transfers}")

# Per-node breakdown
for node_id, count in metrics.per_node_prefill_count.items():
    print(f"GPU {node_id}: {count} prefills")
```

---

## Benchmark Tables

Prefill latency is driven by measurements in `benchmarks/latency_tables/`. The table for Llama3-70B on A100-80GB (FP16, single sequence):

| Tokens | Latency | Notes |
|--------|---------|-------|
| 512 | 45 ms | Measured |
| 1,024 | 90 ms | Measured |
| 2,048 | 200 ms | Measured |
| 4,096 | 520 ms | Measured |
| 8,192 | 1.4 s | Measured |
| 16,384 | 4.8 s | Measured |
| 32,768 | 17 s | Measured |
| 65,536 | 55 s | Extrapolated (O(n²) attention) |
| 131,072 | 180 s | Extrapolated |
| 262,144 | 600 s | Extrapolated |

The oracle uses piecewise-linear interpolation (`np.interp`) and clamps at table bounds. **The table must cover the token range of your workload** — if your workload generates 100k-token contexts, ensure the table has entries above 100k.

To add a new model/GPU, create a JSON file in `benchmarks/latency_tables/` with `tokens` and `latency_us` arrays.

---

## Important Caveats

### Simulation Duration

Short sims (60s) have **not reached steady state** for coding workloads. L2 saturates at ~5 min, session migration effects appear at ~5 min. **Use 20+ min for multi-worker global vs local L3A comparisons.**

### Arrival Rate and Overload

If the system drops >50% of requests, queue metrics are meaningless (survivorship bias). Reduce `arrival_rate_peak` or add more workers. Check `len(metrics.queue_wait_us)` vs total events to detect drops.

### Config Consistency

When changing any parameter, verify dependent parameters:
- Changing workload tokens → check oracle table covers the range
- Changing arrival rates → check system isn't in severe overload
- Changing workers → check L3A capacity semantics (per-worker × N for global)
- Changing sim duration → check steady state reached

See `plan_and_progress/development_methodology.md` for the full dependency map.

## Performance Notes

- 60s sim: ~1-2s wall time (single worker)
- 20-min sim: ~2-5 min wall time (depends on workers and load)
- Full test suite: 27 tests in ~4s
- heavy_coding_analysis.py: ~50 min (runs ~15 × 20-min sims)
