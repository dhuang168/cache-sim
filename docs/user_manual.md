# User Manual

## Installation

Requires Python 3.11+.

```bash
# Clone and install
git clone https://github.com/dhuang168/cache-sim.git
cd cache-sim
pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `polars`, `marisa-trie`, `matplotlib` (for plots).

## Running the Simulator

### 1. Basic Run (Programmatic)

```python
from sim.config import SimConfig
from sim.engine import SimEngine

config = SimConfig.from_json("configs/default.json")
metrics = SimEngine(config).run()
report = metrics.report()
print(report)
```

The `run()` method returns a `MetricsCollector` object with all raw data. Call `.report()` to get a summary dict.

### 2. Running Tests

```bash
# All tests
pytest tests/ -v

# Just invariant tests
pytest tests/test_invariants.py -v

# Just unit tests
pytest tests/test_kv_size.py tests/test_oracle.py -v
```

### 3. Generating Sanity-Check Plots

```bash
python scripts/sanity_plots.py
```

Produces 10 PNG files in the `plots/` directory:
- `tier_occupancy.png` — tier fill levels over time
- `ttft_distribution.png` — TTFT violin plots by cache hit source
- `hit_rate_pie.png` — savings class breakdown
- `queue_depth.png` — GPU queue utilization
- `recompute_fraction.png` — per-request recompute distribution
- `l1_sensitivity.png` — hit rate and eviction rate vs L1 capacity
- `l2_sensitivity.png` — TTFT and hit rate vs L2 capacity
- `node_scaling.png` — 6-panel multi-node analysis: hit rate, eviction, queue wait, queue utilization, TTFT, and L3A latency sensitivity (global vs local L3A)
- `ttl_sensitivity.png` — hit rate and queue wait vs L2 TTL (global vs local L3A comparison)
- `sustaining_qps.png` — max QPS at SLA threshold vs node count (global vs local L3A)

### 4. Running a Parameter Sweep

```bash
python scripts/sweep.py --config configs/default.json --output results/sweep.json
```

Options:
- `--config` — path to base config JSON (required)
- `--output` — where to write results (default: `results/sweep.json`)
- `--workers` — number of parallel workers (default: all CPUs)

The sweep varies these parameters over +/-50% of their nominal values:
- `ttl_l2_s` — L2 TTL before hibernation
- `tiers[1].capacity_bytes` — L2 capacity
- `tiers[1].block_size_bytes` — L2 block size
- `service.n_prefill_slots` — prefill concurrency

Output includes raw results and **elasticity scores** (% change in metric / % change in parameter) for each parameter-metric pair.

## Configuration Reference

The simulator is configured via a JSON file. See `configs/default.json` for the full reference.

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Identifier for this run |
| `seed` | int | Random seed for reproducibility |
| `sim_duration_s` | float | Total simulation time in seconds |
| `warmup_s` | float | Warmup period (metrics not collected) |
| `epoch_report_interval_s` | float | How often to sample tier occupancy and queue depths |
| `ttl_l2_s` | float | Time before L1 objects are moved to L2 |
| `ttl_l3a_s` | float | Time before L2 objects are hibernated to L3A |
| `eviction_hbm_threshold` | float | L1 occupancy threshold triggering eviction (0-1) |
| `eviction_ram_threshold` | float | L2 occupancy threshold triggering eviction (0-1) |

### Tier Configuration (`tiers[]`)

Each tier has:

| Field | Description |
|-------|-------------|
| `name` | `"L1"`, `"L2"`, or `"L3A"` |
| `capacity_bytes` | Total storage capacity |
| `bandwidth_bytes_per_s` | Read bandwidth for KV transfer |
| `latency_floor_us` | Minimum access latency |
| `block_size_bytes` | Allocation granularity |

### Model Configuration (`model`)

| Field | Description |
|-------|-------------|
| `model_id` | Model name (used for benchmark table lookup) |
| `n_layers` | Number of transformer layers |
| `n_kv_heads` | Number of KV attention heads |
| `head_dim` | Dimension per head |
| `bytes_per_element` | 2 for FP16, 1 for FP8 |

KV size formula: `2 * n_layers * n_kv_heads * head_dim * bytes_per_element * token_count`

### Workload Profiles (`profiles[]`)

| Field | Description |
|-------|-------------|
| `name` | Profile identifier |
| `arrival_rate_peak` | Peak new-session arrival rate (sessions/s) |
| `diurnal_peak_trough_ratio` | Ratio of peak to trough in diurnal cycle |
| `iat_mean_s` | Mean inter-arrival time within a session |
| `iat_dist` | `"exponential"` or `"lognormal"` |
| `input_len_mean_tokens` | Mean input length per request |
| `input_len_sigma_tokens` | Spread of input length |
| `output_len_pareto_alpha` | Pareto shape for output length |
| `output_len_pareto_xmin` | Pareto minimum for output length |
| `context_growth_min_tokens` | Min context growth per turn |
| `context_growth_max_tokens` | Max context growth per turn |
| `prefix_stability_initial` | Fraction of context that is a stable prefix (turn 1) |
| `prefix_stability_final` | Fraction of context that is a stable prefix (last turn) |
| `session_duration_mean_s` | Mean session lifetime |
| `session_duration_dist` | `"lognormal"` or `"weibull"` |
| `shared_system_prefix_tokens` | Tokens in the shared system prompt (cross-session) |

### Profile Mix (`profile_mix`)

A dict mapping profile names to weights (must sum to 1.0):

```json
"profile_mix": {
    "chat": 0.4,
    "coding": 0.2,
    "batch": 0.25,
    "agent": 0.15
}
```

### Service Configuration (`service`)

| Field | Description |
|-------|-------------|
| `n_prefill_slots` | Number of concurrent prefill GPU slots (per node) |
| `n_decode_slots` | Number of concurrent decode GPU slots (shared) |
| `prefill_queue_max` | Max prefill queue depth before backpressure (per node) |
| `decode_queue_max` | Max decode queue depth before backpressure |
| `n_prefill_nodes` | Number of prefill nodes (default: 1) |
| `dispatch_algorithm` | `"push"` (affinity routing) or `"pull"` (global queue) |
| `inter_node_latency_us` | Base cross-node transfer latency (default: 5) |
| `inter_node_bandwidth_bytes_per_s` | Cross-node bandwidth (default: 100 GB/s) |
| `l3a_shared` | `true` = global L3A, `false` = per-node L3A (default: true) |
| `l3a_remote_latency_us` | Additional latency for global L3A access (default: 50000 = 50ms) |

## Common Recipes

### Simulate a small L1 (stress test)

```python
config = SimConfig.from_json("configs/default.json")
config.tiers[0].capacity_bytes = 500 * 1024**2  # 500 MB
config.ttl_l2_s = 20.0
config.sim_duration_s = 60.0
config.warmup_s = 5.0
metrics = SimEngine(config).run()
```

### Simulate zero shared prefix

```python
config = SimConfig.from_json("configs/default.json")
for profile in config.profiles:
    profile.shared_system_prefix_tokens = 0
metrics = SimEngine(config).run()
assert abs(metrics.sharing_factor - 1.0) < 0.01
```

### Compare two configurations

```python
import copy

base = SimConfig.from_json("configs/default.json")
base.sim_duration_s = 120.0
base.warmup_s = 10.0

# Variant: double L2 capacity
variant = copy.deepcopy(base)
variant.tiers[1].capacity_bytes *= 2

m_base = SimEngine(base).run()
m_variant = SimEngine(variant).run()

print("Base L2 hit rate:", m_base.report()["cache_hit_rate"]["L2"])
print("2x L2 hit rate:", m_variant.report()["cache_hit_rate"]["L2"])
```

### Simulate multi-node prefill (4 nodes, push dispatch)

```python
config = SimConfig.from_json("configs/default.json")
config.service.n_prefill_nodes = 4
config.service.dispatch_algorithm = "push"
config.sim_duration_s = 60.0
config.warmup_s = 5.0
metrics = SimEngine(config).run()
report = metrics.report()
print(f"Queue wait p95: {report['queue_wait_ms']['p95']:.1f} ms")
print(f"Dispatch stats: {report.get('dispatch_stats', {})}")
```

### Compare global vs local L3A

```python
import copy

base = SimConfig.from_json("configs/default.json")
base.service.n_prefill_nodes = 4
base.sim_duration_s = 60.0
base.warmup_s = 5.0

# Global L3A (shared pool with remote latency)
cfg_global = copy.deepcopy(base)
cfg_global.service.l3a_shared = True
cfg_global.service.l3a_remote_latency_us = 50_000

# Local L3A (per-node, capacity/N, no remote penalty)
cfg_local = copy.deepcopy(base)
cfg_local.service.l3a_shared = False

m_global = SimEngine(cfg_global).run()
m_local = SimEngine(cfg_local).run()
print("Global miss rate:", m_global.report()["cache_hit_rate"]["miss"])
print("Local miss rate:", m_local.report()["cache_hit_rate"]["miss"])
```

### Find sustaining QPS at SLA

```python
from sim.analysis import find_sustaining_qps

config = SimConfig.from_json("configs/default.json")
config.service.n_prefill_nodes = 4
config.sim_duration_s = 10.0
config.warmup_s = 1.0

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

# Raw TTFT samples (microseconds)
l2_ttft_us = metrics.ttft_us["L2_hit"]

# Tier occupancy time-series (%)
l2_occ = metrics.tier_occupancy_pct["L2"]

# Eviction counts
print(f"L1->L2: {metrics.l1_to_l2_evictions}")
print(f"L2->L3A: {metrics.l2_to_l3a_evictions}")
print(f"Cold: {metrics.session_cold_evictions}")

# Savings event counts
for cls, count in metrics.savings_events.items():
    print(f"  {cls}: {count}")

# Queue wait time (multi-node)
print(f"Queue wait samples: {len(metrics.queue_wait_us)}")
if metrics.queue_wait_us:
    import numpy as np
    print(f"Queue wait p95: {np.percentile(metrics.queue_wait_us, 95) / 1000:.1f} ms")

# Per-node metrics
for node_id, count in metrics.per_node_prefill_count.items():
    print(f"Node {node_id}: {count} prefills")
print(f"Affinity dispatches: {metrics.affinity_dispatches}")
print(f"Cross-node transfers: {metrics.cross_node_transfers}")
```

## Benchmark Tables

Prefill latency is driven by real measurements stored in `benchmarks/latency_tables/`. The default table is for Llama3-70B on A100-80GB (FP16, single sequence, no batching):

| Tokens | Latency (ms) |
|--------|-------------|
| 512    | 45          |
| 1024   | 90          |
| 2048   | 200         |
| 4096   | 520         |
| 8192   | 1,400       |
| 16384  | 4,800       |
| 32768  | 17,000      |

To add a new model/GPU, create a JSON file in `benchmarks/latency_tables/` with the same schema (`tokens` and `latency_us` arrays).

## Performance Notes

- A 60-second simulation with 5-second warmup runs in ~1-2 minutes on an M-series Mac
- A full 3600-second simulation takes proportionally longer
- The parameter sweep (`sweep.py`) parallelizes across CPUs
- The invariant test suite takes ~1 minute total (4 simulation runs at 60s each)
- Multi-node (4 nodes, 10s sim) completes in ~3.5s wall time
- Full test suite (22 tests) runs in ~2 minutes
