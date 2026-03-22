# PCS Cache Simulator

A discrete-event simulator for multi-tier prompt cache systems. It models the full lifecycle of KV cache objects across a three-tier storage hierarchy (L1/L2/L3A) to evaluate caching strategies, eviction policies, and their impact on Time-To-First-Token (TTFT) for LLM inference workloads.

Supports **multi-node prefill dispatch** with worker topology (8 GPUs per worker sharing DRAM/SSD), shared or local L3A, and push/pull dispatch algorithms with cache-affinity awareness.

## Architecture

```
                          ┌──────────────────────┐
                          │   WorkloadSynthesizer │
                          │  (NHPP + 5 profiles)  │
                          └──────────┬───────────┘
                                     │ REQUEST_ARRIVAL
                                     ▼
┌────────────────────────────────────────────────────────────┐
│                        SimEngine                           │
│                                                            │
│   sim_clock_us ──► min-heap event loop                     │
│                                                            │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  Worker 0 (8 GPUs)           Worker N (8 GPUs)      │  │
│   │  ┌─────┬─────┬───┐          ┌─────┬─────┬───┐      │  │
│   │  │GPU 0│GPU 1│...│          │GPU 0│GPU 1│...│      │  │
│   │  │ L1  │ L1  │   │          │ L1  │ L1  │   │      │  │
│   │  └──┬──┴──┬──┴───┘          └──┬──┴──┬──┴───┘      │  │
│   │     └──┬──┘                    └──┬──┘              │  │
│   │   Shared L2 (DRAM)          Shared L2 (DRAM)        │  │
│   │   Shared L3A (SSD)          Shared L3A (SSD)        │  │
│   └─────────────────────────────────────────────────────┘  │
│                           │                                │
│              ┌────────────┴────────────────┐               │
│              │  Global L3A (pooled SSDs)   │               │
│              │  OR local L3A (own SSD)     │               │
│              └─────────────────────────────┘               │
│                                                            │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│   │ PushDispatcher│  │ PrefillOracle│  │MetricsCollector│  │
│   │ PullDispatcher│  │ DecodeOracle │  │ (27 metrics)   │  │
│   └──────────────┘  └──────────────┘  └────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

## Worker Topology

```
Cluster
├── Worker 0 (1 server)
│   ├── GPU 0-7: each with own L1 (80 GB HBM)
│   ├── Shared L2 (1 TB host DRAM, shared by 8 GPUs)
│   └── Local L3A (8 TB NVMe SSD, shared by 8 GPUs)
├── Worker 1 (1 server)
│   └── (same structure)
└── Global L3A = pooled SSDs (N workers × 8 TB)
```

- **L1 (HBM)**: Per-GPU. 80 GB each. Hot KV cache.
- **L2 (DRAM)**: Per-worker. Shared by 8 GPUs on same host. No cross-GPU penalty.
- **L3A (SSD)**: Per-worker hardware. **Global mode** pools all workers' SSDs (accessible from any worker with 50ms remote latency). **Local mode** restricts to own SSD only.
- **Push dispatch**: cache-affinity-aware routing — prefers nodes with session's KV in L1/L2
- **Pull dispatch**: global queue with affinity-scored pulling

**Key finding**: Global L3A is essential for multi-worker coding deployments. At 20 min, session migration causes local L3A hit rate to drop to 68.6% while global maintains 99.8%. See [docs/heavy_coding_report.md](docs/heavy_coding_report.md).

## Storage Tier Model

| Tier | Medium | Capacity | Bandwidth | Block Size | Scope |
|------|--------|----------|-----------|------------|-------|
| L1   | HBM    | 80 GB    | 3 TB/s    | 5 KB       | Per-GPU |
| L2   | DRAM   | 1 TB     | 64 GB/s   | 32 MB      | Per-worker (shared by 8 GPUs) |
| L3A  | SSD    | 8 TB     | 7 GB/s    | 256 MB     | Per-worker; global pools all |

## Workload Profiles

Five built-in profiles. `profile_mix` weights are applied as multipliers on each profile's `arrival_rate_peak`.

| Profile | Session Rate | IAT | Session Length | Context Growth | Shared Prefix |
|---------|-------------|-----|----------------|----------------|---------------|
| **chat** | 100/s peak | 45s (exp) | 10 min | 100-500 tok/turn | 2,048 tokens |
| **coding** | 100/s peak | 360s (lognorm) | 1 hour | 2k-10k tok/turn | 20,000 tokens |
| **batch** | 100/s peak | 0.1s (exp) | 1s | none | 2,048 tokens |
| **agent** | 100/s peak | 15s (exp) | 30 min | 300-1.5k tok/turn | 2,048 tokens |
| **agentic_coding** | 100/s peak | 30s (exp) | 30 min | 5k-20k tok/turn | 30,000 tokens |

The **coding** and **agentic_coding** profiles are research-validated against real Claude Code, Cursor, and GitHub Copilot workloads (40-80k tokens, 80-95% cacheable).

## Metrics Collected

- **TTFT** by source (L1/L2/L3A hit, cold miss) — p50/p95/p99
- **Cache hit rate** breakdown by tier
- **Savings classification**: L1, L2 worthwhile, L3A worthwhile/break-even, cold miss
- **Tier occupancy** time-series
- **Eviction rates** (L1→L2 pressure, L1→L2 TTL, L2→L3A, cold)
- **Queue wait time** — time from queue entry to slot obtained
- **Prefill duration** — compute time per request
- **Slot utilization** — fraction of prefill slots busy
- **Multi-node dispatch** — affinity/non-affinity counts, cross-node transfers
- **Per-node metrics** — queue depth, L1/L2 occupancy, prefill count per GPU

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run all tests (27 tests)
pytest tests/ -v

# Generate diagnostic plots (default config)
python scripts/sanity_plots.py

# Generate plots with custom config
python scripts/sanity_plots.py --config configs/heavy_coding.json --outdir plots/heavy_coding

# Run 20-min heavy coding analysis
python scripts/heavy_coding_analysis.py

# Run parameter sweep
python scripts/sweep.py --config configs/default.json --output results/sweep.json
```

See [docs/architecture.md](docs/architecture.md) for how the simulation works end-to-end.
See [docs/user_manual.md](docs/user_manual.md) for full usage instructions.
See [docs/heavy_coding_report.md](docs/heavy_coding_report.md) for the heavy coding workload analysis.

## Project Structure

```
cache_sim/
├── configs/
│   ├── default.json             # v2 reference config (realistic hardware)
│   ├── heavy_coding.json        # 90% coding workload config
│   └── legacy_v1.json           # Original v1 config (small token counts)
├── benchmarks/
│   └── latency_tables/
│       └── prefill_70b_a100.json   # A100 prefill measurements (512-262k tokens)
├── sim/
│   ├── config.py               # Dataclass-based configuration
│   ├── engine.py               # Main DES loop, worker topology, event dispatch
│   ├── events.py               # Event types and request FSM
│   ├── cache.py                # CacheObject, TierStore, PrefixTrie
│   ├── eviction.py             # L1→L2→L3A eviction policies
│   ├── oracle.py               # Prefill/decode latency oracles
│   ├── service.py              # GPU slot model (prefill + decode pools)
│   ├── workload.py             # NHPP workload generator (profile_mix scaling)
│   ├── metrics.py              # MetricsCollector and report generation
│   ├── node.py                 # PrefillNode — per-GPU state with worker_id
│   ├── dispatch.py             # PushDispatcher and PullDispatcher
│   └── analysis.py             # Sustaining QPS binary search helper
├── scripts/
│   ├── sanity_plots.py         # 10 diagnostic plots (--config, --outdir)
│   ├── heavy_coding_analysis.py # 20-min heavy coding analysis (5 plots)
│   └── sweep.py                # Multi-process parameter sweep
├── tests/
│   ├── test_invariants.py      # 4 boundary-condition invariant tests
│   ├── test_kv_size.py         # 4 KV size math tests
│   ├── test_oracle.py          # 5 latency oracle tests
│   └── test_multinode.py       # 14 multi-node dispatch + topology tests
├── docs/
│   ├── architecture.md         # How the simulation works end-to-end
│   ├── user_manual.md          # Full usage instructions
│   ├── heavy_coding_report.md  # Heavy coding workload analysis report
│   └── example_report.md       # Example sanity-check report
├── plan_and_progress/          # Plans, progress reports, research, methodology
└── pyproject.toml
```

## License

Internal research tool.
