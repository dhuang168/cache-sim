# PCS Cache Simulator

A discrete-event simulator for multi-tier prompt cache systems. It models the full lifecycle of KV cache objects across a three-tier storage hierarchy (L1/L2/L3A) to evaluate caching strategies, eviction policies, and their impact on Time-To-First-Token (TTFT) for LLM inference workloads.

## Architecture

```
                          ┌──────────────────────┐
                          │   WorkloadSynthesizer │
                          │  (NHPP + profiles)    │
                          └──────────┬───────────┘
                                     │ REQUEST_ARRIVAL
                                     ▼
┌────────────────────────────────────────────────────────────┐
│                        SimEngine                           │
│                                                            │
│   sim_clock_us ──► min-heap event loop                     │
│                                                            │
│   Event dispatch:                                          │
│     ARRIVAL → PREFILL_START → PREFILL_COMPLETE             │
│            → DECODE_START → DECODE_COMPLETE                │
│            → KV_WRITE → TTL_FIRE → TIER_EVICTION           │
│                                                            │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│   │ PrefixTrie  │  │ ServiceModel │  │ EvictionEngine   │ │
│   │ (per-session │  │ (prefill +   │  │ (L1→L2→L3A      │ │
│   │  + shared)   │  │  decode GPU  │  │  watermark +     │ │
│   │             │  │  slots)      │  │  TTL + LRU)      │ │
│   └─────────────┘  └──────────────┘  └──────────────────┘ │
│                                                            │
│   ┌──────────────────────────────────────────────────────┐ │
│   │              TierStore  ×  3                         │ │
│   │   L1 (HBM)          L2 (DRAM)         L3A (SSD)     │ │
│   │   80 GB / 3 TB/s    4 TB / 64 GB/s    20 TB / 7 GB/s│ │
│   │   5 KB blocks        32 MB blocks     256 MB blocks  │ │
│   └──────────────────────────────────────────────────────┘ │
│                                                            │
│   ┌────────────────┐  ┌──────────────┐                     │
│   │ PrefillOracle  │  │ DecodeOracle │                     │
│   │ (benchmark     │  │ (sqrt batch  │                     │
│   │  interpolation)│  │  degradation)│                     │
│   └────────────────┘  └──────────────┘                     │
│                                                            │
│   ┌──────────────────────────────────────────────────────┐ │
│   │              MetricsCollector                        │ │
│   │  TTFT histograms, tier occupancy, hit rates,         │ │
│   │  eviction counts, sharing factor, queue depths       │ │
│   └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Microsecond sim clock** | Avoids floating-point drift over long simulations |
| **Min-heap event queue** | O(log n) scheduling; natural for DES |
| **Request FSM** | `QUEUED → CACHE_LOOKUP → HIT/MISS → PREFILLING → DECODE_QUEUED → DECODING → KV_WRITE → COMPLETE` — catches illegal state transitions |
| **Benchmark-driven prefill oracle** | Piecewise-linear interpolation from real A100 measurements instead of synthetic formulas |
| **NHPP workload** | Non-homogeneous Poisson process with sinusoidal diurnal modulation and thinning algorithm |
| **Per-session + shared prefix tries** | Two-level trie lookup: shared system prefix (cross-session) and per-session context prefix |

## Storage Tier Model

| Tier | Medium | Default Capacity | Bandwidth | Block Size | Purpose |
|------|--------|-------------------|-----------|------------|---------|
| L1   | HBM    | 80 GB             | 3 TB/s    | 5 KB       | Hot KV cache, zero-copy prefill |
| L2   | DRAM   | 4 TB              | 64 GB/s   | 32 MB      | Warm KV cache, transfer cost |
| L3A  | SSD    | 20 TB             | 7 GB/s    | 256 MB     | Hibernated KV, high latency |

Objects flow downward (L1 → L2 → L3A) via TTL expiration and occupancy-based eviction, and are restored upward on cache hits.

## Workload Profiles

Four built-in profiles model different inference patterns:

| Profile | Arrival Rate | IAT | Session Length | Context Growth | Shared Prefix |
|---------|-------------|-----|----------------|----------------|---------------|
| **chat** | 100 req/s peak | 45s (exp) | 10 min | 100-500 tok/turn | 2048 tokens |
| **coding** | 50 req/s peak | 360s (lognorm) | 1 hour | 500-2000 tok/turn | 2048 tokens |
| **batch** | 200 req/s peak | 0.1s (exp) | 1s | none | 2048 tokens |
| **agent** | 30 req/s peak | 15s (exp) | 30 min | 300-1500 tok/turn | 2048 tokens |

## Metrics Collected

- **TTFT** by source (L1 hit, L2 hit, L3A hit, cold miss) — p50/p95/p99
- **Cache hit rate** breakdown by tier
- **Savings classification**: L1 hit, L2 worthwhile, L3A worthwhile, L3A break-even, cold miss
- **Tier occupancy** time-series
- **Eviction rates** (L1→L2, L2→L3A, cold evictions)
- **Sharing factor** — total tokens served / unique tokens served
- **Memory pollution** (internal fragmentation per tier)
- **GPU queue depths** (prefill and decode)

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run invariant tests
pytest tests/

# Generate sanity-check plots
python scripts/sanity_plots.py

# Run parameter sweep
python scripts/sweep.py --config configs/default.json --output results/sweep.json
```

See [docs/user_manual.md](docs/user_manual.md) for full usage instructions.

## Project Structure

```
cache_sim/
├── configs/
│   └── default.json            # Reference configuration
├── benchmarks/
│   └── latency_tables/
│       └── prefill_70b_a100.json   # Real A100 prefill measurements
├── sim/
│   ├── config.py               # Dataclass-based configuration
│   ├── engine.py               # Main DES loop and event dispatch
│   ├── events.py               # Event types and request FSM
│   ├── cache.py                # CacheObject, TierStore, PrefixTrie
│   ├── eviction.py             # L1→L2→L3A eviction policies
│   ├── oracle.py               # Prefill/decode latency oracles
│   ├── service.py              # GPU slot model (prefill + decode pools)
│   ├── workload.py             # NHPP workload generator
│   └── metrics.py              # MetricsCollector and report generation
├── scripts/
│   ├── sanity_plots.py         # 7 diagnostic plots
│   └── sweep.py                # Multi-process parameter sweep
├── tests/
│   ├── test_invariants.py      # 4 boundary-condition invariant tests
│   ├── test_kv_size.py         # KV size math tests
│   └── test_oracle.py          # Latency oracle tests
└── pyproject.toml
```

## License

Internal research tool.
