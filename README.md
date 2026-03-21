# PCS Cache Simulator

A discrete-event simulator for multi-tier prompt cache systems. It models the full lifecycle of KV cache objects across a three-tier storage hierarchy (L1/L2/L3A) to evaluate caching strategies, eviction policies, and their impact on Time-To-First-Token (TTFT) for LLM inference workloads.

Supports **multi-node prefill dispatch** with per-node L1/L2 caches, shared or local L3A, and push/pull dispatch algorithms with cache-affinity awareness.

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

## Multi-Node Architecture

```
                    ┌──────────────────────┐
                    │   Central Dispatcher  │
                    │  (push or pull mode)  │
                    └──────┬───────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
        ┌──────────┐ ┌──────────┐    ┌──────────┐
        │  Node 0  │ │  Node 1  │    │  Node N  │
        │ L1 (HBM) │ │ L1 (HBM) │    │ L1 (HBM) │
        │ L2 (DRAM)│ │ L2 (DRAM)│    │ L2 (DRAM)│
        │ 32 slots │ │ 32 slots │    │ 32 slots │
        └────┬─────┘ └────┬─────┘    └────┬─────┘
             └─────────────┼───────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Shared L3A (or local)  │
              │  Shared Decode Pool     │
              └─────────────────────────┘
```

- Each node has its own L1 (HBM) and L2 (DRAM) cache stores
- L3A can be **shared** (global pool, all nodes access with remote latency) or **local** (per-node, `capacity/N`)
- **Push dispatch**: cache-affinity-aware routing — prefers nodes with session's KV in local cache
- **Pull dispatch**: global queue with affinity-scored pulling when slots free up
- Single-node mode (`n_prefill_nodes=1`) is backward-compatible with the original single-node behavior

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

Five built-in profiles model different inference patterns:

| Profile | Arrival Rate | IAT | Session Length | Context Growth | Shared Prefix |
|---------|-------------|-----|----------------|----------------|---------------|
| **chat** | 100 req/s peak | 45s (exp) | 10 min | 100-500 tok/turn | 2,048 tokens |
| **coding** | 50 req/s peak | 360s (lognorm) | 1 hour | 2k-10k tok/turn | 20,000 tokens |
| **batch** | 200 req/s peak | 0.1s (exp) | 1s | none | 2,048 tokens |
| **agent** | 30 req/s peak | 15s (exp) | 30 min | 300-1.5k tok/turn | 2,048 tokens |
| **agentic_coding** | 20 req/s peak | 30s (exp) | 30 min | 5k-20k tok/turn | 30,000 tokens |

The **coding** and **agentic_coding** profiles reflect real-world coding assistant workloads (Claude Code, Cursor, GitHub Copilot) where total prompt lengths reach 40-80k tokens with 80-95% cacheability. See `plan_and_progress/research_coding_workload_tokens.md` for the supporting research.

## Metrics Collected

- **TTFT** by source (L1 hit, L2 hit, L3A hit, cold miss) — p50/p95/p99
- **Cache hit rate** breakdown by tier
- **Savings classification**: L1 hit, L2 worthwhile, L3A worthwhile, L3A break-even, cold miss
- **Tier occupancy** time-series
- **Eviction rates** (L1→L2, L2→L3A, cold evictions)
- **Sharing factor** — total tokens served / unique tokens served
- **Memory pollution** (internal fragmentation per tier)
- **GPU queue depths** (prefill and decode)
- **Queue wait time** — time from entering pending queue to slot obtained (p50/p95/p99)
- **Multi-node dispatch stats** — affinity dispatches, non-affinity dispatches, cross-node transfers
- **Per-node metrics** — queue depth, L1/L2 occupancy, and prefill count per node

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
│   ├── metrics.py              # MetricsCollector and report generation
│   ├── node.py                 # PrefillNode — per-node L1/L2/slots state
│   ├── dispatch.py             # PushDispatcher and PullDispatcher
│   └── analysis.py             # Sustaining QPS binary search helper
├── scripts/
│   ├── sanity_plots.py         # 10 diagnostic plots (incl. multi-node)
│   └── sweep.py                # Multi-process parameter sweep
├── tests/
│   ├── test_invariants.py      # 4 boundary-condition invariant tests
│   ├── test_kv_size.py         # KV size math tests
│   ├── test_oracle.py          # Latency oracle tests
│   └── test_multinode.py       # 9 multi-node dispatch tests
├── plan_and_progress/          # Implementation plans and progress reports
└── pyproject.toml
```

## License

Internal research tool.
