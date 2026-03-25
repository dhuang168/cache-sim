# AgentSim User Manual v2

## Overview

AgentSim is a discrete-event simulator for multi-tier (L1/L2/L3A) prompt KV cache systems targeting LLM inference workloads. It models request lifecycles from arrival through prefill, decode, and KV cache placement across GPU memory (HBM), host DRAM, and SSD storage.

**Two engines available:**
- `sim/` — Original cache-sim engine (v0.1-prototype, 105 tests)
- `agentsim/core/des/` — Ported engine with contracts, DESEvent emission, observer support (211 tests)

Both produce identical simulation results. The new engine adds:
- DESEvent emission for protocol observers (Anthropic, OpenAI)
- CacheKey-based identity, SavingsEvent classification
- Confidence labels on all oracle outputs
- Framework config adapters (vLLM, SGLang, LMCache)
- 349× performance improvement on high-load scenarios

---

## Installation

```bash
git clone https://github.com/dhuang168/cache-sim.git
cd cache-sim
pip install -e ".[dev]"
```

Requires Python 3.11+. Dependencies: `numpy`, `scipy`, `polars`, `marisa-trie`, `matplotlib`, `import-linter`.

---

## Quick Start

### Using the New Engine (recommended)

```python
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

config = SimConfig.from_json("configs/heavy_coding.json")
config.sim_duration_s = 300.0      # 5 min
config.warmup_s = 30.0
config.sim_start_time_s = 36000.0  # 10 AM
config.service.n_prefill_nodes = 32
config.service.n_gpus_per_worker = 8
config.service.dispatch_algorithm = "pull"

metrics = SimEngine(config).run()
report = metrics.report()
print(f"Hit rate: {report['cache_hit_rate']}")
print(f"TTFT p50: {report['ttft_ms']['L1_hit']['p50']}ms")
```

### With Protocol Observers

```python
from agentsim.core.des.engine import SimEngine
from agentsim.core.observation.events import EventStream
from agentsim.core.observation.anthropic import AnthropicProtocolObserver
from agentsim.core.observation.openai_chat import OpenAIProtocolObserver

stream_a = EventStream()
stream_o = EventStream()

metrics = SimEngine(config, observers=[
    AnthropicProtocolObserver(stream_a),
    OpenAIProtocolObserver(stream_o, baseline_ttft_us=50_000_000),
]).run()

# Anthropic miss taxonomy
summary = stream_a.miss_summary()
print(f"Hit rate: {summary['full_hit_rate']:.1%}")
print(f"Cold miss: {summary['cold_miss_rate']:.1%}")
print(f"Expiry miss: {summary['expiry_miss_rate']:.1%}")
print(f"Eviction miss: {summary['eviction_miss_rate']:.1%}")
```

### Using Framework Adapters

```python
from agentsim.core.des.config import SimConfig, ModelConfig
from agentsim.core.des.engine import SimEngine
from agentsim.integration.adapters.vllm import VLLMConfigAdapter
from agentsim.integration.adapters.lmcache import LMCacheConfigAdapter
from agentsim.integration.adapters.sglang import SGLangConfigAdapter

MODEL = ModelConfig(model_id="llama3-70b", n_layers=80, n_kv_heads=8, head_dim=128, bytes_per_element=2)
base = SimConfig.from_json("configs/heavy_coding.json")

# vLLM config
cfg = VLLMConfigAdapter().from_vllm_config(
    {"block_size": 16, "max_num_seqs": 256, "enable_prefix_caching": True},
    model=MODEL, base_config=base,
)

# LMCache config (chunk dedup + demand-pull)
cfg = LMCacheConfigAdapter().from_lmcache_config(
    {"chunk_size": 256, "local_cpu": True, "max_local_cpu_size": 64.0},
    model=MODEL, base_config=base,
)

# SGLang config (RadixAttention → tail-first eviction)
cfg = SGLangConfigAdapter().from_sglang_config(
    {"radix_attention": True, "max_running_requests": 256},
    model=MODEL, base_config=base,
)

metrics = SimEngine(cfg).run()
```

---

## Running Tests

```bash
# All 211 tests
pytest tests/ -v

# Original sim/ tests only (105)
pytest tests/test_invariants.py tests/test_kv_size.py tests/test_oracle.py \
       tests/test_blocks.py tests/test_sharing.py tests/test_multinode.py \
       tests/test_config_consistency.py tests/test_eviction_policy.py \
       tests/test_disaggregated.py tests/test_chunk_store.py tests/test_chunk_dedup.py -v

# New agentsim/ tests only
pytest tests/test_phase0_contracts.py tests/test_phase1_engine.py \
       tests/test_phase1_tolerance.py tests/test_phase2_observers.py \
       tests/test_phase3_chip_profiles.py tests/test_phase4_adapters.py \
       tests/test_phase05_baseline.py -v

# Import boundary enforcement
lint-imports
```

---

## Configuration Reference

### Available Configs

| Config | Workload | Notes |
|--------|----------|-------|
| `configs/default.json` | Mixed (30% chat, 20% coding, 20% batch, 15% agent, 15% agentic) | Standard reference |
| `configs/heavy_coding.json` | 90% coding (45% coding + 45% agentic_coding) | Cache pressure analysis |
| `configs/disaggregated.json` | Mixed + 3P:1D prefill-decode separation | Disaggregated mode |
| `configs/lmcache.json` | Mixed + chunk dedup + demand-pull | LMCache-style caching |
| `configs/custom_npu_local_l3a.json` | Coding on 32GB HBM + 256GB DDR NPU, local L3A | NPU analysis |
| `configs/custom_npu_global_l3a.json` | Same NPU, global L3A | NPU with shared SSD |

### Disaggregated Mode Fields (`service`)

| Field | Default | Description |
|-------|---------|-------------|
| `disaggregated` | `false` | Enable prefill-decode separation |
| `n_decode_nodes` | `0` | Dedicated decode GPUs (0 = colocated) |
| `kv_transfer_bandwidth_bytes_per_s` | 50 GB/s | RDMA transfer bandwidth |
| `kv_transfer_latency_floor_us` | 2000 | 2ms fixed transfer overhead |
| `prefill_latency_multiplier` | 1.0 | <1.0 = speedup from isolation |
| `decode_batch_fill_factor` | 0.7 | Effective decode batch utilization |

### Cache Mode Fields (`cache`)

| Field | Default | Options |
|-------|---------|---------|
| `block_size_tokens` | 0 | 0=legacy, 16=vLLM, 256=LMCache, 4096=page |
| `eviction_policy` | `"ttl"` | `"ttl"` (scheduled migration) or `"lru"` (reactive) |
| `deduplication` | `"per_session"` | `"per_session"` (monolithic) or `"chunk"` (LMCache-style) |
| `tier_migration` | `"ttl_push"` | `"ttl_push"` (scheduled demotion) or `"demand_pull"` (promote on hit) |
| `chunk_eviction` | `"lru"` | `"lru"` (LMCache) or `"tail_first"` (vLLM-style, preserves prefix) |

---

## Key Findings from Research

### Dispatch Algorithm is More Important Than L3A Mode

| Dispatch | Global L3A needed? | Why |
|----------|-------------------|-----|
| Push | Yes (critical at 4+ workers) | Sessions migrate → local cold misses → cascade |
| Pull | No (local sufficient) | Self-selection prevents migration → 99.8% hit |

### Disaggregated P/D: Consistent 14% TTFT Reduction

The 0.85 prefill multiplier (no decode interference) saves 14% TTFT across all cluster sizes. KV transfer overhead is <2% of TTFT at 50 GB/s RDMA.

### Chunk Dedup Not Suited for Heavy Coding

LMCache-style chunk dedup achieves 45-55% hit rate vs 99.9% for per-session objects on heavy coding workloads. The consecutive-chunk lookup breaks when any single chunk is evicted. vLLM tail-first eviction helps (76.8% vs 70.3%) but object mode remains superior for long-context coding.

### DDR (L2) Highly Valuable on NPU

On custom NPU with 32GB HBM + 256GB DDR: L2 hit rate = 74.2%, L2 saturation = 100%. DDR capacity is the binding constraint. On A100 with 80GB HBM: L2 hit rate = 0.0% (L1 is large enough to absorb most objects).

---

## Performance Notes

| Scenario | Wall Time |
|----------|-----------|
| 30s sim, 1 worker | ~0.1s |
| 5 min sim, 4 workers, peak=15 | ~8s |
| 5 min sim, 4 workers, peak=100 | ~8s (after perf fixes) |
| 20 min sim, 4 workers, peak=100 | ~60s |
| Full test suite (211 tests) | ~15 min (includes Scenario D at peak=100) |

**Performance gates:** Profile with cProfile at small scale before running long sims. The simulator achieves 349× speedup over the unoptimized baseline through O(1) session affinity lookup, capped pull dispatch scan, and heapq eviction.

---

## Reproducibility

All research scripts are in `scripts/`:
- `scripts/research_q[1-11].py` — 11 research questions with golden files
- `scripts/investigation_i[1-7].py` — 7 investigation scripts
- `scripts/disagg_lmcache_study.py` — 18-config comparison study
- `scripts/capture_baseline_metrics.py` — baseline metric capture

Golden reference files in `results/`:
- `results/golden/phase05_reference.json` — tolerance bands
- `results/phase[1-4]-baseline/` — per-phase golden files
- `results/v0.1-prototype-baseline/` — pre-migration baseline

Tag `v0.1-prototype` preserves the exact pre-migration state.
