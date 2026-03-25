# Agentic LLM Inference Simulator — Architecture

## Two-Mode Design

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        YOUR SIMULATOR REPO                              ║
║                                                                          ║
║  ┌──────────────────────────┐    ┌──────────────────────────────────┐   ║
║  │   MODE 1: ANALYTICAL     │    │   MODE 2: LIVE INTEGRATION       │   ║
║  │   (SimPy-based)          │    │   (Mock Hardware Plugin)         │   ║
║  │                          │    │                                  │   ║
║  │  Fast, no GPU needed.    │    │  Real vLLM/SGLang process.       │   ║
║  │  Sweep 1000s of configs. │    │  Tests actual scheduler,         │   ║
║  │  Pure Python, MIT.       │    │  batching, KV cache code paths.  │   ║
║  └──────────┬───────────────┘    └────────────┬─────────────────────┘   ║
║             │                                 │                          ║
║             │   shared hardware model         │                          ║
║             └─────────────┬───────────────────┘                          ║
║                           ▼                                              ║
║             ┌─────────────────────────┐                                  ║
║             │   HardwareModel         │                                  ║
║             │   (your memory model)   │                                  ║
║             │                         │                                  ║
║             │  - HBM tier: bw, cap    │                                  ║
║             │  - DDR tier: bw, cap    │                                  ║
║             │  - page_size: 4K        │                                  ║
║             │  - predict_latency()    │                                  ║
║             └─────────────────────────┘                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Mode 1: SimPy Simulator — Class Diagram

```
simpy.Environment
        │
        │ owns
        ▼
┌─────────────────────────────────────────────────────┐
│  AgenticSessionGenerator                            │
│  ─────────────────────────────────────────────────  │
│  + workload_type: "claude_code"                     │
│  + arrival_mode: "poisson" | "trace"                │
│  + trace_file: Optional[Path]                       │
│  ─────────────────────────────────────────────────  │
│  + generate(env) → Iterator[AgenticSession]         │
└──────────────────────┬──────────────────────────────┘
                       │ produces
                       ▼
┌──────────────────────────────────────────────────────┐
│  AgenticSession                                      │
│  ──────────────────────────────────────────────────  │
│  + session_id: str                                   │
│  + system_prompt_tokens: int   ← large, stable      │
│  + turns: List[Turn]                                 │
│  + sub_agents: List[AgenticSession]                  │
│                                                      │
│  A Turn is:                                          │
│    input_tokens: int  (file content + history)       │
│    output_tokens: int (code edit / response)         │
│    cached_prefix_len: int  ← key KV cache signal     │
└──────────────────────┬───────────────────────────────┘
                       │ fed into
                       ▼
┌─────────────────────────────────────────────────────┐
│  Scheduler  (abstract base)                         │
│  ─────────────────────────────────────────────────  │
│  + schedule(batch: List[Request]) → BatchPlan       │
│                                                     │
│  Implementations (each ~150 lines):                 │
│    OrcaScheduler                                    │
│    SarathiScheduler   ← chunked prefill             │
│    VLLMScheduler      ← continuous batching         │
│    SGLangScheduler    ← RadixAttention aware         │
└──────────────────────┬──────────────────────────────┘
                       │ uses
                       ▼
┌─────────────────────────────────────────────────────┐
│  KVCacheModel                                       │
│  ─────────────────────────────────────────────────  │
│  + page_size: int = 4096   ← your 4K requirement    │
│  + hbm_capacity: int                               │
│  + ddr_capacity: int                               │
│  + policy: "lru" | "prefix_aware"                  │
│  ─────────────────────────────────────────────────  │
│  + alloc(request) → List[Page]                     │
│  + evict() → freed_pages                           │
│  + hit_rate() → float                              │
│                                                     │
│  TieredKVCacheModel (subclass)                      │
│  + hbm_pages: PagePool                             │
│  + ddr_pages: PagePool                             │
│  + promote(page) / demote(page)                    │
└──────────────────────┬──────────────────────────────┘
                       │ calls
                       ▼
┌─────────────────────────────────────────────────────┐
│  HardwareModel  ← SHARED with Mode 2                │
│  ─────────────────────────────────────────────────  │
│  + hbm_bandwidth_gbps: float                       │
│  + ddr_bandwidth_gbps: float                       │
│  + compute_tops: float                             │
│  + page_size_tokens: int = 4096                    │
│  ─────────────────────────────────────────────────  │
│  + predict_prefill_latency(tokens, cache_hit) → ms  │
│  + predict_decode_latency(batch_size) → ms          │
│  + predict_kv_transfer(pages, src, dst) → ms        │
│                                                     │
│  Implementations:                                   │
│    GPUHardwareModel   (A100, H100 profiles)         │
│    NPUHardwareModel   ← your non-GPU target         │
│    CpuHardwareModel                                 │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  MetricsCollector                                   │
│  ─────────────────────────────────────────────────  │
│  + record_ttft(session_id, value_ms)               │
│  + record_tbt(session_id, value_ms)                │
│  + record_cache_hit(session_id, hit: bool)          │
│  ─────────────────────────────────────────────────  │
│  + summary() → DataFrame                           │
│  + plot_cdf(metric)                                │
└─────────────────────────────────────────────────────┘
```

---

## Mode 2: Mock vLLM Hardware Plugin — Structure

```
vllm-mock-npu/               ← your pip-installable package (MIT)
├── setup.py
│     entry_points={
│       "vllm.platform_plugins": [
│         "mock_npu = vllm_mock_npu:register"
│       ]
│     }
│
├── vllm_mock_npu/
│   ├── __init__.py
│   │     def register():
│   │         return "vllm_mock_npu.platform.MockNPUPlatform"
│   │
│   ├── platform.py          ← inherits vllm.platforms.Platform
│   │     MockNPUPlatform
│   │       check_and_update_config()  ← sets block_size=4096
│   │       get_attn_backend_cls()
│   │       get_device_communicator_cls()
│   │
│   ├── worker.py            ← inherits WorkerBase  [CORE]
│   │     MockNPUWorker
│   │       init_device()           → sets up HardwareModel config
│   │       load_model()            → loads weights to CPU RAM (real)
│   │       determine_available_memory() → returns configured HBM cap
│   │       initialize_cache()      → sets up TieredKVCacheModel
│   │       get_kv_cache_spec()     → reports 4K page size
│   │       execute_model(batch)    → ← THE KEY METHOD
│   │                                    1. compute logical latency
│   │                                       via HardwareModel
│   │                                    2. time.sleep(latency_s)
│   │                                    3. return real token output
│   │                                       (using CPU inference)
│   │                                    OR return mock token output
│   │                                       (stub, no real model)
│   │
│   ├── attention.py         ← inherits AttentionBackend
│   │     MockNPUAttention
│   │       Uses CPU torch ops (or stubs)
│   │       Reports 4K block size
│   │
│   ├── hardware_model.py    ← SHARED MODULE (same as Mode 1)
│   │     NPUHardwareModel
│   │
│   └── tiered_kv_cache.py   ← SHARED MODULE (same as Mode 1)
│         TieredKVCacheModel
│
└── tests/
    ├── test_vllm_integration.py   ← start vLLM, send requests,
    │                                 verify scheduling paths
    └── test_sglang_integration.py ← same for SGLang
```

---

## How the Two Modes Relate

```
                    HardwareModel (shared)
                   /                      \
                  /                        \
    Mode 1: SimPy                     Mode 2: Mock Plugin
    ─────────────                     ──────────────────
    Uses analytically                 Uses HardwareModel
    No real framework                 to inject sleep()
    Fast: 1000 configs/min            into real vLLM/SGLang
    Tests: scheduling policy,         Tests: framework scheduler
           KV eviction, cache hit     behavior, LMCache connector,
           rate vs. memory size       API compatibility, batching
                  │                          │
                  └──────────┬───────────────┘
                             │
                     Both produce same metrics:
                     TTFT, TBT, KV hit rate,
                     HBM/DDR utilization
                     → directly comparable
```

---

## Key Design Decisions

### 4K Page Size
Both modes use `page_size = 4096` tokens as a first-class config.
In Mode 2, `get_kv_cache_spec()` returns this to vLLM — overrides
vLLM's default 16-token blocks so the framework schedules around
your memory model, not CUDA's.

### execute_model() in Mock Worker — Two Sub-modes
```python
def execute_model(self, batch):
    if self.config.mode == "latency_injected":
        # Real model weights on CPU, real output, injected delay
        latency = self.hw_model.predict_prefill_latency(...)
        output = self.cpu_runner.forward(batch)   # real inference
        time.sleep(latency)
        return output

    elif self.config.mode == "stub":
        # No real weights needed — returns dummy tokens
        # Useful for scheduler/batching/KV cache path testing
        latency = self.hw_model.predict_prefill_latency(...)
        time.sleep(latency)
        return self._generate_stub_output(batch)
```

### The Shared HardwareModel Matters
Because both modes call the same `HardwareModel.predict_latency()`,
results are directly comparable. If the SimPy simulation says
LMCache should give 40% TTFT improvement, the mock plugin run
should show the same — if it doesn't, it reveals a framework
scheduling behavior you hadn't modeled.

---

## Build Order / Milestones

```
Week 1–2:  HardwareModel + TieredKVCacheModel (shared core)
           ← 4K pages, HBM/DDR bandwidth, latency functions

Week 3–4:  SimPy simulator (Mode 1)
           ← AgenticSession generator, Scheduler base,
             MetricsCollector, end-to-end test sweep

Week 5–6:  Mock vLLM Plugin (Mode 2 stub mode)
           ← Platform + Worker + Attention skeleton
           ← Verify vLLM schedules through your worker

Week 7–8:  LMCache connector + SGLang variant
           ← Add LMCache integration to Mode 2
           ← Port plugin to SGLang's backend interface

Week 9+:   Latency-injected mode + validation
           ← Real CPU inference in worker
           ← Cross-validate Mode 1 vs Mode 2 results
```
