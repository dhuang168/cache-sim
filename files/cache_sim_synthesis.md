# cache-sim Prototype Analysis & AgentSim Integration Plan

---

## Part A: What I Learned from cache-sim

### A1. Architecture — Much More Complete Than Our Proposal

cache-sim is not a sketch — it's a working DES with 62 commits, 27 metrics,
14 multi-node tests, and real simulation results. Key architectural elements
that are absent or incomplete in our current AgentSim:

**Three-tier cache (L1/L2/L3A), not two**

| Tier | Medium | Capacity | Bandwidth | Block Size | Scope |
|------|--------|----------|-----------|------------|-------|
| L1   | HBM    | 80 GB    | 3 TB/s    | 5 KB       | Per-GPU |
| L2   | DRAM   | 1 TB     | 64 GB/s   | 32 MB      | Per-worker (8 GPUs share) |
| L3A  | SSD    | 8 TB     | 7 GB/s    | 256 MB     | Per-worker or global pool |

Our AgentSim current proposal only has HBM + DDR. The SSD tier (L3A) is
critical — it's the tier where session migration breaks local L3A deployments.
Missing it means we'd miss the single most important finding from this prototype.

**Worker topology: 8 GPUs per worker sharing L2 and L3A**

L2 is *shared DRAM across 8 GPUs on the same host* — not per-GPU. This means:
- Same-worker L2 hits have zero cross-GPU transfer penalty
- Cross-worker L2 hits don't exist — you either hit L1/L2 locally or fall to L3A
- Our AgentSim has no concept of workers, nodes, or GPU-to-worker mapping

**Dispatch layer: Push and Pull with affinity scoring**

cache-sim has both `PushDispatcher` (proactive affinity-aware routing) and
`PullDispatcher` (global queue with affinity scoring on pull). Affinity is
checked only against L1/L2 — once KV falls to L3A, session has no affinity
and gets freely routed. This is the root cause of session migration.
Our AgentSim has no dispatch model at all.

**GPU slot model: prefill and decode pools**

Each GPU has a prefill slot pool and decode pool. Slots are a first-class
resource — requests queue when slots are full, get dropped when queue is full.
Queue wait is tracked separately from prefill compute time. Our AgentSim
has no slot/queue model — we can't currently measure queue starvation or
slot utilization, which are critical metrics.

**Real prefill oracle vs. roofline approximation**

cache-sim uses piecewise-linear interpolation from *actual A100 benchmark data*:

| Tokens | Real Measured Prefill Latency |
|--------|-------------------------------|
| 10,000 | 2.2s |
| 30,000 | 14.9s |
| 50,000 | 37.0s |
| 65,000 | 54.4s |
| 100,000 | 120.7s |

Our roofline model is a mathematical approximation. For agentic coding at
50-100k token contexts (which is the real operating range), the roofline
will diverge substantially from measured behavior. The oracle approach
is calibrated per hardware and gives ~O(n²) accuracy that matches reality.

**NHPP with diurnal pattern, not simple Poisson**

Arrival rate uses a Non-Homogeneous Poisson Process with 24-hour sinusoidal
shape peaking at 9 AM. Our simple Poisson generator misses the time-of-day
load variation that drives steady-state cache behavior. Steady-state at 20 min
is very different from cold-start behavior — warmup period modeling is required.

**PrefixTrie for cache matching, not token-count arithmetic**

cache-sim uses a trie per session to match cached prefixes. Our current
code uses `min(cached_tokens, prefix_len)` which is too simple — it can't
distinguish whether the cached prefix actually matches the new request's
prefix, especially as context evolves across turns.

---

### A2. Key Empirical Findings — Must Be Reproduced in AgentSim

**Finding 1: Global L3A is essential for multi-worker coding deployments**

The single most important result. At 4 workers × 8 GPUs, 20 minutes:
- Global L3A hit rate: 99.8%
- Local L3A hit rate: 68.6%
- Global completes 84% more requests in the same time window
- Queue wait: global 2.7s mean vs local 50.8s mean (19× difference)

This is not a small optimization — it's a system architecture question.
AgentSim must be able to model global vs local L3A as a first-class toggle.

**Finding 2: Session migration is the root cause, not TTL or eviction**

At steady state, 97-99% of dispatches are non-affinity (no L1/L2 hit found).
This is because L1/L2 saturate within 5 minutes for a heavy coding workload,
forcing KV objects to L3A where affinity is lost. TTL sensitivity tests show
that changing L2 TTL from 10s to 300s has essentially zero impact on hit rate.
The bottleneck is architectural, not tunable.

Implication for AgentSim: our current ThinkTimeDistribution and TTL expiry
model are secondary. The primary driver of cache miss in a real multi-worker
deployment is session migration from tier exhaustion, not think-time TTL crossing.
Our Anthropic 5-min TTL expiry model is relevant only for single-worker scenarios.

**Finding 3: Recompute fraction is ~31% structurally, not a cache failure**

Even with 99.8% hit rate, ~31% of tokens per turn must always be recomputed.
This is because each turn introduces 8-15k new input tokens (the user message,
file diff, tool output) that were never in cache. Prefix stability ≠ 100%
even for a perfect cache — there's always a new-tokens component.

Implication: our session model's `cached_prefix_len` metric conflates two
different things — tokens that could be cached but weren't (true misses) vs
tokens that are new this turn and can never be cached (structural recompute).
We need to track these separately.

**Finding 4: KV object sizes for coding are massive**

70B FP16 model: coding session KV = ~8.2 GB, agentic_coding = ~15.6 GB.
A single 80k-token agentic coding context occupies a huge fraction of L1.
This means L1 (80 GB) can hold roughly 5-10 concurrent coding sessions max.
L2 saturation happens in ~5 minutes under moderate load.

Implication: our `KVCacheManager` using simple token counts misses the
massive KV object sizes that are the real L1 pressure driver.

**Finding 5: Savings are nonlinear with context size**

Cache hit on 95k-token context saves 88% of prefill time (13s hit vs 111s miss).
Cache hit on 30k-token context saves 70% (4.5s vs 14.9s).
The value of caching grows super-linearly with context size because prefill
scales O(n²). This means our metric of "hit rate" is insufficient — we need
"compute-seconds saved" which weights larger context hits more heavily.

**Finding 6: Infrastructure metrics mask the performance gap**

Slot utilization and L1/L2/L3A occupancy are identical between global and
local L3A modes. Without TTFT and throughput metrics, you'd think the systems
perform identically. This is a critical observability lesson — occupancy alone
is not sufficient.

---

### A3. Workload Profile Calibration — Already Validated

The five profiles are research-validated against real Claude Code, Cursor,
and GitHub Copilot workloads:

| Profile | IAT | Session Length | Context/Turn | System Prefix |
|---------|-----|----------------|--------------|---------------|
| coding | 360s lognorm | 1 hour | 2-10k | 20k tokens |
| agentic_coding | 30s exp | 30 min | 5-20k | 30k tokens |

These are the profiles we should adopt directly rather than re-deriving
our own. They are already calibrated and validated.

---

### A4. Code Quality and Methodology

The repo has 27 tests, sweep scripts, sanity plot scripts, and a
`simulation_development_guide.md` that encodes methodology for building
and validating simulators. The `CLAUDE.md` suggests Claude Code was used
actively as part of the development workflow. The `plan_and_progress/`
directory tracks the evolution of the simulator. This is a mature
research prototype, not a sketch.

The `int64 microsecond` sim clock (no float drift) is a production-quality
choice that we should adopt. Our current SimPy float-based time can
accumulate drift over long simulations with many events.

---

## Part B: How to Apply This to AgentSim

### B1. What to Port Directly (Don't Rewrite)

These are complete, validated, and directly usable:

1. **Workload profiles** — adopt coding and agentic_coding directly.
   Replace our TokenProfileDistribution with cache-sim's profile definitions.

2. **Prefill oracle** — port `oracle.py` and `benchmarks/latency_tables/`.
   Replace our roofline `predict_prefill()` with piecewise-linear interpolation.
   Add oracle tables for other chips (MI300X, Gaudi3, custom NPU).

3. **NHPP + diurnal generator** — port `workload.py`.
   Replace our `expovariate` arrival model.

4. **27 metrics collector** — port `metrics.py`.
   Our MetricsCollector is too thin. Adopt the full set including queue wait,
   slot utilization, per-node metrics, and savings classification.

5. **Sanity plots** — port `scripts/sanity_plots.py`.
   Ready-made diagnostic visualization for verifying simulator correctness.

6. **Global vs local L3A toggle** — port the `l3a_shared` config flag and
   the dispatch behavior difference. This is one config line but the single
   most important experimental variable.

### B2. What to Extend (Port + Add AgentSim Capabilities)

**Tier model: L1 + L2 + L3A → keep, add non-GPU variants**

```
cache-sim (current):           AgentSim extension:
L1: HBM 80GB  3TB/s            L1: configurable (HBM spec from ChipProfile)
L2: DRAM 1TB  64GB/s           L2: configurable (DDR or DRAM)
L3A: SSD 8TB  7GB/s            L3A: configurable (SSD, NVMe, remote)
                                + CXL tier (optional)
                                + block_size per tier from ChipProfile
```

**Worker topology: generalize to ChipProfile**

```
cache-sim: hardcoded 8 GPUs/worker, 80GB L1, 1TB L2, 8TB L3A
AgentSim:  n_gpus_per_worker from config
           L1 capacity = chip.hbm_capacity_gb
           L2 capacity = config.dram_per_worker_gb
           L3A capacity = config.ssd_per_worker_tb
           → supports H100, MI300X, custom NPU with same code
```

**Dispatch: add affinity-awareness for non-GPU targets**

cache-sim's push/pull dispatch is GPU-centric. For custom NPU targets
with different interconnect (400 Gbps vs 900 Gbps NVLink), the affinity
scoring and cross-worker transfer latency need to use ChipProfile values.

**Protocol event layer: add on top of existing engine**

cache-sim's engine fires internal events. AgentSim's protocol layer
(AnthropicEventMapper, OpenAIChatEventMapper) sits as an observer on
top of the engine events, translating them to SSE-protocol checkpoints.
This is additive — doesn't change cache-sim's core engine.

**Cache miss taxonomy: replace binary hit/miss with 4-type taxonomy**

cache-sim tracks: L1_hit, L2_hit, L3A_hit, cold_miss (4 types).
AgentSim adds: COLD, EXPIRY (Anthropic 5-min TTL), EVICTION, TRANSFER.
These map onto cache-sim's event types with additional metadata.

### B3. What to Add (New Capabilities)

**Sub-agent session spawning**

cache-sim has no notion of sub-agents. In Claude Code, a turn can spawn
lightweight sub-agents (Haiku) running concurrently. These create competing
demand for L1 slots on the same worker. Add to engine.py's session lifecycle.

**Non-GPU ChipProfile for oracle**

cache-sim's oracle is A100-specific. For custom NPU targets, we need:
- Either: measured latency tables for the target chip (preferred)
- Or: roofline model as a fallback when no measurements exist

The oracle interface should accept a chip_name parameter and look up
the appropriate table, falling back to roofline if not found.

**Anthropic 5-min TTL as a tier eviction policy**

cache-sim's TTL is L2→L3A migration. We need a separate Anthropic-TTL
eviction that removes objects from ALL tiers after 300s of session idle time,
regardless of memory pressure. This is a different eviction trigger.

---

## Part C: Revised Build Plan

### Updated Repository Structure

```
agentsim/
├── core/
│   ├── hardware_model.py    ← keep + extend with non-GPU chip profiles
│   ├── session_model.py     ← replace with cache-sim workload profiles (port)
│   └── events.py            ← keep + extend with miss taxonomy
│
├── sim/
│   ├── engine.py            ← NEW: port cache-sim engine.py as the core DES
│   │                            (int64 us clock, min-heap, worker topology)
│   ├── cache.py             ← NEW: port cache-sim cache.py (TierStore, PrefixTrie)
│   ├── eviction.py          ← NEW: port cache-sim eviction.py + add Anthropic TTL
│   ├── oracle.py            ← NEW: port cache-sim oracle.py + add chip table lookup
│   ├── workload.py          ← NEW: port cache-sim workload.py (NHPP, 5 profiles)
│   ├── dispatch.py          ← NEW: port cache-sim dispatch.py + ChipProfile BWs
│   ├── service.py           ← NEW: port cache-sim service.py (slot model)
│   ├── node.py              ← NEW: port cache-sim node.py (per-GPU state)
│   ├── metrics.py           ← NEW: port cache-sim metrics.py (27 metrics)
│   └── request_sim.py       ← KEEP: thin SimPy wrapper around engine (Goal 1)
│
├── protocol/                ← NEW (AgentSim unique - protocol event mapping)
│   ├── anthropic_adapter.py
│   ├── openai_chat_adapter.py
│   └── openai_responses_adapter.py
│
├── framework/               ← NEW (Goal 3 - real framework adapters)
│   ├── vllm_adapter.py
│   └── sglang_adapter.py
│
├── chips/
│   ├── profiles.py          ← keep + extend
│   └── oracle_tables/       ← NEW: per-chip prefill latency benchmark tables
│       ├── a100_70b.json    ← port from cache-sim
│       ├── h100_70b.json
│       ├── mi300x_70b.json
│       └── custom_npu.json
│
├── configs/                 ← port from cache-sim
│   ├── heavy_coding.json
│   ├── default.json
│   └── custom_npu.json      ← new
│
└── tests/                   ← port cache-sim's 27 tests + add new ones
```

### Updated Phase Plan

```
Phase 0 — Port (2 weeks) [NEW — replaces Week 1]
  Port cache-sim engine, cache, eviction, oracle, workload, dispatch,
  service, node, metrics as-is.
  Run cache-sim's 27 tests to verify the port is correct.
  Reproduce heavy_coding_report.md results exactly.
  → Deliverable: cache-sim running inside AgentSim repo, all tests green.

Phase 1 — Extend for non-GPU (Weeks 3-4) [was Week 1]
  Parameterize engine with ChipProfile (not hardcoded A100/H100).
  Add oracle table lookup per chip_name.
  Add custom_npu config with HBM+DDR+SSD tiers and 4K page size.
  Add global vs local L3A toggle to ChipProfile.
  → Deliverable: same heavy_coding results on parameterized NPU config.

Phase 2 — Protocol event layer (Weeks 5-6) [was Phase 2]
  Add AnthropicEventMapper on top of engine events.
  Add Anthropic 5-min TTL as a dedicated eviction policy.
  Add OpenAI TTFT-inferred cache detection.
  → Deliverable: Goal 2 cache miss detection working end-to-end.

Phase 3 — Sub-agents + agentic session model (Week 7)
  Add sub-agent spawning to engine session lifecycle.
  Add think_time distribution to session model.
  Validate against cache-sim's agentic_coding profile results.

Phase 4 — Framework adapters (Weeks 8-10) [was Phase 3]
  VLLMAdapter, SGLangAdapter using real scheduler in-process.
  Non-GPU oracle table for custom NPU.
  → Deliverable: Goal 3 interop validation.
```

---

## Part D: The Three Critical Changes from cache-sim Learnings

These three changes have the largest impact on correctness and should happen
in Phase 0/1 before any new development:

**1. Replace roofline with oracle**

Our `predict_prefill()` using roofline math will be wrong at 50-100k tokens
(the real operating range). The O(n²) attention term dominates, and the
actual latency is 2-5× higher than what our bandwidth model predicts.
Port the oracle with benchmark tables immediately.

**2. Add L3A tier and global/local toggle**

Without L3A, we cannot study the most important finding from cache-sim:
that multi-worker deployments live or die on global L3A accessibility.
This is a fundamental omission that makes our simulator blind to the
dominant failure mode in production agentic coding deployments.

**3. Replace per-session token-count KV model with TierStore + PrefixTrie**

Our current KVCacheManager tracks tokens, not bytes and not KV object sizes.
For a 70B model, a 50k-token coding session KV is ~15 GB — it barely fits
in one GPU's L1. Tracking tokens without KV byte sizes gives completely
wrong eviction timing and tier occupancy numbers. The TierStore with actual
KV byte size tracking is essential for correct simulation.
