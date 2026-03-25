# AgentSim — Master Document Index
# Complete list of all specs, plans, and code artifacts from this design thread
# Version 1.0

---

## How to Read This Index

Documents are grouped by type and status. For each document, the status
tells you exactly what role it plays:

  ✅ CURRENT    — authoritative, use this
  ⚠️  SUPERSEDED — replaced, kept for historical reference only
  🔧 CODE       — implementation artifact, status reflects currency

---

## 1. Execution Plans & Architecture

### ✅ agentsim_final_plan.md — PRIMARY PLAN (v1.1)
The authoritative execution plan. Incorporates all reviewer feedback
across three review rounds plus dev team repo management recommendations.
Supersedes all earlier architecture documents.

Contains:
  - 10 guiding principles (non-negotiable)
  - Mental model: "replace the engine of a plane mid-flight"
  - Three-layer architecture (Core DES / Observation / Integration)
  - **Pre-Phase 0** (1 day): v0.1-prototype tag, migration branch, import lint
  - Phase 0:   Architecture contracts (1 week)
  - Phase 0.5: Validation contract — canonical benchmarks + tolerance bands (3 days)
  - Phase 1:   Core DES layer — port cache-sim, byte-based accounting (3 weeks)
  - Phase 2:   Observation layer — protocol event mappers, miss taxonomy (2 weeks)
  - Phase 3:   Non-GPU parameterization — ChipProfile, confidence labels (2 weeks)
  - Phase 4:   Framework adapters — vLLM/SGLang schema compat (2 weeks)
  - SimPy sweep tool constraints
  - Full repository structure with deprecated file lifecycle
  - Old code lifecycle: Deprecate → Disconnect → Delete

Key decisions encoded (v1.1 additions):
  - Stay in existing repo — structured evolution, not rewrite
  - feature/des-core-swap migration branch (merge only after Phase 1 gate)
  - v0.1-prototype tag as permanent fallback baseline
  - Import enforcement via .importlinter + CI (not code review discipline)
  - core/des/README.md points to contracts.py — no duplicate definitions
  - _deprecated/ folder for disconnect phase before deletion
  - Branch merge rule: main stays green throughout migration

---

### ⚠️ agentsim_design.md — SUPERSEDED
Original "locked design document" from before cache-sim analysis.
Describes the SimPy request-level engine as primary, uses a 2-tier
HBM+DDR model, and has a Phase 1–4 plan that has been fully replaced.

What is still valid: the Goal 1/2/3 framing, the protocol event schema.
What is replaced: everything else.

---

### ⚠️ simulator_architecture.md — SUPERSEDED
First architecture sketch. Proposed a mock vLLM hardware plugin as the
primary approach. Replaced by simulator_architecture_v2.md and then
by the final plan.

---

### ⚠️ simulator_architecture_v2.md — SUPERSEDED
Second architecture sketch. Correctly identified the "framework drives
sessions" pattern and request-level granularity. Directionally right,
but predates cache-sim analysis and the three-layer architecture.
Superseded by agentsim_final_plan.md.

---

## 2. Research & Analysis

### ✅ cache_sim_synthesis.md — REFERENCE
Analysis of the cache-sim prototype repo (github.com/dhuang168/cache-sim).
Summarizes what was learned and how it changes the plan.

Key findings documented:
  - Three-tier L1/L2/L3A model (not two-tier HBM+DDR)
  - Worker topology: 8 GPUs sharing L2/L3A on same host
  - Global vs local L3A: 99.8% vs 68.6% hit rate — the central finding
  - Session migration as root cause (not TTL, not eviction policy)
  - Prefill oracle from measured data (not roofline) required at 50-100k tokens
  - Five validated workload profiles (coding, agentic_coding, chat, batch, agent)
  - Byte-based KV object sizing: 70B model coding session = ~8.2 GB

Part A: What the prototype taught us
Part B: How to apply it to AgentSim
Part C: Revised build plan (superseded by agentsim_final_plan.md)
Part D: Three critical changes

---

### ✅ gap_analysis.md — REFERENCE
Systematic analysis of every artifact produced before the final plan,
identifying specific inconsistencies and required changes.

Documents:
  - hardware_model.py: 3 violations, HIGH severity → split + rewrite
  - request_sim.py: 2 violations, HIGH severity → move + constrain
  - session_model.py: 3 violations, MEDIUM → demote + annotate
  - events.py: 1 violation, LOW → move + re-position
  - agentsim_design.md: superseded
  - simulator_architecture*.md: superseded

---

## 3. Code Artifacts

### ✅ core/contracts.py — PRIMARY CODE ARTIFACT (Phase 0)
The single source of truth for all layer boundaries and interface
contracts. Incorporates all feedback from three review rounds.

Defines:
  - ConfidenceLabel (CALIBRATED / SEMI_CALIBRATED / ANALYTICAL_ONLY)
  - CacheKey — exact prefix identity for cache lookups
  - SavingsEvent — HIT_L1 / HIT_L2_WIN / HIT_L3_WIN / HIT_L3_BREAK_EVEN
                   / MISS_RECOMPUTE with classify() method
  - TierSpec — byte-based tier specification
  - CacheObject — byte-based, token_count as metadata only
  - TransferRecord — full accounting of every KV movement
  - RequestResult — prefill and decode separated in all fields
  - CacheOracleBase ABC
  - DispatcherBase ABC
  - EvictionPolicyBase ABC
  - DESEventKind enum (all event types)
  - DESEvent — canonical event emitted by Core DES
  - ObserverBase ABC — with time-coupling constraints + randomized-order
    CI test requirement documented
  - SweepEstimator ABC — explicit list of what SimPy is prohibited from doing
  - ReportMetadata — required on every output, confidence in headline

This file is the Phase 0 deliverable. No code should be ported until
these interfaces are reviewed and frozen by the team.

---

### ✅ integration/chips/profiles.py — CURRENT
Replaces the old hardware_model.py ChipProfile and ModelConfig.
Byte-based throughout. page_size_tokens removed.

Contains:
  - TierSpec (byte-based) with helper constructors
  - ChipProfile with tiers tuple, confidence label, oracle_table path
  - ModelConfig with bytes_per_token_kv and kv_bytes_for_tokens()
    (unchanged from original — was already correct)
  - block_size_bytes_for_tokens() helper for legacy token-based interfaces
  - CHIP_PROFILES dict: H100, A100, MI300X, Gaudi3, custom_npu_hbm_ddr
    (all with correct confidence labels and oracle_table paths)
  - Note on "4K page size" translation to bytes for custom NPU

---

### ✅ core/des/oracle.py — CURRENT
Replaces the HardwareModel roofline from hardware_model.py.
Piecewise oracle is primary. Roofline is ANALYTICAL_ONLY fallback.

Contains:
  - CacheOracleBase ABC
  - PiecewiseOracle — piecewise-linear interpolation from JSON tables
    with confidence label detection from metadata
  - RooflineOracle — analytical fallback, always ANALYTICAL_ONLY
  - OracleFactory — selects correct oracle per chip, falls back gracefully

Key: every call returns (latency_us, ConfidenceLabel). The roofline
is never used silently — it always declares itself as ANALYTICAL_ONLY.

---

### ✅ sweep/request_sweep.py — CURRENT
Replaces sim/request_sim.py. SimPy demoted to sweep tool.
All SimPy cache state tracking removed.

Contains:
  - SweepEstimator ABC (mirrors contracts.py)
  - SimpleRooflineSweepEstimator — stateless, no tier tracking
  - SweepCacheEstimate — probabilistic only, not authoritative
  - SweepMetrics — all output labeled "sweep-estimate" + ANALYTICAL_ONLY
  - RequestSweep — SimPy runner with explicit prohibition docstrings

Prohibited patterns documented in code, not just documentation.

---

### ⚠️ core/hardware_model.py — SUPERSEDED
Has deprecation header added. Points to replacements.
Split into integration/chips/profiles.py + core/des/oracle.py.
Three violations resolved: page_size_tokens removed, roofline
demoted, latency in ms changed to us.

Retained for historical reference.

---

### ⚠️ core/session_model.py — DEMOTED (sweep only)
Has "DEMOTED — sweep tool only" header added.
Will be replaced in Phase 1 by core/des/workload.py ported from cache-sim.

What's wrong: simple Poisson (not NHPP), uncalibrated token profiles.
What's right: AgenticSession/Turn dataclass structure, ThinkTimeDistribution.

---

### ⚠️ core/events.py — PENDING MOVE (usable until Phase 2)
Has positioning note added. Content is correct.
Final location: core/observation/events.py.

One change needed in Phase 2: AnthropicEventMapper and
OpenAIChatEventMapper must implement ObserverBase.on_event(DESEvent)
instead of receiving explicit method calls.

---

### ⚠️ sim/request_sim.py — SUPERSEDED
Replaced by sweep/request_sweep.py.
Was positioning SimPy as co-equal engine — now prohibited.
KVCacheManager removed (violated "SimPy cannot own cache state" rule).

---

## 4. Document Lineage / Decision Trail

```
Conversation start
│
├─ simulator_architecture.md       ← first sketch (mock vLLM plugin)
├─ simulator_architecture_v2.md    ← second sketch (session-driven, better)
│
├─ agentsim_design.md              ← "locked" design (pre-cache-sim)
│   ├─ core/hardware_model.py      ← roofline oracle (now superseded)
│   ├─ core/session_model.py       ← session generator (now demoted)
│   ├─ core/events.py              ← protocol events (content OK, location wrong)
│   └─ sim/request_sim.py          ← SimPy engine (now demoted to sweep)
│
├─ cache_sim_synthesis.md          ← analysis of cache-sim prototype
│   └─ gap_analysis.md             ← what needs to change
│
├─ agentsim_final_plan.md          ← CURRENT AUTHORITATIVE PLAN (v1.0)
│   (review round 1: "port-heavy, not selective enough")
│   (review round 2: "core swap is right, add validation contract")
│   (review round 3: "approved — add 6 tightening items")
│
└─ Code updates (consistency pass)
    ├─ integration/chips/profiles.py   ← replaces hardware_model.py
    ├─ core/des/oracle.py              ← replaces HardwareModel roofline
    ├─ sweep/request_sweep.py          ← replaces sim/request_sim.py
    └─ core/contracts.py              ← Phase 0 interface contracts (new)
```

---

## 5. What Is Not Yet Built (Phase 1–4 deliverables)

These are defined in agentsim_final_plan.md but not yet implemented:

| File | Phase | Description |
|------|-------|-------------|
| `core/des/engine.py` | 1 | Main DES loop — port from cache-sim |
| `core/des/cache.py` | 1 | TierStore, PrefixTrie — byte-based |
| `core/des/eviction.py` | 1 | LRU + pressure eviction policies |
| `core/des/workload.py` | 1 | NHPP + 5 validated workload profiles |
| `core/des/dispatch.py` | 1 | Push/pull dispatcher with affinity |
| `core/des/service.py` | 1 | GPU slot model (prefill + decode pools) |
| `core/des/node.py` | 1 | Per-GPU state with worker_id |
| `core/des/metrics.py` | 1 | 27 metrics collector |
| `core/observation/events.py` | 2 | Moved + refactored from core/events.py |
| `core/observation/anthropic.py` | 2 | Anthropic SSE miss taxonomy |
| `core/observation/openai_chat.py` | 2 | OpenAI Chat inferred detection |
| `core/observation/openai_resp.py` | 2 | OpenAI Responses API |
| `integration/adapters/vllm.py` | 4 | VLLMConfigAdapter |
| `integration/adapters/sglang.py` | 4 | SGLangConfigAdapter |
| `integration/adapters/lmcache.py` | 4 | LMCacheConfigAdapter |
| `benchmarks/oracle_tables/a100_llama3_70b.json` | 1 | Calibrated latency table |
| `configs/heavy_coding.json` | 1 | Port from cache-sim |
| `scripts/reproduce_baseline.py` | 1 | Phase 1 gate validation script |
| `tests/test_phase0_contracts.py` | 0 | Interface enforcement tests |
| `tests/test_phase1_baseline.py` | 1 | Tolerance-band reproduction tests |
| `tests/test_phase2_observers.py` | 2 | Observer read-only enforcement tests |
