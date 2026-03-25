# AgentSim — Final Execution Plan
# Version 1.1 | Approved Architecture Migration Plan

---

## Guiding Principles (Non-Negotiable)

These are not preferences. Every phase decision flows from these.

1. **Core swap, not file port.** cache-sim's DES becomes the execution engine.
   The current prototype's protocol/event layer is preserved and re-homed.
   No two engines coexist as co-equals.

2. **Byte-based accounting is the Core DES API contract.**
   Tokens are metadata. The accounting unit is bytes everywhere:
   - All tier capacities: bytes
   - All cache objects: report bytes
   - All transfers: bytes moved + latency source
   - Tokens are stored as metadata on cache objects, never as capacity units

3. **DES events are canonical. Observation is strictly downstream.**
   The Core DES emits events. Protocol adapters interpret them.
   Protocol adapters never write back to simulation state.
   One-way dependency, enforced at the interface level.

4. **SimPy is a sweep tool, not an engine.**
   SimPy may call simplified estimators. It shares config objects
   and workload definitions with the Core DES. It never owns cache
   state, eviction logic, or tier accounting independently.

5. **Every hardware target carries a confidence label.**
   Calibrated | Semi-calibrated | Analytical-only.
   Labels appear in all output reports, not just documentation.

6. **Validation gates are tolerance-based, not exact.**
   "Reproduce" means: same scenarios, headline metrics within agreed band,
   qualitative rankings preserved. Not bit-for-bit identical numbers.

7. **Cache identity is exact prefix match, not token-count proximity.**
   A cache hit requires an exact prefix hash match. Partial hits are
   first-K-tokens match with the remainder recomputed. Cross-worker reuse
   requires compatible memory layout plus explicit transfer. Defined once
   in `CacheKey`, enforced everywhere — never inferred ad-hoc.

8. **Prefill and decode are separately tracked in all metrics.**
   Cache affects only prefill. Decode dominates long outputs.
   Blending them obscures optimization signals. Every metrics collector
   must report `prefill_latency_us` and `decode_latency_us` independently,
   along with separate queue wait for each phase.

9. **Observers must not assume event ordering beyond timestamp.**
   Observers may not infer future events, assume delivery order beyond
   `sim_time_us`, or maintain hidden state that affects interpretation of
   other observers. This prevents subtle bugs in protocol inference layers.

10. **Structured evolution inside the existing repo — not a rewrite, not casual mixing.**
    This is a re-architecture of the same system, not a pivot to a different product.
    A new repo would lose commit history, existing protocol/event work, and design
    context. Casual in-place development would recreate the "two worldviews" problem.
    The right mechanic: a dedicated migration branch, a prototype tag as fallback,
    `core/des/` as a hard internal boundary, and import enforcement via lint to
    prevent old modules from re-entering the execution path.

---

## Mental Model

> "Replace the engine of a plane mid-flight, without losing the cockpit."
>
> - **Engine** = Core DES layer (`core/des/`) — being replaced
> - **Cockpit** = Observation / protocol layer — preserved and re-homed
> - **Instrumentation** = Metrics / reporting — adapted, not replaced

This clarifies scope: swap the one part that was wrong (simulation engine)
while keeping the parts that were right (event schema, protocol mappers,
goal framing). A new repo is not warranted — this is engine replacement,
not product replacement.

---

## Three-Layer Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 3: INTEGRATION LAYER                                      │
│  Chip profiles, framework adapters, experiment configs,          │
│  sweep scripts, plotting, reporting, confidence labels           │
│  ─────────────────────────────────────────────────────────────   │
│  LAYER 2: OBSERVATION LAYER            [downstream only]         │
│  Checkpoint event schema, miss taxonomy (COLD/EXPIRY/EVICTION/   │
│  TRANSFER), Anthropic/OpenAI protocol mappers, TTFT/streaming    │
│  metrics, 5-min TTL tracker                                      │
│  ─────────────────────────────────────────────────────────────   │
│  LAYER 1: CORE DES LAYER                                         │
│  Workers, GPUs, tiers (L1/L2/L3A), dispatch (push/pull),        │
│  cache objects (byte-based), eviction, oracle, workload (NHPP),  │
│  slot model, metrics collector (27 metrics)                      │
└──────────────────────────────────────────────────────────────────┘
```

Layer dependencies flow **downward only**:
- Layer 3 calls Layer 2 and Layer 1 APIs
- Layer 2 consumes Layer 1 events, never modifies Layer 1 state
- Layer 1 has zero knowledge of Layer 2 or Layer 3

---

## Pre-Phase 0 — Repo Setup (1 day)
### Goal: Establish safe working conditions before any code changes

These three steps take one day and must happen before Phase 0 starts.
They are not optional — they define the fallback baseline and
the enforcement boundary for everything that follows.

### Step 1: Tag the current prototype

```bash
git tag v0.1-prototype
git push origin v0.1-prototype
```

This makes the current prototype permanently reproducible. It is the
comparison baseline for Phase 1 validation. If anything goes wrong
during the migration, this tag is the clean fallback.

Capture baseline outputs before tagging:
```bash
python scripts/sanity_plots.py          # current prototype output
python scripts/heavy_coding_analysis.py # reference numbers for Phase 0.5
```
Commit these outputs as `results/v0.1-prototype-baseline/`.

### Step 2: Create the migration branch

```bash
git checkout -b feature/des-core-swap
```

**Rules for this branch:**
- All Phase 0–1 work happens here, not on `main`
- Allowed to break things internally
- `main` stays green throughout the migration
- Branch merges to `main` only after the Phase 1 gate passes
  (all three benchmark scenarios within tolerance bands)

### Step 3: Set up import enforcement

Add an `import-linter` configuration to block imports from deprecated
modules before any new code is written. This enforces the layer boundary
structurally — not through code review discipline, which fails under
deadline pressure.

```ini
# .importlinter
[importlinter]
root_package = agentsim

[importlinter:contract:no-old-engine]
name = New code must not import from deprecated modules
type = forbidden
source_modules =
    agentsim.core.des
    agentsim.core.observation
    agentsim.integration
    agentsim.sweep
forbidden_modules =
    agentsim.core.hardware_model
    agentsim.core.session_model
    agentsim.sim.request_sim
```

Add to CI:
```yaml
# .github/workflows/ci.yml
- name: Check import boundaries
  run: lint-imports
```

Add to Phase 0 gate: import enforcement active in CI before porting begins.

### Old Code Lifecycle: Deprecate → Disconnect → Delete

Old modules follow a three-step lifecycle. **Do not delete early.**
Early deletion creates risk. Late deletion creates clarity.

| Step | When | Action |
|------|------|--------|
| Deprecate | Now (pre-Phase 0) | Add `# DEPRECATED: replaced by core/des/` header — already done on hardware_model.py, session_model.py, sim/request_sim.py |
| Disconnect | Phase 1 complete | Remove from all execution paths, scripts, imports |
| Delete | Phase 2 or 3 | Remove files after observation layer is fully migrated |

---

## Phase 0 — Architecture Contracts (1 week)
### Goal: Define interfaces before any porting begins

No code is ported in this phase. Interfaces are written, reviewed, and
frozen. This prevents the "two worldviews" problem.

### Deliverables

**0a. Core DES Layer API (Python ABCs + dataclasses)**

```python
# core/contracts.py

@dataclass(frozen=True)
class CacheObject:
    object_id:     str          # session_id + turn_id
    size_bytes:    int          # KV cache size in bytes (NOT tokens)
    token_count:   int          # metadata only
    created_at_us: int          # sim clock at creation
    tier:          Tier         # current resident tier
    worker_id:     int
    gpu_id:        int

@dataclass(frozen=True)
class TierSpec:
    name:           str          # "L1" | "L2" | "L3A"
    capacity_bytes: int          # total capacity
    bandwidth_bps:  int          # read bandwidth
    block_size_bytes: int        # minimum transfer unit
    scope:          str          # "per_gpu" | "per_worker" | "global"
    latency_floor_us: int        # minimum access latency

@dataclass
class TransferRecord:
    object_id:      str
    src_tier:       str
    dst_tier:       str
    bytes_moved:    int          # actual bytes transferred
    latency_us:     int          # transfer latency
    latency_source: str          # "bandwidth" | "floor" | "remote_penalty"

class CacheOracleBase(ABC):
    @abstractmethod
    def prefill_latency_us(
        self,
        uncached_tokens: int,
        chip_name:       str,
        model_name:      str,
    ) -> tuple[int, str]:        # (latency_us, confidence_label)
        ...

class DispatcherBase(ABC):
    @abstractmethod
    def select_node(
        self,
        request:    Request,
        cluster:    ClusterState,
    ) -> tuple[int, int]:        # (worker_id, gpu_id)
        ...

class EvictionPolicyBase(ABC):
    @abstractmethod
    def select_eviction_candidates(
        self,
        tier:        TierStore,
        needed_bytes: int,
    ) -> list[CacheObject]:
        ...

@dataclass(frozen=True)
class CacheKey:
    """
    Exact cache identity. A hit requires ALL fields to match.
    Partial hit = first K tokens match; defined by prefix_hash of
    the longest matching prefix, not the full sequence.
    Cross-worker reuse additionally requires transfer to be modeled.
    """
    model_id:      str   # model name + version
    tokenizer_id:  str   # tokenizer variant (affects token boundaries)
    prefix_hash:   str   # hash of the exact token prefix sequence

class SavingsEvent(str, Enum):
    """
    Core output metric for every request.
    This is the primary decision boundary the simulator is built to study.
    L3A usefulness is impossible to interpret without this classification.
    """
    HIT_L1           = "hit_l1"           # served from HBM, no transfer
    HIT_L2_WIN       = "hit_l2_win"       # DRAM hit, transfer cost < recompute
    HIT_L3_WIN       = "hit_l3_win"       # SSD hit, transfer cost < recompute
    HIT_L3_BREAK_EVEN = "hit_l3_break_even"# SSD hit, marginal benefit
    MISS_RECOMPUTE   = "miss_recompute"   # full recompute — cold/expiry/eviction
```

**0b. Observation Layer API**

```python
# observation/contracts.py

@dataclass
class DESEvent:
    """Emitted by Core DES. Observation layer reads these. Never writes back."""
    event_id:    int
    sim_time_us: int
    kind:        DESEventKind    # REQUEST_ARRIVAL, PREFILL_START,
                                  # CACHE_LOOKUP, TIER_TRANSFER,
                                  # EVICTION, PREFILL_COMPLETE,
                                  # DECODE_COMPLETE, REQUEST_DROP
    payload:     dict            # event-specific fields

class ObserverBase(ABC):
    """Observation layer components implement this."""
    @abstractmethod
    def on_event(self, event: DESEvent) -> None:
        """Read-only. Must not modify simulation state."""
        ...
```

**0c. SimPy Sweep Tool Contract**

```python
# sweep/contracts.py

class SweepEstimator(ABC):
    """
    What SimPy is allowed to call.
    These are simplified estimators — NOT the authoritative DES.
    Results are labeled 'sweep-estimate' in all outputs.

    SimPy MUST NOT:
    - instantiate TierStore directly
    - track eviction state
    - call EvictionPolicyBase directly
    - compute tier occupancy
    """
    @abstractmethod
    def estimate_turn_latency_ms(
        self,
        input_tokens:    int,
        output_tokens:   int,
        cache_hit:       bool,
        cache_tier:      str,
        hw_profile:      ChipProfile,
    ) -> tuple[float, float]:    # (ttft_ms, total_ms)
        ...
```

**0d. Confidence Label Contract**

```python
class ConfidenceLabel(str, Enum):
    CALIBRATED       = "calibrated"       # measured oracle tables exist
    SEMI_CALIBRATED  = "semi-calibrated"  # partial tables + interpolation
    ANALYTICAL_ONLY  = "analytical-only"  # roofline math, no measurements

# Every report output must include:
@dataclass
class ReportMetadata:
    chip_name:        str
    model_name:       str
    confidence:       ConfidenceLabel
    oracle_source:    str          # e.g., "benchmarks/a100_70b.json"
    generated_at:     str
    agentsim_version: str
```

### Gate: Phase 0 Complete
- [ ] All four contracts reviewed and approved by team
- [ ] No ambiguity about what Layer 1 vs Layer 2 owns
- [ ] SimPy constraint documented and understood by all contributors
- [ ] `core/des/README.md` exists with two sentences pointing to `contracts.py`
  as the authoritative source — not a second copy of interface definitions
- [ ] Import enforcement active in CI (`lint-imports` passes on `feature/des-core-swap`)
- [ ] `v0.1-prototype` tag pushed and baseline outputs committed to
  `results/v0.1-prototype-baseline/`

---

## Phase 0.5 — Validation Contract (3 days)
### Goal: Define success before porting begins

This is the small but high-value phase the reviewer added.
Without this, the team argues about success after the port is underway.

### Canonical Benchmark Scenarios

Three scenarios, each with defined acceptance tolerances:

**Scenario A: Single-worker heavy coding (baseline correctness)**
```
Config:    heavy_coding.json, 1 worker × 8 GPUs, 20 min sim
Reference: cache-sim heavy_coding_report.md, single-worker section
Metrics:
  - Overall cache hit rate:    target 99.5%–100%   (reference: 99.8%)
  - TTFT p50:                  within ±20%          (reference: ~8.4s)
  - TTFT p95:                  within ±25%          (reference: ~45.4s)
  - Slot utilization:          target ~100%         (reference: 100%)
  - L1/L2/L3A occupancy:       qualitative match    (reference: L2=100%, L3A=100%)
```

**Scenario B: Multi-worker global vs local L3A (the critical finding)**
```
Config:    heavy_coding.json, 4 workers × 8 GPUs, 20 min sim
Reference: cache-sim heavy_coding_report.md, global-vs-local section
Metrics:
  - Global L3A hit rate:       target ≥99.0%        (reference: 99.8%)
  - Local L3A hit rate:        target 65%–72%       (reference: 68.6%)
  - QUALITATIVE REQUIREMENT:   global > local by ≥25 percentage points
  - Queue wait global mean:    within ±30%          (reference: 2.7s)
  - Queue wait local mean:     within ±30%          (reference: 50.8s)
  - Completed requests ratio:  global/local ≥ 1.5×  (reference: 1.84×)
```

**Scenario C: Node scaling (topology correctness)**
```
Config:    heavy_coding.json, 1/2/4 workers, 20 min sim
Reference: cache-sim node_scaling table
Metrics:
  - At 1 worker: global ≈ local hit rate (within ±2 points)
  - At 2+ workers: global maintains ≥98%, local degrades
  - Ranking of queue wait p95 preserved across worker counts
```

### Authoritative vs Exploratory Metrics

| Metric | Status | Rationale |
|--------|--------|-----------|
| Cache hit rate by tier | Authoritative | Core correctness check |
| TTFT p50/p95/p99 | Authoritative | User-facing latency |
| Queue wait mean/p95 | Authoritative | System health |
| Completed vs dropped requests | Authoritative | Throughput |
| Global vs local hit rate delta | Authoritative | Architecture decision |
| Slot utilization | Exploratory | Useful but less sensitive |
| Per-node L1/L2 occupancy | Exploratory | Debug aid |
| Cold eviction rate | Exploratory | Secondary signal |

### Gate: Phase 0.5 Complete
- [ ] All three benchmark scenarios defined in code as test fixtures
- [ ] Tolerance bands approved by team
- [ ] Reference numbers from cache-sim committed to repo as golden files
- [ ] Authoritative vs exploratory metric list agreed

---

## Phase 1 — Core DES Layer (3 weeks)
### Goal: cache-sim becomes the canonical Layer 1 engine

### What Gets Ported

Port these cache-sim modules into `agentsim/core/des/`, adapting interfaces
to the contracts from Phase 0. Not file-for-file copy — interface-first port.

| cache-sim module | AgentSim target | Key changes |
|------------------|-----------------|-------------|
| `engine.py` | `core/des/engine.py` | Emit `DESEvent` objects; accept `ObserverBase` list |
| `cache.py` | `core/des/cache.py` | All sizes in bytes; tokens as metadata field |
| `eviction.py` | `core/des/eviction.py` | Implement `EvictionPolicyBase` ABC |
| `oracle.py` | `core/des/oracle.py` | Implement `CacheOracleBase`; return confidence label |
| `workload.py` | `core/des/workload.py` | Adopt NHPP + 5 profiles directly |
| `dispatch.py` | `core/des/dispatch.py` | Implement `DispatcherBase`; push + pull |
| `service.py` | `core/des/service.py` | Slot model; prefill + decode pools |
| `node.py` | `core/des/node.py` | Per-GPU state with worker_id |
| `metrics.py` | `core/des/metrics.py` | 27 metrics; implement `ObserverBase` |

### Byte-Based Accounting — Implementation Detail

Every cache object in the ported engine uses bytes, not tokens:

```python
# When creating a KV cache object from a request:
kv_bytes = (
    request.total_context_tokens
    * model.num_layers
    * model.num_kv_heads
    * model.head_dim
    * 2           # K and V
    * dtype_bytes  # 2 for BF16
)

cache_obj = CacheObject(
    object_id    = f"{request.session_id}:{request.turn_id}",
    size_bytes   = kv_bytes,       # ← bytes, authoritative
    token_count  = request.total_context_tokens,  # ← metadata only
    ...
)
```

Tier capacities:
```python
# NOT: hbm_capacity_tokens = 80_000_000_000 / bytes_per_token
# YES:
L1_spec = TierSpec(
    name            = "L1",
    capacity_bytes  = 80 * 1024**3,   # 80 GB
    bandwidth_bps   = 3_000 * 1024**3, # 3 TB/s
    block_size_bytes = 5 * 1024,       # 5 KB (cache-sim default)
    scope           = "per_gpu",
    latency_floor_us = 1,
)
```

### Oracle Implementation

Port cache-sim's piecewise-linear interpolation from measured A100 tables.
Add confidence label at load time:

```python
class PiecewiseOracle(CacheOracleBase):
    def __init__(self, table_path: str, chip_name: str):
        self.table     = load_json(table_path)
        self.chip_name = chip_name
        self.confidence = self._detect_confidence(table_path)

    def _detect_confidence(self, path: str) -> ConfidenceLabel:
        meta = load_json(path).get("metadata", {})
        source = meta.get("source", "unknown")
        if source == "measured":
            return ConfidenceLabel.CALIBRATED
        elif source == "interpolated":
            return ConfidenceLabel.SEMI_CALIBRATED
        else:
            return ConfidenceLabel.ANALYTICAL_ONLY

    def prefill_latency_us(self, uncached_tokens, chip_name, model_name):
        latency_us = self._interpolate(uncached_tokens)
        return latency_us, self.confidence
```

### DES Event Emission

The engine emits events. Observers receive them. No callbacks into engine:

```python
class SimEngine:
    def __init__(self, config, observers: list[ObserverBase] = None):
        self._observers = observers or []

    def _emit(self, event: DESEvent):
        for obs in self._observers:
            obs.on_event(event)   # observers are read-only

    def _on_prefill_complete(self, request, result):
        self._emit(DESEvent(
            event_id    = next(self._id_counter),
            sim_time_us = self._clock,
            kind        = DESEventKind.PREFILL_COMPLETE,
            payload     = {
                "session_id":        request.session_id,
                "request_id":        request.request_id,
                "ttft_us":           self._clock - request.arrival_us,
                "prefill_latency_us": result.prefill_latency_us,  # compute only
                "queue_wait_prefill_us": result.queue_wait_prefill_us,
                "cache_tier":        result.cache_tier,
                "savings_event":     result.savings_event.value,  # SavingsEvent
                "bytes_loaded":      result.bytes_loaded,
                "latency_source":    result.latency_source,
                "cache_key":         result.cache_key,  # CacheKey for reuse tracking
            }
        ))

    def _on_decode_complete(self, request, result):
        self._emit(DESEvent(
            event_id    = next(self._id_counter),
            sim_time_us = self._clock,
            kind        = DESEventKind.DECODE_COMPLETE,
            payload     = {
                "session_id":         request.session_id,
                "request_id":         request.request_id,
                "decode_latency_us":  result.decode_latency_us,   # separate from prefill
                "queue_wait_decode_us": result.queue_wait_decode_us,
                "output_tokens":      result.output_tokens,
            }
        ))
```

### Research Questions This Phase Answers

Phase 1 is the first phase where the engine runs real workloads. These
questions can be answered here for the first time and golden files
captured for regression testing going forward.

**Q1: At what worker count does global L3A become essential?**
Run Scenario C (node scaling: 1/2/4 workers) and record the crossover
point where local L3A hit rate drops below 90%. This is the inflection
point that informs deployment topology decisions.

**Q2: What is the steady-state cache hit rate distribution under heavy
agentic coding load on a single worker?**
Run Scenario A and record the SavingsEvent breakdown: how many requests
are HIT_L1 vs HIT_L2_WIN vs HIT_L3_WIN vs MISS_RECOMPUTE. This is the
baseline against which all framework optimizations in Phase 4 are measured.

**Q3: What fraction of sessions experience at least one TTL expiry
miss under realistic inter-turn think times?**
Run the agentic_coding profile for 20 minutes and record the expiry
miss rate. This establishes whether the 5-min TTL is a meaningful
problem for this workload or a theoretical concern.

Golden files for these three questions are captured at the Phase 1 gate
and committed to `results/phase1-baseline/`.

### Gate: Phase 1 Complete
- [ ] All three benchmark scenarios pass tolerance bands (from Phase 0.5)
- [ ] Global/local L3A delta ≥25 percentage points reproduced
- [ ] All cache object creation uses bytes, zero token-count accounting
- [ ] Every prefill oracle call returns (latency, confidence_label)
- [ ] `prefill_latency_us` and `decode_latency_us` reported separately in all metrics
- [ ] `SavingsEvent` classification present on every completed request
- [ ] `CacheKey` used for all cache lookups — no token-count-based hit detection
- [ ] cache-sim's 27 tests ported and passing
- [ ] **At least one stress scenario passes**: high arrival rate (peak=5, 4 workers)
  with queue saturation. Verify dropped request count and queue wait p95 are
  plausible — most bugs surface under contention, not steady state.
- [ ] No-cache baseline test passes (see `test_invariants.py` — new test):
  with cache disabled, hit rate = 0%, all TTFT values match cold oracle
- [ ] Heavy coding report reproducible via `python scripts/reproduce_cache_sim_baseline.py`
- [ ] Golden files captured and committed to `results/phase1-baseline/`:
  Q1 node-scaling crossover point, Q2 SavingsEvent breakdown, Q3 expiry miss rate

---

## Phase 2 — Observation Layer (2 weeks)
### Goal: Re-home protocol/event work as downstream-only observers

### What This Phase Does

Takes the existing `events.py`, `AnthropicEventMapper`, `OpenAIChatEventMapper`
from the current AgentSim prototype and wires them to consume `DESEvent`
objects from the Core DES engine instead of driving their own SimPy loop.

**Key constraint enforced at code level:**

```python
class ProtocolObserver(ObserverBase):
    """
    Reads DES events. Translates to protocol-level checkpoints.
    Cannot modify simulation state.
    """
    def on_event(self, event: DESEvent) -> None:
        if event.kind == DESEventKind.PREFILL_COMPLETE:
            self._on_prefill_complete(event)
        elif event.kind == DESEventKind.CACHE_LOOKUP:
            self._on_cache_lookup(event)
        # ... etc

    def _modify_simulation(self):
        raise NotImplementedError("Observers are read-only")
```

### Anthropic Miss Taxonomy — Mapped to DES Events

```
DES Event                         → Checkpoint           → Miss Type
──────────────────────────────────────────────────────────────────
CACHE_LOOKUP (tier=None, turn=0)  → CACHE_DECISION       → COLD
CACHE_LOOKUP (tier=None,          → CACHE_DECISION       → EXPIRY
  session_idle > 300s)
CACHE_LOOKUP (tier=None,          → CACHE_DECISION       → EVICTION
  prev_write > 0, idle < 300s)
CACHE_LOOKUP (tier=L3A_global,    → CACHE_DECISION       → TRANSFER
  cross_worker=True)
PREFILL_COMPLETE                  → TTFT
TIER_TRANSFER (bytes, latency)    → KV_TRANSFER_COMPLETE
EVICTION (pressure)               → EVICTION_TRIGGER
DECODE_COMPLETE                   → TURN_COMPLETE
```

### Anthropic 5-Minute TTL — Implemented as Eviction Policy

The 5-min TTL is not a cache-sim TTL (that's a tier-migration timer).
It's a full object invalidation when session is idle. Implemented as
a dedicated eviction policy that coexists with pressure-based eviction:

```python
class AnthropicTTLEvictionPolicy(EvictionPolicyBase):
    """
    Invalidates all tiers for a session after 300s idle.
    Separate from L1→L2→L3A pressure eviction.
    Not part of cache-sim's eviction model — AgentSim addition.
    """
    TTL_US = 300 * 1_000_000  # 300 seconds in microseconds

    def select_eviction_candidates(self, tier, needed_bytes):
        now = self._engine.clock_us
        return [
            obj for obj in tier.objects.values()
            if now - obj.last_access_us >= self.TTL_US
        ]
```

### OpenAI Cache Detection — Inference from TTFT

OpenAI Chat Completions and Responses API don't expose cache tokens.
Detection is inference-based:

```python
class OpenAIProtocolObserver(ProtocolObserver):
    """
    Infers cache hit/miss from TTFT anomaly relative to calibrated baseline.
    CANNOT determine EXPIRY vs COLD_MISS — both appear as PROBABLE_MISS.
    Confidence: lower than Anthropic protocol.
    """
    HIT_THRESHOLD_RATIO = 0.35   # TTFT < 35% of baseline → probable hit

    def _on_prefill_complete(self, event):
        ttft_us   = event.payload["ttft_us"]
        ratio     = ttft_us / self.baseline_ttft_us
        hit       = ratio <= self.HIT_THRESHOLD_RATIO
        self._emit_checkpoint(
            kind          = CheckpointKind.CACHE_DECISION,
            cache_hit     = hit,
            confidence    = ConfidenceLabel.SEMI_CALIBRATED,
            inferred_from = "ttft_ratio",
        )
```

### Research Questions This Phase Answers

Phase 2 adds the protocol observation layer. For the first time, cache
decisions can be interpreted through the lens of the Anthropic and
OpenAI API event streams — the same view a real client would have.

**Q4: Does the Anthropic cache miss taxonomy correctly classify all
miss types in the heavy coding workload?**
Run the heavy_coding scenario through the AnthropicProtocolObserver
and verify the COLD / EXPIRY / EVICTION / TRANSFER breakdown is
consistent with the Phase 1 SavingsEvent numbers. If they diverge,
the observation layer has a mapping bug.

**Q5: What fraction of cache misses would be invisible to an OpenAI
API client vs an Anthropic API client?**
Compare the miss detection rate of OpenAIChatEventMapper (TTFT-inferred)
against AnthropicProtocolObserver (explicit cache tokens) on the same
session stream. This quantifies the observability gap between protocols
and informs monitoring strategy.

### Gate: Phase 2 Complete
- [ ] All checkpoint events consumed from DES events, not SimPy
- [ ] `AnthropicProtocolObserver.on_event()` correctly classifies all
  four miss types (COLD/EXPIRY/EVICTION/TRANSFER) from DES payloads
- [ ] No method in any observer modifies engine state
- [ ] Anthropic 5-min TTL eviction policy integrated and tested
- [ ] `SavingsEvent` visible in observation layer output (translated from DES payload)
- [ ] Prefill and decode latencies reported separately in checkpoint events
- [ ] Observer time-coupling test passes: observers produce identical output
  regardless of the order they receive same-timestamp events
- [ ] `EventStream.miss_summary()` produces correct output on
  heavy_coding scenario (hit rate consistent with Phase 1 numbers)

---

## Phase 3 — Non-GPU Parameterization (2 weeks)
### Goal: ChipProfile generalization, alternative tiers, confidence labels

This phase is deliberately narrower than the original proposal.
It does NOT attempt full fidelity for all targets immediately.
It establishes the correct framework and ships calibrated support
for one non-GPU target.

### Tier Parameterization

The Core DES engine's tier specs become config-driven. Two NPU configs
are provided — local and global L3A — mirroring the GPU toggle that
produced the central finding in Phase 1. This lets us ask the same
global vs local question on non-GPU hardware.

```python
# configs/custom_npu_local_l3a.json
{
  "chip_name": "custom_npu_local_l3a",
  "confidence": "analytical-only",
  "oracle_table": null,
  "tiers": [
    {
      "name": "L1",
      "medium": "HBM",
      "capacity_bytes": 34359738368,    // 32 GB
      "bandwidth_bps":  966367641600,   // 900 GB/s
      "block_size_bytes": 4194304,      // 4K-token page × ~320 bytes/token (placeholder)
      "scope": "per_gpu",
      "latency_floor_us": 1
    },
    {
      "name": "L2",
      "medium": "DDR",
      "capacity_bytes": 274877906944,   // 256 GB
      "bandwidth_bps":  214748364800,   // 200 GB/s
      "block_size_bytes": 33554432,     // 32 MB
      "scope": "per_worker",
      "latency_floor_us": 200
    },
    {
      "name": "L3A",
      "medium": "SSD",
      "capacity_bytes": 8796093022208,  // 8 TB
      "bandwidth_bps":  7516192768,     // 7 GB/s
      "block_size_bytes": 268435456,    // 256 MB
      "scope": "per_worker",
      "latency_floor_us": 1000,
      "remote_latency_us": 0
    }
  ],
  "interconnect_bps": 429496729600,    // 400 GB/s
  "n_gpus_per_worker": 4
}

// configs/custom_npu_global_l3a.json — identical except:
//   "scope": "global"
//   "remote_latency_us": 50000        // 50ms cross-worker, same as GPU
```

**Note on "4K page size"**: "4K" means 4096 tokens per KV page.
In bytes for Llama3-70B BF16: 4096 × (80 layers × 8 kv_heads × 128 dim × 2 × 2) ≈ 671 MB.
The placeholder `block_size_bytes` above should be overridden at runtime
with `model.block_size_bytes_for_tokens(4096)` once the model config is known.

### Oracle Table Structure

```json
// benchmarks/oracle_tables/a100_llama3_70b.json
{
  "metadata": {
    "chip": "nvidia_a100_80g",
    "model": "llama3_70b_fp16",
    "source": "measured",
    "confidence": "calibrated",
    "measured_by": "cache-sim prototype",
    "date": "2024"
  },
  "points": [
    {"tokens": 512,    "latency_us": 45000},
    {"tokens": 1024,   "latency_us": 112000},
    {"tokens": 4096,   "latency_us": 680000},
    {"tokens": 10000,  "latency_us": 2200000},
    {"tokens": 30000,  "latency_us": 14900000},
    {"tokens": 50000,  "latency_us": 37000000},
    {"tokens": 65000,  "latency_us": 54400000},
    {"tokens": 100000, "latency_us": 120700000}
  ],
  "extrapolation": "quadratic"
}
```

For custom NPU targets with no measurements:
```json
// benchmarks/oracle_tables/custom_npu_v1_llama3_70b.json
{
  "metadata": {
    "source": "roofline_model",
    "confidence": "analytical-only",
    "roofline_params": {
      "hbm_bandwidth_bps": 966367641600,
      "compute_tflops": 200,
      "compute_efficiency": 0.55,
      "memory_efficiency": 0.85
    }
  },
  "points": null,
  "extrapolation": "roofline"
}
```

### Confidence Label in Every Output

Every report, every metrics CSV, every plot title includes the label:

```python
def generate_report(metrics, config) -> Report:
    confidence = config.oracle.confidence

    return Report(
        metadata = ReportMetadata(
            chip_name    = config.chip_name,
            model_name   = config.model_name,
            confidence   = confidence,
            oracle_source = config.oracle.table_path or "roofline",
        ),
        headline = f"[{confidence.upper()}] {config.chip_name} / {config.model_name}",
        # ... metrics ...
    )
```

### Research Questions This Phase Answers

Phase 3 parameterizes the hardware model. For the first time, the
simulator can compare different chip targets on the same workload.

**Q6: How does a custom NPU with HBM+DDR+SSD compare to an A100 on
agentic coding TTFT at the same cache hit rate?**
Run the agentic_coding profile on both `custom_npu_global_l3a` and
`nvidia_a100_80g` configs and compare TTFT p50/p95. The NPU result is
labeled ANALYTICAL_ONLY — directional comparison is still meaningful
for architecture decisions even without a calibrated oracle table.

**Q7: What is the break-even point where DDR tier access becomes
worthwhile vs. full recompute on the custom NPU?**
The `SavingsEvent.classify()` method computes this per request.
Aggregate across sessions to find the context length threshold where
HIT_L2_WIN rate exceeds HIT_L3_BREAK_EVEN rate. This directly informs
DDR capacity sizing for the NPU.

**Q8 (new): Does global L3A provide the same benefit on the custom NPU
as it does on GPU hardware?**
Run `custom_npu_local_l3a` vs `custom_npu_global_l3a` on the heavy
coding profile with 4 workers. Compare hit rate delta and queue wait
ratio against the GPU result from Phase 1 (Scenario B). Expected:
global NPU L3A shows the same directional improvement as global GPU L3A,
with magnitude differences due to different interconnect bandwidth
(400 GB/s NPU vs 900 GB/s NVLink). If the delta is smaller on the NPU,
the lower interconnect bandwidth reduces the benefit of cross-worker
KV transfer.

Golden files for Q6/Q7/Q8 committed to `results/phase3-baseline/`.
All NPU results labeled ANALYTICAL_ONLY in every output.

### Gate: Phase 3 Complete
- [ ] At least one non-GPU chip (`custom_npu_local_l3a`) runs end-to-end
  through Core DES with correct byte-based tier accounting
- [ ] Both `custom_npu_local_l3a` and `custom_npu_global_l3a` configs run
  and produce different hit rates at 4 workers (same directional delta as GPU)
- [ ] All outputs for analytical-only targets show "ANALYTICAL-ONLY" label
- [ ] Calibrated oracle (A100) and analytical-only oracle (custom NPU)
  produce meaningfully different TTFT predictions at 50k tokens
  (verifies roofline fallback is not silently wrong)
- [ ] Tier parameterization confirmed working with all three configs:
  2-tier (HBM+DDR — remove L3A from custom_npu to test),
  3-tier local L3A, 3-tier global L3A
- [ ] Golden files captured and committed to `results/phase3-baseline/`:
  Q6 NPU vs A100 TTFT, Q7 DDR break-even context length,
  Q8 global vs local L3A delta on NPU vs GPU

---

## Phase 4 — Framework Adapters (2 weeks)
### Goal: Schema and policy compatibility with vLLM and SGLang

Scope is deliberately narrow. We are not promising runtime fidelity.
Success is defined as conceptual and schema-level compatibility.

### What This Phase Delivers

**4a. Config/schema compatibility**

vLLM and SGLang concepts map to AgentSim config objects:

```python
# Framework concept → AgentSim equivalent
vLLM BlockManager block_size      → TierSpec.block_size_bytes (L1)
vLLM scheduler.max_num_seqs       → ServiceConfig.n_prefill_slots_per_gpu
vLLM prefix_caching = True        → DispatchConfig.affinity_mode = "prefix"
SGLang RadixAttention             → EvictionPolicy = "prefix_aware_lru"
LMCache cpu_offloading            → TierSpec L2 active, L3A optional
LMCache chunk_size = 256          → TierSpec.block_size_bytes L2
```

**4b. Workload/policy mapping**

Two adapters that translate framework config files into AgentSim configs:

```python
class VLLMConfigAdapter:
    """
    Reads a vLLM serve config and produces an AgentSim SimConfig.
    Does not run vLLM. Maps scheduling semantics to DES equivalents.
    """
    def from_vllm_config(self, vllm_config: dict) -> SimConfig:
        return SimConfig(
            service = ServiceConfig(
                n_prefill_slots_per_gpu = vllm_config["max_num_seqs"],
                block_size_bytes        = vllm_config["block_size"] * bytes_per_token,
            ),
            dispatch = DispatchConfig(
                policy = "sarathi" if vllm_config.get("chunked_prefill") else "orca",
            ),
            ...
        )

class LMCacheConfigAdapter:
    """Maps LMCache tier config to AgentSim L2/L3A tier specs."""
    def from_lmcache_config(self, lmc_config: dict) -> list[TierSpec]:
        tiers = []
        if lmc_config.get("local_cpu"):
            tiers.append(TierSpec(
                name            = "L2",
                medium          = "DRAM",
                capacity_bytes  = lmc_config["max_local_cpu_size"] * 1024**3,
                block_size_bytes = lmc_config["chunk_size"] * bytes_per_token,
                ...
            ))
        return tiers
```

**4c. Sub-agent spawning**

Added to Core DES engine session lifecycle (not a separate system):

```python
# In engine._on_prefill_complete():
if request.spawns_subagent:
    sub_request = SubAgentRequest(
        parent_session_id = request.session_id,
        model_tier        = "lightweight",  # Haiku-class
        system_prefix_tokens = request.subagent_system_tokens,
    )
    self._schedule(ARRIVAL, sub_request, delay_us=0)
```

### Research Questions This Phase Answers

Phase 4 adds framework-level schema mapping. These are the questions
that motivated the entire project. Results are schema-level comparisons,
not runtime fidelity — see "What This Phase Does NOT Do" below.

**Q9: Does LMCache CPU offload reduce TTFT p95 on the agentic_coding
workload compared to no KV offload, modeled via tier config?**
Compare two configs: vLLM baseline (L1 only) vs vLLM + LMCache
(L1 + L2 CPU offload). Run on heavy agentic_coding profile, 4 workers.
Expected: TTFT p95 improves by ≥20% with LMCache. If it doesn't, the
L2 bandwidth model for the target hardware needs revisiting.

**Q10: Does Sarathi chunked prefill reduce chat TTFT p99 in a mixed
coding+chat workload without degrading coding throughput?**
Compare two scheduler configs: Orca (prefill-prioritizing) vs Sarathi
(chunked prefill). Run on a 50/50 coding+chat mixed profile, single worker.
Expected: chat TTFT p99 drops, coding throughput within 10%.
Note: this compares scheduling policy semantics via config mapping —
not actual vLLM chunked prefill execution.

**Q11: How does SGLang RadixAttention (prefix-aware LRU eviction) compare
to vLLM LRU eviction on cache hit rate for agentic coding sessions?**
Compare SGLangAdapter (prefix_aware_lru eviction) vs VLLMAdapter (standard
LRU) on the agentic_coding profile. Expected: SGLang config shows higher
partial-hit rate because RadixAttention preserves shared prefixes longer.

Golden files for Q9/Q10/Q11 committed to `results/phase4-baseline/`.
These become the regression baseline for any future engine changes.

### What This Phase Does NOT Do

- Does not run real vLLM or SGLang processes
- Does not validate actual scheduling behavior against live framework
- Does not claim TTFT predictions match vLLM/SGLang exactly
- Does not implement chunked-prefill micro-scheduling

### Gate: Phase 4 Complete
- [ ] VLLMConfigAdapter converts a real vLLM config file without error
- [ ] LMCacheConfigAdapter produces correct L2 tier spec from real LMCache config
- [ ] Sub-agent spawning tested: parent + 2 sub-agents run concurrently in DES
- [ ] Config compatibility documented with explicit "schema mapping only" caveat
- [ ] Golden files captured and committed to `results/phase4-baseline/`:
  Q9 LMCache vs baseline TTFT, Q10 Sarathi vs Orca mixed workload,
  Q11 SGLang vs vLLM prefix hit rate
- [ ] All 11 research questions (Q1–Q11) are answerable from the
  results in `results/phase1-baseline/` through `results/phase4-baseline/`

---

## SimPy Sweep Tool — Constraints and Allowed Usage

SimPy lives in `agentsim/sweep/` not `agentsim/sim/`.
Its role is explicit and bounded.

**Allowed:**
- Use `SweepEstimator.estimate_turn_latency_ms()` for latency
- Share `SimConfig`, `ChipProfile`, `WorkloadProfile` objects with Core DES
- Run thousands of config combinations quickly to identify interesting parameter regions
- Label all output as `"source": "sweep-estimate"`

**Prohibited:**
- Instantiating `TierStore` directly
- Calling `EvictionPolicyBase` implementations
- Computing tier occupancy over time
- Any output labeled as authoritative simulation results

**Handoff pattern:**

```python
# Sweep identifies interesting region
sweep_results = run_simpy_sweep(
    configs    = param_grid,
    estimator  = SimpleTTFTEstimator(oracle),
    label      = "sweep-estimate",
)

# Then run the Core DES on the top-N configs from the sweep
top_configs = sweep_results.top_n(5, metric="hit_rate")
for cfg in top_configs:
    des_result = run_des_simulation(cfg)   # authoritative
    compare(sweep_results[cfg], des_result)  # validates sweep accuracy
```

---

## Repository Final Structure

```
agentsim/
├── core/
│   ├── contracts.py        ← Phase 0: all ABCs and dataclasses
│   ├── des/                ← Phase 1: Core DES layer (from cache-sim)
│   │   ├── README.md       ← "This is the only authoritative simulation engine.
│   │   │                      See ../contracts.py for all interface definitions."
│   │   ├── engine.py       ← emits DESEvent objects
│   │   ├── cache.py        ← byte-based TierStore, PrefixTrie
│   │   ├── eviction.py     ← implements EvictionPolicyBase
│   │   ├── oracle.py       ← implements CacheOracleBase + confidence labels
│   │   ├── workload.py     ← NHPP, 5 profiles
│   │   ├── dispatch.py     ← push/pull, implements DispatcherBase
│   │   ├── service.py      ← GPU slot model
│   │   ├── node.py         ← per-GPU state
│   │   └── metrics.py      ← 27 metrics, implements ObserverBase
│   └── observation/        ← Phase 2: protocol event layer
│       ├── events.py       ← DESEvent → CheckpointEvent translation
│       ├── anthropic.py    ← Anthropic SSE miss taxonomy
│       ├── openai_chat.py  ← OpenAI Chat inferred detection
│       └── openai_resp.py  ← OpenAI Responses API
│
├── integration/            ← Phase 3+4
│   ├── chips/
│   │   ├── profiles.py     ← ChipProfile dataclasses
│   │   └── oracle_tables/  ← per-chip JSON benchmark tables
│   ├── adapters/
│   │   ├── vllm.py         ← VLLMConfigAdapter
│   │   ├── sglang.py       ← SGLangConfigAdapter
│   │   └── lmcache.py      ← LMCacheConfigAdapter
│   └── reporting.py        ← confidence-labeled output generation
│
├── sweep/                  ← SimPy sweep tool (non-authoritative)
│   ├── contracts.py        ← SweepEstimator ABC + "sweep-estimate" label
│   ├── estimators.py       ← simple latency estimators (not tier-stateful)
│   └── runner.py           ← parallel sweep execution
│
├── configs/
│   ├── heavy_coding.json        ← from cache-sim (canonical)
│   ├── default.json             ← from cache-sim
│   ├── custom_npu_local_l3a.json  ← NPU: HBM+DDR+SSD, local L3A
│   └── custom_npu_global_l3a.json ← NPU: HBM+DDR+SSD, global L3A (pooled)
│
├── benchmarks/
│   └── oracle_tables/
│       ├── a100_llama3_70b.json    ← from cache-sim (calibrated)
│       ├── h100_llama3_70b.json    ← new (semi-calibrated)
│       └── custom_npu_llama3_70b.json  ← new (analytical-only)
│
├── results/
│   └── v0.1-prototype-baseline/   ← captured before migration branch
│       ├── sanity_plots/           ← reference plots from old engine
│       └── heavy_coding_outputs/   ← reference numbers for Phase 1 validation
│
├── scripts/
│   ├── reproduce_baseline.py    ← Phase 1 gate: reproduce cache-sim results
│   ├── sanity_plots.py          ← from cache-sim
│   └── sweep.py                 ← parameter sweep runner
│
├── tests/
│   ├── test_invariants.py       ← from cache-sim (27 tests)
│   ├── test_phase0_contracts.py ← new: interface enforcement tests
│   ├── test_phase1_baseline.py  ← new: tolerance-band reproduction
│   └── test_phase2_observers.py ← new: observer read-only enforcement
│
├── .importlinter               ← import boundary enforcement (CI)
│
└── _deprecated/                ← DISCONNECT phase (after Phase 1 gate)
    │   Old modules moved here before deletion in Phase 2/3.
    │   Not on import path. Kept for reference only.
    ├── hardware_model.py        ← replaced by integration/chips/profiles.py
    ├── session_model.py         ← replaced by core/des/workload.py
    └── sim/request_sim.py       ← replaced by sweep/request_sweep.py
```

**Branch merge rule:** `feature/des-core-swap` merges to `main` only after
the Phase 1 gate passes. No exceptions.

---

## Timeline Summary

| Phase | Duration | Hard Gate |
|-------|----------|-----------|
| Pre-0: Repo setup | 1 day | v0.1-prototype tagged, branch created, lint CI active |
| 0: Architecture Contracts | 1 week | Interfaces reviewed + frozen, des/README committed |
| 0.5: Validation Contract | 3 days | Golden files committed, tolerances agreed |
| 1: Core DES Layer | 3 weeks | 3 benchmark scenarios pass; branch merges to main |
| 2: Observation Layer | 2 weeks | Miss taxonomy correct on heavy_coding |
| 3: Non-GPU Parameterization | 2 weeks | custom_npu end-to-end + confidence labels |
| 4: Framework Adapters | 2 weeks | Schema compat + sub-agents working |
| **Total** | **~11 weeks** | |

---

## What This Plan Preserves from Current AgentSim Prototype

| Artifact | Status | Where It Lives |
|----------|--------|----------------|
| Goal 1/2/3 framing | Preserved | This document |
| `events.py` checkpoint schema | Preserved + adapted | `core/observation/events.py` |
| `AnthropicEventMapper` | Preserved + adapted | `core/observation/anthropic.py` |
| `OpenAIChatEventMapper` | Preserved + adapted | `core/observation/openai_chat.py` |
| `hardware_model.py` ChipProfile | Preserved + extended | `integration/chips/profiles.py` |
| `session_model.py` AgenticSession | Demoted to concept | Replaced by workload.py profiles |
| `request_sim.py` SimPy engine | Demoted to sweep tool | `sweep/runner.py` |
| `KVCacheManager` token-based | Replaced | cache.py byte-based TierStore |
| `predict_prefill()` roofline | Demoted to fallback | oracle.py analytical-only mode |
