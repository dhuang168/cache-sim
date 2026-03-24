# Prompt Cache Effectiveness Simulator — Implementation Spec

**Version:** 1.0 (v1 scope)
**Status:** Ready for implementation
**Target:** Claude Code / engineer handoff

---

## 0. How to use this spec

This document is structured for top-down implementation. Build and validate each section in order — later sections depend on earlier ones being correct. The v1/v2 boundary is explicit throughout; do not implement anything marked `[v2]` until v1 passes validation.

**Primary questions this simulator must answer:**

1. What is the marginal value of each additional TB of L2 RAM? (at what capacity does TTFT benefit flatten?)
2. Under what workload conditions is resume-from-cache cheaper than recompute? (crossover curve as a function of ISS and IAT)
3. How sensitive are p50/p95/p99 TTFT and GPU utilization to TTL threshold and block size?
4. What fraction of cache ROI comes from within-session reuse vs. cross-session shared system prefix?

**Authoritative outputs** (high fidelity required): tier saturation %, TTFT distributions, cache hit rate by tier, sharing factor.

**Exploratory outputs** (directional only, report with caveats): memory pollution overhead, eviction rates, GPU queue depth.

---

## 1. Repository layout

```
cache_sim/
├── sim/
│   ├── __init__.py
│   ├── config.py          # SimConfig dataclass + JSON loader
│   ├── engine.py          # Discrete-event engine (main loop)
│   ├── events.py          # Event types and FSM transitions
│   ├── cache.py           # CacheObject, radix trie, tier store
│   ├── service.py         # Two-stage prefill/decode service model
│   ├── workload.py        # Workload synthesizer (session profiles)
│   ├── oracle.py          # Latency oracle (calibrated curves)
│   ├── eviction.py        # Eviction policies
│   └── metrics.py         # Metrics collector and reporter
├── benchmarks/
│   └── latency_tables/
│       ├── prefill_70b_a100.json   # benchmark-derived lookup table
│       └── prefill_400b_a100.json
├── configs/
│   └── default.json        # reference SimConfig
├── tests/
│   ├── test_invariants.py  # 4 mandatory invariant tests
│   ├── test_kv_size.py
│   ├── test_oracle.py
│   └── test_engine.py
├── scripts/
│   └── sweep.py            # sensitivity sweep runner (parallelized)
└── README.md
```

---

## 2. `SimConfig` — single source of truth

All parameters live in one dataclass. Any run can be reproduced by saving its config JSON.

```python
# sim/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json

@dataclass
class TierConfig:
    name: str
    capacity_bytes: int
    bandwidth_bytes_per_s: int       # peak bandwidth
    latency_floor_us: int            # microseconds
    block_size_bytes: int            # allocation granularity

@dataclass
class ModelConfig:
    model_id: str                    # e.g. "llama3-70b"
    n_layers: int
    n_kv_heads: int
    head_dim: int
    bytes_per_element: int           # 2 = FP16, 1 = FP8

@dataclass
class WorkloadProfile:
    name: str                        # "chat" | "coding" | "batch" | "agent"
    arrival_rate_peak: float         # requests/second at peak
    diurnal_peak_trough_ratio: float # 4.0 typical
    iat_mean_s: float                # mean inter-arrival time within session
    iat_dist: str                    # "exponential" | "lognormal"
    input_len_mean_tokens: int
    input_len_sigma_tokens: int
    output_len_pareto_alpha: float
    output_len_pareto_xmin: int
    context_growth_min_tokens: int
    context_growth_max_tokens: int
    prefix_stability_initial: float  # fraction of context that is stable prefix
    prefix_stability_final: float    # decays toward this over session
    session_duration_mean_s: float
    session_duration_dist: str       # "lognormal" | "exponential"
    shared_system_prefix_tokens: int # 0 = no shared prefix

@dataclass
class ServiceConfig:
    n_prefill_slots: int             # concurrent prefill sequences
    n_decode_slots: int              # concurrent decode sequences
    prefill_queue_max: int           # max queued prefills before backpressure
    decode_queue_max: int

@dataclass
class SimConfig:
    # Identity
    run_id: str
    seed: int

    # Hardware tiers (ordered L1 -> L2 -> L3a)
    tiers: list[TierConfig]

    # Model
    model: ModelConfig

    # Workload
    profiles: list[WorkloadProfile]
    profile_mix: dict[str, float]    # name -> fraction of total arrivals; must sum to 1.0

    # Service model
    service: ServiceConfig

    # Cache policy
    ttl_l2_s: float                  # seconds before L2 -> L3a hibernation
    ttl_l3a_s: float                 # seconds before L3a eviction
    eviction_hbm_threshold: float    # 0.9 = evict when L1 is 90% full
    eviction_ram_threshold: float    # 0.95

    # Simulation control
    sim_duration_s: float            # simulated seconds to run
    warmup_s: float                  # exclude from metrics (fill caches first)
    epoch_report_interval_s: float   # how often to emit EPOCH_REPORT events

    # Extensions (v2, keep False for v1)
    enable_suffix_cache: bool = False
    enable_l3b_object_store: bool = False

    @classmethod
    def from_json(cls, path: str) -> SimConfig:
        with open(path) as f:
            return cls(**json.load(f))

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
```

**Reference config** (`configs/default.json`) must encode:
- L1: 80 GB HBM, 3 TB/s, 1 µs floor, 16-token blocks (~5 KB for 70B FP16)
- L2: 4 TB RAM, 64 GB/s, 10 µs floor, 32 MB blocks
- L3a: 20 TB NVMe, 7 GB/s, 100 µs floor, 256 MB blocks
- Model: llama3-70b (80 layers, 8 KV heads, 128 head_dim, FP16)
- TTL L2: 300 s (5 minutes)

---

## 3. Event system

### 3.1 Event types

```python
# sim/events.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

class EventType(Enum):
    REQUEST_ARRIVAL   = auto()
    PREFILL_START     = auto()
    PREFILL_COMPLETE  = auto()
    DECODE_START      = auto()
    DECODE_COMPLETE   = auto()
    TTL_FIRE          = auto()
    TIER_EVICTION     = auto()
    SESSION_RESUME    = auto()
    EPOCH_REPORT      = auto()

@dataclass(order=True)
class Event:
    time_us: int              # simulated clock, microseconds (uint64)
    seq: int                  # tie-breaker for same-time events
    event_type: EventType
    session_id: str
    request_id: Optional[str] = None
    payload: Optional[dict] = None   # event-specific data

# Priority queue: heapq on (time_us, seq, event_type.value, ...)
```

### 3.2 Request lifecycle FSM

Each request moves through these states. Illegal transitions must raise `SimError`.

```
QUEUED
  -> CACHE_LOOKUP       (on PREFILL_START attempt)
  -> HIT_L1             (if CacheObject found in L1)
  -> HIT_L2             (if CacheObject found in L2; triggers L2->L1 transfer)
  -> HIT_L3A            (if CacheObject found in L3a; triggers L3a->L2->L1 transfer)
  -> MISS               (no cache object; full prefill)
  -> PREFILLING         (prefill in progress; input = uncached suffix only)
  -> DECODE_QUEUED
  -> DECODING
  -> KV_WRITE           (write new KV blocks to L1)
  -> COMPLETE
```

Cache hit states compute `T_trans` and schedule `PREFILL_COMPLETE` accordingly (transfer acts as a fast prefill). Miss state schedules `PREFILL_COMPLETE` using `oracle.prefill_latency(uncached_tokens)`.

---

## 4. Cache object model

### 4.1 Data structure

```python
# sim/cache.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

class Tier(Enum):
    L1 = 1
    L2 = 2
    L3A = 3

class BlockLayout(Enum):
    L1_SMALL = "l1_small"     # 16-token blocks
    L2_LARGE = "l2_large"     # 32 MB blocks
    L3A_ALIGNED = "l3a_aligned"  # 256 MB blocks

@dataclass
class CacheObject:
    session_id: str
    shared_prefix_id: Optional[str]   # None if session-private
    token_range: tuple[int, int]       # (start_token, end_token) inclusive
    model_id: str
    tier: Tier
    size_bytes: int                    # actual KV bytes
    block_count: int                   # blocks allocated (>= ceil(size_bytes/block_size))
    created_at_us: int
    last_accessed_at_us: int
    ref_count: int                     # >1 = shared across sessions
    block_layout: BlockLayout
    is_hibernated: bool = False        # True when written to L3a

    @property
    def allocated_bytes(self) -> int:
        # block_count * block_size derived from tier config
        raise NotImplementedError  # implemented as method on TierStore

    @property
    def internal_fragmentation_bytes(self) -> int:
        return self.allocated_bytes - self.size_bytes

    @property
    def prefix_length(self) -> int:
        return self.token_range[1] - self.token_range[0] + 1
```

### 4.2 KV size formula

```python
# sim/cache.py

def kv_size_bytes(
    token_count: int,
    model: ModelConfig,
    precision_override: Optional[int] = None,
) -> int:
    """
    Compute exact KV cache size for a token sequence.

    Formula: 2 * n_layers * n_kv_heads * head_dim * bytes_per_element * token_count

    The factor of 2 accounts for both K and V tensors.
    Validate against torch.profiler trace; expect <5% error.
    """
    bpe = precision_override or model.bytes_per_element
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * bpe * token_count
```

**Validation target:** `kv_size_bytes(1000, model_70b_fp16)` must equal approximately 327,680 bytes (0.328 MB). Fail fast with an assertion if it deviates by more than 5% from a reference measurement.

### 4.3 Block allocation and memory pollution

```python
def allocated_blocks(size_bytes: int, block_size_bytes: int) -> int:
    """Ceiling division — always round up to next full block."""
    import math
    return math.ceil(size_bytes / block_size_bytes)

def block_waste_ratio(size_bytes: int, block_size_bytes: int) -> float:
    """
    Internal fragmentation as a fraction of allocated space.
    block_waste = (allocated_blocks * block_size - actual_bytes) / (allocated_blocks * block_size)
    """
    n_blocks = allocated_blocks(size_bytes, block_size_bytes)
    allocated = n_blocks * block_size_bytes
    return (allocated - size_bytes) / allocated
```

Track aggregate `memory_pollution_overhead_bytes` in `MetricsCollector` as the running sum of `(allocated_bytes - size_bytes)` across all live CacheObjects in each tier.

### 4.4 Prefix radix trie

Use `marisa-trie` for the shared-prefix pool (read-heavy) and a plain `dict`-backed trie for per-session prefix tracking (write-heavy at session creation).

```python
class PrefixTrie:
    """
    Radix trie mapping token-sequence hash prefixes to CacheObject keys.

    Lookup: given a request's full token prefix of length N, return
    (matched_cache_key, matched_length_k). The caller then:
      - serves tokens [0, k) from cache
      - recomputes tokens [k, N)
    """
    def lookup(self, token_hashes: list[int]) -> tuple[Optional[str], int]:
        """Returns (cache_key_or_None, depth_k)."""
        ...

    def insert(self, token_hashes: list[int], cache_key: str) -> None:
        ...

    def evict_leaves(self, n: int) -> list[str]:
        """
        Evict n leaf nodes (least-shared, safest to remove).
        Returns list of evicted cache_keys.
        Shared-prefix nodes (ref_count > 1) are not eligible.
        """
        ...
```

**Shared prefix pool** is a module-level singleton `PrefixTrie` keyed by `(model_id, shared_prefix_id)`. All sessions sharing the same system prompt share this trie and increment `ref_count` on lookup.

---

## 5. Latency oracle

### 5.1 Prefill oracle

Do not compute prefill latency from a formula. Load from a benchmark table.

```python
# sim/oracle.py
import numpy as np
import json

class PrefillOracle:
    """
    Piecewise-linear interpolation over benchmark measurements.
    Input: number of tokens to prefill (uncached suffix only).
    Output: latency in microseconds.
    """
    def __init__(self, table_path: str):
        with open(table_path) as f:
            data = json.load(f)
        # data = {"tokens": [512, 1024, 2048, ...], "latency_us": [45000, 90000, ...]}
        self._tokens = np.array(data["tokens"], dtype=float)
        self._latency_us = np.array(data["latency_us"], dtype=float)
        assert np.all(np.diff(self._tokens) > 0), "token breakpoints must be strictly increasing"

    def prefill_latency_us(self, uncached_tokens: int) -> int:
        """
        Returns interpolated prefill latency.
        Clamps to table bounds (do not extrapolate below min or above max).
        """
        return int(np.interp(uncached_tokens, self._tokens, self._latency_us))
```

**Benchmark table format** (`benchmarks/latency_tables/prefill_70b_a100.json`):

```json
{
  "model": "llama3-70b",
  "gpu": "A100-80GB",
  "precision": "fp16",
  "note": "single-sequence prefill, no batching",
  "tokens":     [512,   1024,  2048,  4096,  8192,  16384, 32768],
  "latency_us": [45000, 90000, 200000, 520000, 1400000, 4800000, 17000000]
}
```

If no real measurements are available, populate from published benchmarks. The table must be present before any simulation run. Missing table = hard error, not a fallback to formula.

### 5.2 Decode oracle

```python
class DecodeOracle:
    """
    2D lookup: (output_tokens, n_active_sequences) -> latency_us.
    Decode throughput degrades nonlinearly with batch size due to
    memory bandwidth saturation.
    """
    def decode_latency_us(self, output_tokens: int, active_sequences: int) -> int:
        ...
```

### 5.3 Tier transfer time

```python
def transfer_time_us(
    size_bytes: int,
    tier: TierConfig,
) -> int:
    """
    T_trans = latency_floor_us + (size_bytes / bandwidth_bytes_per_s) * 1e6
    Formula is acceptable here (simple physics, validated by memcpy benchmark).
    """
    return int(tier.latency_floor_us + (size_bytes / tier.bandwidth_bytes_per_s) * 1_000_000)
```

### 5.4 Crossover check

```python
def is_cache_worthwhile(
    kv_bytes: int,
    tier: TierConfig,
    uncached_tokens: int,
    prefill_oracle: PrefillOracle,
) -> bool:
    """
    Returns True if restoring from cache is faster than recomputing.
    T_trans < T_recalc
    """
    t_trans = transfer_time_us(kv_bytes, tier)
    t_recalc = prefill_oracle.prefill_latency_us(uncached_tokens)
    return t_trans < t_recalc
```

### 5.5 Savings event classification

Log each cache hit as one of these four outcomes:

| Class | Condition | Meaning |
|---|---|---|
| `CACHE_HIT_L1` | Hit in HBM | Best case, no transfer |
| `CACHE_HIT_L2_WORTHWHILE` | `T_trans_pcie < T_recalc` | RAM offload paid off |
| `CACHE_HIT_L3A_WORTHWHILE` | `T_trans_nvme < T_recalc` | NVMe marginal win |
| `CACHE_HIT_L3A_BREAK_EVEN` | `T_trans_nvme >= T_recalc` | State preservation only, not a latency win |

The last class is the most important one to track: it quantifies how often cold storage is not saving compute, just preserving session state — a different value proposition.

---

## 6. Two-stage service model

```python
# sim/service.py

class ServiceModel:
    """
    Models GPU serving as two distinct resource pools with a queue between them.

        ARRIVAL -> [PrefillQueue] -> PrefillPool -> [DecodeQueue] -> DecodePool -> COMPLETE

    Cache savings reduce effective input length to the prefill pool only.
    Decode pool is unaffected by cache hits directly, but lower prefill latency
    reduces time-to-decode-queue, which reduces decode queue depth under load.
    """

    def __init__(self, config: ServiceConfig):
        self.prefill_slots_free = config.n_prefill_slots
        self.decode_slots_free = config.n_decode_slots
        self.prefill_queue: deque[Event] = deque(maxlen=config.prefill_queue_max)
        self.decode_queue: deque[Event] = deque(maxlen=config.decode_queue_max)

    def try_admit_prefill(self, event: Event) -> bool:
        """Returns True if a prefill slot was available and consumed."""
        ...

    def complete_prefill(self, event: Event) -> Optional[Event]:
        """
        Frees a prefill slot, returns a DECODE_START event if decode slot available.
        If decode queue is full, returns None (backpressure; prefill slot stays held).
        """
        ...

    def complete_decode(self, event: Event) -> None:
        """Frees decode slot; attempts to drain decode queue."""
        ...
```

**Important:** when `decode_queue` is full, completed prefills block and hold their prefill slot. This is the cascade that makes cache miss rate nonlinearly costly under saturation. Measure `prefill_slot_blocked_us` as a metric — it surfaces GPU saturation.

---

## 7. Workload synthesizer

```python
# sim/workload.py
import numpy as np
from scipy.stats import pareto, lognorm, weibull_min

class WorkloadSynthesizer:
    """
    Generates request arrival events from configured session profiles.
    Uses non-homogeneous Poisson process with sinusoidal diurnal rate.
    """

    def __init__(self, config: SimConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def diurnal_rate(self, time_s: float, profile: WorkloadProfile) -> float:
        """
        Rate function for NHPP.
        Sinusoidal with period 86400s, peak at business hours (offset ~32400s = 9 AM).
        peak_trough_ratio = 4.0 -> amplitude = (peak-trough)/2 = 1.5 * mean_rate
        """
        mean_rate = profile.arrival_rate_peak / profile.diurnal_peak_trough_ratio
        amplitude = mean_rate * (profile.diurnal_peak_trough_ratio - 1) / 2
        phase = (2 * np.pi * (time_s - 32400)) / 86400
        return mean_rate + amplitude * np.sin(phase)

    def sample_iat(self, profile: WorkloadProfile) -> float:
        """Inter-arrival time within a session, in seconds."""
        if profile.iat_dist == "exponential":
            return self.rng.exponential(profile.iat_mean_s)
        elif profile.iat_dist == "lognormal":
            sigma = 0.8  # fixed shape; tune if needed
            mu = np.log(profile.iat_mean_s) - 0.5 * sigma ** 2
            return float(lognorm.rvs(sigma, scale=np.exp(mu), random_state=self.rng))
        raise ValueError(f"Unknown iat_dist: {profile.iat_dist}")

    def sample_input_length(self, profile: WorkloadProfile) -> int:
        """Tokens, log-normal."""
        mu = np.log(profile.input_len_mean_tokens)
        sigma = profile.input_len_sigma_tokens / profile.input_len_mean_tokens
        return max(1, int(self.rng.lognormal(mu, sigma)))

    def sample_output_length(self, profile: WorkloadProfile) -> int:
        """Tokens, Pareto-tailed."""
        return int(pareto.rvs(
            profile.output_len_pareto_alpha,
            scale=profile.output_len_pareto_xmin,
            random_state=self.rng,
        ))

    def sample_session_duration(self, profile: WorkloadProfile) -> float:
        """
        Session lifetime in seconds.
        Uses Weibull survival function to model session churn/abandonment.
        k=0.8 -> decreasing hazard (sessions that last longer are more likely to continue)
        """
        if profile.session_duration_dist == "lognormal":
            return float(lognorm.rvs(
                0.5, scale=profile.session_duration_mean_s, random_state=self.rng
            ))
        return float(weibull_min.rvs(
            0.8, scale=profile.session_duration_mean_s, random_state=self.rng
        ))

    def prefix_stability(self, profile: WorkloadProfile, turn: int, total_turns: int) -> float:
        """
        Fraction of current context that is a stable cached prefix.
        Decays linearly from initial to final over the session.
        """
        if total_turns <= 1:
            return profile.prefix_stability_initial
        t = turn / (total_turns - 1)
        return (profile.prefix_stability_initial * (1 - t) +
                profile.prefix_stability_final * t)
```

**Session profiles** encode the four named workloads. Use these values as defaults in `configs/default.json`:

| Profile | `iat_mean_s` | `input_len_mean` | `output_pareto_alpha` | `shared_prefix_tokens` |
|---|---|---|---|---|
| Chat assistant | 45 | 150 | 2.0 | 2048 |
| Coding copilot | 360 | 800 | 1.5 | 2048 |
| Batch async | 0 (offline) | 1200 | 2.5 | 2048 |
| Agent/tool-use | 15 | 400 | 1.2 | 2048 |

---

## 8. Eviction engine

```python
# sim/eviction.py

class EvictionEngine:
    """
    Manages tier-to-tier movement of CacheObjects.

    Policy:
      L1 -> L2: triggered when L1 occupancy > eviction_hbm_threshold.
                Evict leaf nodes first (least shared).
                Among leaves, prefer ref_count==1 and oldest last_accessed_at.
                Never evict shared-prefix nodes (ref_count > 1) before private ones.

      L2 -> L3a: triggered by TTL (ttl_l2_s elapsed since last access)
                 OR when L2 occupancy > eviction_ram_threshold.
                 Write CacheObject to L3a; set is_hibernated=True.

      L3a cleanup: triggered when L3a occupancy > 0.90.
                   Evict by LRU. Log session_cold_evictions counter.
    """

    def evict_l1_to_l2(self, n_bytes_needed: int) -> list[str]:
        """
        Evict enough L1 CacheObjects to free n_bytes_needed.
        Returns list of evicted cache_keys.
        """
        ...

    def hibernate_l2_to_l3a(self, cache_key: str) -> None:
        """Move CacheObject from L2 to L3a storage."""
        ...

    def cleanup_l3a(self, n_bytes_needed: int) -> list[str]:
        """
        Evict LRU objects from L3a.
        Returns list of cache_keys permanently lost (logs session_cold_evictions).
        """
        ...
```

---

## 9. Metrics collector

```python
# sim/metrics.py
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Literal

SavingsClass = Literal[
    "CACHE_HIT_L1",
    "CACHE_HIT_L2_WORTHWHILE",
    "CACHE_HIT_L3A_WORTHWHILE",
    "CACHE_HIT_L3A_BREAK_EVEN",
    "COLD_MISS",
]

@dataclass
class MetricsCollector:
    # TTFT by tier source (microseconds, list for distribution)
    ttft_us: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    # Savings event counts
    savings_events: dict[SavingsClass, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # Recompute fraction distribution (0.0 = full hit, 1.0 = full miss)
    recompute_fraction: list[float] = field(default_factory=list)

    # Tier occupancy time-series (sampled at EPOCH_REPORT events)
    tier_occupancy_pct: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # GPU queue depth time-series
    prefill_queue_depth: list[int] = field(default_factory=list)
    decode_queue_depth: list[int] = field(default_factory=list)
    prefill_slot_blocked_us: int = 0  # cumulative microseconds prefill blocked

    # Sharing
    tokens_served_from_shared_prefix: int = 0
    total_tokens_served: int = 0

    # Memory pollution
    memory_pollution_bytes: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )  # tier name -> bytes wasted by block fragmentation

    # Eviction counts
    l1_to_l2_evictions: int = 0
    l2_to_l3a_evictions: int = 0
    session_cold_evictions: int = 0    # permanently lost L3a objects

    @property
    def sharing_factor(self) -> float:
        """tokens_served / unique_tokens_computed. >1.0 = cache is paying off."""
        if self.total_tokens_served == 0:
            return 1.0
        return self.total_tokens_served / max(
            1, self.total_tokens_served - self.tokens_served_from_shared_prefix
        )

    def ttft_percentiles(
        self, tier: str, percentiles: list[int] = [50, 95, 99]
    ) -> dict[int, float]:
        import numpy as np
        data = self.ttft_us.get(tier, [])
        if not data:
            return {p: float("nan") for p in percentiles}
        return {p: float(np.percentile(data, p)) for p in percentiles}

    def report(self) -> dict:
        """Produce final report dict. Authoritative outputs first, exploratory below."""
        ...
```

---

## 10. Discrete-event engine

```python
# sim/engine.py
import heapq
from sim.events import Event, EventType
from sim.config import SimConfig
from sim.cache import CacheObject, PrefixTrie
from sim.service import ServiceModel
from sim.workload import WorkloadSynthesizer
from sim.oracle import PrefillOracle, DecodeOracle
from sim.eviction import EvictionEngine
from sim.metrics import MetricsCollector

class SimEngine:
    """
    Main simulation loop. sim_clock is uint64 microseconds.
    Never read wall-clock time inside this class.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.sim_clock_us: int = 0
        self._event_seq: int = 0
        self._heap: list[tuple] = []

        rng = np.random.default_rng(config.seed)
        self.workload = WorkloadSynthesizer(config, rng)
        self.service = ServiceModel(config.service)
        self.metrics = MetricsCollector()
        self.eviction = EvictionEngine(config)
        self.prefill_oracle = PrefillOracle(...)  # load from benchmarks/
        self.decode_oracle = DecodeOracle(...)
        self.shared_prefix_trie = PrefixTrie()    # global singleton
        self.session_tries: dict[str, PrefixTrie] = {}

    def schedule(self, event: Event) -> None:
        assert event.time_us >= self.sim_clock_us, (
            f"Cannot schedule event in the past: {event.time_us} < {self.sim_clock_us}"
        )
        heapq.heappush(self._heap, (event.time_us, self._event_seq, event))
        self._event_seq += 1

    def run(self) -> MetricsCollector:
        end_us = int(self.config.sim_duration_s * 1_000_000)
        warmup_us = int(self.config.warmup_s * 1_000_000)

        # Seed initial arrivals
        self._seed_arrivals()

        while self._heap:
            _, _, event = heapq.heappop(self._heap)
            if event.time_us > end_us:
                break
            self.sim_clock_us = event.time_us
            collecting = self.sim_clock_us >= warmup_us
            self._dispatch(event, collecting)

        return self.metrics

    def _dispatch(self, event: Event, collecting: bool) -> None:
        handler = {
            EventType.REQUEST_ARRIVAL:  self._on_arrival,
            EventType.PREFILL_START:    self._on_prefill_start,
            EventType.PREFILL_COMPLETE: self._on_prefill_complete,
            EventType.DECODE_START:     self._on_decode_start,
            EventType.DECODE_COMPLETE:  self._on_decode_complete,
            EventType.TTL_FIRE:         self._on_ttl_fire,
            EventType.TIER_EVICTION:    self._on_tier_eviction,
            EventType.SESSION_RESUME:   self._on_session_resume,
            EventType.EPOCH_REPORT:     self._on_epoch_report,
        }[event.event_type]
        handler(event, collecting)
```

**Clock discipline:** `sim_clock_us` is `int`, never `float`. All arithmetic on times uses integer microseconds. If a latency value from an oracle is fractional, cast with `int(...)` before arithmetic. This prevents floating-point drift accumulating over long simulations.

---

## 11. Validation and invariant tests

All four invariant tests must pass before any exploratory run. They are cheap to run and catch most implementation bugs.

```python
# tests/test_invariants.py

def test_zero_ttl_collapses_l2():
    """
    With TTL=0, all L2 objects hibernate immediately.
    L2 occupancy must stay at 0 after warmup.
    """
    config = SimConfig.from_json("configs/default.json")
    config.ttl_l2_s = 0.0
    metrics = SimEngine(config).run()
    assert max(metrics.tier_occupancy_pct["L2"]) < 1.0  # <1% rounding tolerance

def test_infinite_l1_no_evictions():
    """
    With L1 capacity = total possible KV across all sessions,
    L1 eviction count must be 0.
    """
    config = SimConfig.from_json("configs/default.json")
    config.tiers[0].capacity_bytes = 2 ** 50  # effectively infinite
    metrics = SimEngine(config).run()
    assert metrics.l1_to_l2_evictions == 0

def test_no_shared_prefix_sharing_factor_one():
    """
    With all profiles having shared_system_prefix_tokens=0,
    sharing_factor must equal 1.0 (no cross-session reuse).
    """
    config = SimConfig.from_json("configs/default.json")
    for profile in config.profiles:
        profile.shared_system_prefix_tokens = 0
    metrics = SimEngine(config).run()
    assert abs(metrics.sharing_factor - 1.0) < 0.01

def test_zero_bandwidth_penalty_prefers_restore():
    """
    With all tier bandwidths set to sys.maxsize (effectively infinite),
    every L2 and L3a hit must be classified as WORTHWHILE (never BREAK_EVEN).
    """
    import sys
    config = SimConfig.from_json("configs/default.json")
    for tier in config.tiers:
        tier.bandwidth_bytes_per_s = sys.maxsize
        tier.latency_floor_us = 0
    metrics = SimEngine(config).run()
    assert metrics.savings_events["CACHE_HIT_L3A_BREAK_EVEN"] == 0
```

### Confidence tiers for output interpretation

| Output | Confidence | Rationale |
|---|---|---|
| KV size, block waste | **High** | Deterministic arithmetic |
| Tier transfer time | **High** | Simple physics; validate with memcpy benchmark (target: <10% error) |
| Cache hit rate, residency curves | **Medium** | Depends on workload model accuracy |
| Eviction rate | **Medium** | Sensitive to IAT distribution tail |
| TTFT p50 | **Medium-high** after oracle calibration | Prefill oracle removes largest uncertainty |
| TTFT p99 | **Lower** — needs real trace | Tail driven by rare outlier sessions |
| GPU queue depth under saturation | **Lower** | Requires accurate concurrent session count |

### Calibration checklist (run once before decision-making)

- [ ] `kv_size_bytes(1000, model_70b_fp16)` within 5% of torch.profiler trace
- [ ] `PrefillOracle.prefill_latency_us` within 20% at 3 sequence lengths vs. real hardware
- [ ] `transfer_time_us` within 10% of memcpy benchmark for L2 and L3a
- [ ] If production trace available: replay through simulator; check p50 TTFT within ±20%, p99 within ±40%
- [ ] All 4 invariant tests pass

---

## 12. Sensitivity sweep runner

```python
# scripts/sweep.py
"""
Vary one parameter at a time over ±50% of nominal value.
Runs are independent -> use multiprocessing.Pool.

Usage:
    python scripts/sweep.py --config configs/default.json --output results/sweep.json
"""
import multiprocessing
import itertools

SWEEP_PARAMS = {
    "ttl_l2_s":              [150, 225, 300, 375, 450],   # ±50% of 300s
    "tiers[1].capacity_bytes": [2e12, 3e12, 4e12, 5e12, 6e12],  # 2TB to 6TB
    "tiers[1].block_size_bytes": [8e6, 16e6, 32e6, 64e6],       # L2 block size
    "service.n_prefill_slots": [16, 24, 32, 48, 64],
}

TARGET_METRICS = ["ttft_p95", "ttft_p99", "gpu_utilization", "sharing_factor"]

def run_single(args):
    config_path, param_path, param_value = args
    config = SimConfig.from_json(config_path)
    # apply param_path (dot-notation) override to config
    set_nested(config, param_path, param_value)
    metrics = SimEngine(config).run()
    return {
        "param": param_path,
        "value": param_value,
        "metrics": metrics.report(),
    }

if __name__ == "__main__":
    jobs = [
        (args.config, param, value)
        for param, values in SWEEP_PARAMS.items()
        for value in values
    ]
    with multiprocessing.Pool() as pool:
        results = pool.map(run_single, jobs)
    # compute elasticity: d(metric)/d(param) normalized
    ...
```

---

## 13. Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.11"
numpy = "^1.26"
scipy = "^1.13"
polars = "^0.20"        # output trace files
marisa-trie = "^0.8"    # shared prefix pool (read-heavy)
pytest = "^8.0"

[tool.poetry.dev-dependencies]
pytest-benchmark = "*"
```

No torch dependency in the simulator itself. Torch is only needed when running the calibration benchmarks (separate script, not part of sim/).

---

## 14. v2 extensions (do not implement in v1)

These are out of scope for v1. The `SimConfig` has boolean flags to gate them. Both must default to `False` and raise `NotImplementedError` if set to `True` in v1.

**`enable_suffix_cache`**: Suffix caching requires a prompt template contract where stable document content is placed at a fixed token offset from the end of the context. Only then can a right-aligned token sequence be hashed and looked up. When enabled, implement a second radix trie rooted at the right end of the sequence. Track `suffix_hit_fraction` separately. Do not implement until the serving runtime confirms the prompt template constraint is enforced.

**`enable_l3b_object_store`**: Model S3-class storage with 500 MB/s bandwidth and ~50 ms latency floor. The crossover check will almost always return `BREAK_EVEN` at this tier — L3b's value is session continuity, not latency. Report `preservation_events` as a separate counter distinct from `savings_events`. Implement after v1 validation confirms the L3a model is working correctly.

---

## 15. Output contract

The final `metrics.report()` must return a dict with at minimum:

```python
{
  # Authoritative (report with confidence intervals from bootstrap)
  "tier_saturation_pct": {"L1": float, "L2": float, "L3A": float},
  "ttft_ms": {
    "L1_hit":    {"p50": float, "p95": float, "p99": float},
    "L2_hit":    {"p50": float, "p95": float, "p99": float},
    "L3A_hit":   {"p50": float, "p95": float, "p99": float},
    "cold_miss": {"p50": float, "p95": float, "p99": float},
  },
  "cache_hit_rate": {"L1": float, "L2": float, "L3A": float, "miss": float},
  "sharing_factor": float,

  # Exploratory (flag as lower confidence)
  "memory_pollution_gb": {"L1": float, "L2": float, "L3A": float},
  "eviction_rate_per_s": {"L1_to_L2": float, "L2_to_L3A": float},
  "session_cold_evictions": int,
  "savings_class_distribution": dict[str, float],  # fraction in each class
  "prefill_slot_blocked_pct": float,  # % of sim time prefill was backpressured

  # Metadata
  "config_run_id": str,
  "sim_duration_s": float,
  "warmup_s": float,
  "calibration_residuals": dict,  # from oracle validation
}
```

All floating-point values in the report must be rounded to 4 significant figures. Do not let raw float64 precision leak into output.
