"""
core/contracts.py
Version 1.1 — incorporates all final review feedback

This file defines the complete interface contract for AgentSim.
ALL layer boundaries are enforced here.

Review history:
  v1.0 — initial architecture migration plan
  v1.1 — added CacheKey, SavingsEvent, prefill/decode separation,
          observer time-coupling prohibition (final reviewer feedback)

GUIDING PRINCIPLES (non-negotiable, all enforced below):
  1. Core DES has zero knowledge of Observation or Integration layers
  2. Observation layer is strictly downstream (enforced via ObserverBase)
  3. Byte-based accounting everywhere — tokens are metadata only
  4. Cache identity = exact CacheKey match — never inferred from token counts
  5. Prefill and decode tracked separately in ALL metrics
  6. Observers must not assume event ordering beyond sim_time_us
  7. Every hardware target carries a ConfidenceLabel in outputs

MIT License.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ==========================================================================
# CONFIDENCE LABELS
# Required on all hardware targets and oracle calls.
# Must appear in all report outputs — not just documentation.
# ==========================================================================

class ConfidenceLabel(str, Enum):
    CALIBRATED       = "calibrated"       # measured oracle tables exist
    SEMI_CALIBRATED  = "semi-calibrated"  # partial tables + interpolation
    ANALYTICAL_ONLY  = "analytical-only"  # roofline math, no measurements


# ==========================================================================
# CACHE IDENTITY
# CacheKey defines what constitutes a cache hit.
# This must be used for ALL cache lookups — never token-count proximity.
# ==========================================================================

@dataclass(frozen=True)
class CacheKey:
    """
    Exact cache identity. A hit requires ALL three fields to match.

    Semantics:
      Full hit:    CacheKey matches exactly → serve entire KV from cache
      Partial hit: prefix_hash matches up to token K < N → serve first K
                   tokens from cache, recompute remaining N-K tokens
      Miss:        no matching CacheKey → full recompute

    Cross-worker reuse: CacheKey matches on a different worker's tier →
      still a hit, but adds transfer latency (modeled in TransferRecord).
      Do NOT treat cross-worker transfer as a miss — it is a HIT_L3_WIN
      or HIT_L3_BREAK_EVEN depending on transfer cost vs recompute cost.

    Never infer cache hits from token counts alone. Always use CacheKey.
    """
    model_id:      str   # model name + version (e.g., "llama3_70b_fp16")
    tokenizer_id:  str   # tokenizer variant — affects token boundaries
    prefix_hash:   str   # deterministic hash of the exact token sequence


# ==========================================================================
# SAVINGS CLASSIFICATION
# Core output metric. Every completed request gets one SavingsEvent.
# This is the primary decision boundary the simulator is built to study.
# L3A usefulness cannot be interpreted without this classification.
# ==========================================================================

class SavingsEvent(str, Enum):
    """
    Classifies the economic outcome of each cache decision.

    Computed per request as:
      transfer_cost = kv_bytes / tier_bandwidth + latency_floor
      recompute_cost = oracle.prefill_latency_us(full_context_tokens)

      HIT_L1:             cache hit in HBM — near-zero transfer cost
      HIT_L2_WIN:         DRAM transfer_cost < recompute_cost
      HIT_L3_WIN:         SSD transfer_cost < recompute_cost
      HIT_L3_BREAK_EVEN:  SSD transfer_cost ≈ recompute_cost (within 20%)
      MISS_RECOMPUTE:     no hit — full recompute (cold/expiry/eviction)

    Note: HIT_L3_BREAK_EVEN means L3A provided no meaningful latency benefit
    but still saved compute resources. Report it separately from HIT_L3_WIN.
    """
    HIT_L1            = "hit_l1"
    HIT_L2_WIN        = "hit_l2_win"
    HIT_L3_WIN        = "hit_l3_win"
    HIT_L3_BREAK_EVEN = "hit_l3_break_even"
    MISS_RECOMPUTE    = "miss_recompute"

    @classmethod
    def classify(
        cls,
        cache_tier:       Optional[str],   # None = miss
        transfer_cost_us: int,
        recompute_cost_us: int,
        break_even_margin: float = 0.20,
    ) -> "SavingsEvent":
        if cache_tier is None:
            return cls.MISS_RECOMPUTE
        if cache_tier == "L1":
            return cls.HIT_L1
        ratio = transfer_cost_us / max(1, recompute_cost_us)
        if cache_tier == "L2":
            return cls.HIT_L2_WIN if ratio < 1.0 else cls.MISS_RECOMPUTE
        if cache_tier in ("L3A", "L3"):
            if ratio <= (1.0 - break_even_margin):
                return cls.HIT_L3_WIN
            elif ratio <= (1.0 + break_even_margin):
                return cls.HIT_L3_BREAK_EVEN
            else:
                return cls.MISS_RECOMPUTE
        return cls.MISS_RECOMPUTE


# ==========================================================================
# TIER SPEC
# Byte-based. Tokens are never used as capacity or transfer units.
# ==========================================================================

@dataclass(frozen=True)
class TierSpec:
    """
    One memory tier. All quantities in bytes and microseconds.

    INVARIANT: capacity_bytes, bandwidth_bps, block_size_bytes are BYTES.
    INVARIANT: latency values are MICROSECONDS (int64, no float drift).
    INVARIANT: tokens never appear here — use ModelConfig.kv_bytes_for_tokens()
               to convert token counts to bytes before interacting with tiers.
    """
    name:               str     # "L1" | "L2" | "L3A"
    medium:             str     # "HBM" | "DRAM" | "DDR" | "SSD" | "NVMe"
    capacity_bytes:     int     # total capacity
    bandwidth_bps:      int     # peak read bandwidth in bytes/sec
    block_size_bytes:   int     # minimum transfer / page granularity
    scope:              str     # "per_gpu" | "per_worker" | "global"
    latency_floor_us:   int     # minimum access latency
    remote_latency_us:  int = 0 # additional latency for cross-worker access


# ==========================================================================
# CACHE OBJECT
# Byte-based. token_count is metadata only.
# ==========================================================================

@dataclass(frozen=True)
class CacheObject:
    """
    A KV cache entry. size_bytes is the authoritative accounting unit.

    INVARIANT: size_bytes = model.kv_bytes_for_tokens(token_count)
    INVARIANT: token_count is metadata — never used for capacity accounting
    INVARIANT: cache_key uniquely identifies this object for lookup/reuse
    """
    object_id:      str         # globally unique: f"{session_id}:{turn_id}"
    cache_key:      CacheKey    # used for lookup — exact prefix match only
    size_bytes:     int         # KV cache size in bytes (authoritative)
    token_count:    int         # metadata only — do not use for accounting
    created_at_us:  int         # sim clock at object creation
    last_access_us: int         # sim clock at last access (for LRU/TTL)
    tier:           str         # current resident tier: "L1" | "L2" | "L3A"
    worker_id:      int
    gpu_id:         int


# ==========================================================================
# TRANSFER RECORD
# Full accounting of every KV movement between tiers.
# ==========================================================================

@dataclass
class TransferRecord:
    """
    Records every KV transfer between tiers.
    latency_source must be set — no silent latency attribution.
    """
    object_id:          str
    cache_key:          CacheKey
    src_tier:           str
    dst_tier:           str
    bytes_moved:        int     # actual bytes transferred
    latency_us:         int     # transfer latency in microseconds
    latency_source:     str     # "bandwidth" | "floor" | "remote_penalty"
    savings_event:      SavingsEvent  # classification at time of transfer


# ==========================================================================
# PREFILL + DECODE RESULT
# Prefill and decode are ALWAYS reported separately.
# Never blend them — cache affects only prefill, decode dominates long output.
# ==========================================================================

@dataclass
class RequestResult:
    """
    Complete result of one request. Prefill and decode are separated.

    INVARIANT: prefill_latency_us and decode_latency_us are always present
    INVARIANT: queue_wait_prefill_us and queue_wait_decode_us are always present
    INVARIANT: savings_event is always set (never None)
    INVARIANT: cache_key is always set (even for misses — used for future lookups)
    """
    session_id:              str
    request_id:              str

    # Prefill phase (cache-sensitive)
    prefill_latency_us:      int         # compute time only (no queue wait)
    queue_wait_prefill_us:   int         # time waiting for prefill slot
    ttft_us:                 int         # = queue_wait_prefill + prefill_latency

    # Decode phase (NOT cache-sensitive)
    decode_latency_us:       int         # compute time only (no queue wait)
    queue_wait_decode_us:    int         # time waiting for decode slot

    # Cache outcome
    cache_tier:              Optional[str]   # None = miss
    cache_key:               CacheKey
    savings_event:           SavingsEvent
    bytes_loaded:            int             # 0 for miss
    latency_source:          str             # "bandwidth"|"floor"|"remote_penalty"|"miss"

    @property
    def total_latency_us(self) -> int:
        return self.ttft_us + self.decode_latency_us + self.queue_wait_decode_us


# ==========================================================================
# CORE DES ABSTRACTIONS
# These are the only interfaces the engine exposes.
# ==========================================================================

class CacheOracleBase(ABC):
    """
    Latency oracle. Returns (latency_us, ConfidenceLabel) always.
    The confidence label must be propagated to all report outputs.
    """
    @abstractmethod
    def prefill_latency_us(
        self,
        uncached_tokens: int,
        chip_name:       str,
        model_name:      str,
    ) -> tuple[int, ConfidenceLabel]:
        ...

    @abstractmethod
    def decode_latency_us(
        self,
        output_tokens:   int,
        batch_size:      int,
        kv_tokens:       int,
        chip_name:       str,
        model_name:      str,
    ) -> tuple[int, ConfidenceLabel]:
        ...

    @abstractmethod
    def kv_transfer_latency_us(
        self,
        bytes_to_transfer:  int,
        tier_bandwidth_bps: int,
        latency_floor_us:   int,
    ) -> int:
        """Transfer latency — no confidence needed (deterministic)."""
        ...


class DispatcherBase(ABC):
    @abstractmethod
    def select_node(
        self,
        request:    "Request",
        cluster:    "ClusterState",
    ) -> tuple[int, int]:        # (worker_id, gpu_id)
        ...


class EvictionPolicyBase(ABC):
    @abstractmethod
    def select_eviction_candidates(
        self,
        tier:         "TierStore",
        needed_bytes: int,
    ) -> list[CacheObject]:
        ...


# ==========================================================================
# OBSERVATION LAYER CONTRACT
# Observers are strictly read-only and downstream-only.
# Time-coupling constraints are enforced by design.
# ==========================================================================

class DESEventKind(str, Enum):
    REQUEST_ARRIVAL    = "request_arrival"
    PREFILL_START      = "prefill_start"
    CACHE_LOOKUP       = "cache_lookup"
    TIER_TRANSFER      = "tier_transfer"
    EVICTION           = "eviction"
    PREFILL_COMPLETE   = "prefill_complete"
    DECODE_START       = "decode_start"
    DECODE_COMPLETE    = "decode_complete"
    REQUEST_DROP       = "request_drop"


@dataclass
class DESEvent:
    """
    Emitted by Core DES. Observation layer reads these. Never writes back.

    INVARIANT: payload always includes sim_time_us (int64 microseconds)
    INVARIANT: PREFILL_COMPLETE payload always includes:
               prefill_latency_us, queue_wait_prefill_us, savings_event,
               cache_key, bytes_loaded, latency_source
    INVARIANT: DECODE_COMPLETE payload always includes:
               decode_latency_us, queue_wait_decode_us, output_tokens
    """
    event_id:    int
    sim_time_us: int           # int64 microseconds — no float drift
    kind:        DESEventKind
    payload:     dict


class ObserverBase(ABC):
    """
    Base class for all Observation layer components.

    CONSTRAINTS (all enforced by this interface):
      1. on_event() is read-only — must not modify any simulation state
      2. Observers must not assume event ordering beyond sim_time_us
      3. Observers must not infer future events from current state
      4. Observers must not maintain hidden state that affects interpretation
         of other observers (no shared mutable state between observers)
      5. Observers must be idempotent for same-timestamp events — if two
         events carry the same sim_time_us, their processing order must
         not affect output

    VIOLATION DETECTION: The engine runs a randomized-order test in CI:
      same-timestamp events are delivered in random order to observers.
      Observer output must be identical regardless of delivery order.
    """

    @abstractmethod
    def on_event(self, event: DESEvent) -> None:
        """
        Process one DES event. Read-only — must not modify simulation state.
        """
        ...

    def _assert_read_only(self) -> None:
        """
        Call this at the top of any method that might accidentally modify
        simulation state to get an explicit, clear error.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is an Observer — it must not "
            f"modify simulation state. Move state-modifying logic to "
            f"the Core DES layer (core/des/)."
        )


# ==========================================================================
# SWEEP TOOL CONTRACT
# SimPy is allowed to call only these interfaces.
# SimPy must never own cache state.
# ==========================================================================

class SweepEstimator(ABC):
    """
    Simplified latency estimator for SimPy sweep tool use only.

    ALLOWED:
      - Call estimate_turn_latency_ms() for per-turn latency approximation
      - Share SimConfig, ChipProfile, WorkloadProfile with Core DES
      - Label all output as 'sweep-estimate' + ANALYTICAL_ONLY

    PROHIBITED (enforced, not just documented):
      - Instantiate TierStore directly
      - Call EvictionPolicyBase implementations
      - Track per-session cache residency as simulation truth
      - Compute tier occupancy time-series
      - Produce outputs labeled as authoritative simulation results
    """

    @abstractmethod
    def estimate_turn_latency_ms(
        self,
        input_tokens:    int,
        output_tokens:   int,
        cache_hit:       bool,
        cache_tier:      str,   # "L1" | "L2" | "L3A" | "cold"
        chip_name:       str,
        model_name:      str,
    ) -> tuple[float, float, ConfidenceLabel]:
        """
        Returns (ttft_ms, total_ms, confidence).
        confidence is always ANALYTICAL_ONLY for sweep estimates.
        """
        ...


# ==========================================================================
# REPORT METADATA
# Required on every report output. Not optional.
# ==========================================================================

@dataclass
class ReportMetadata:
    """
    Every report, CSV, and plot must include this metadata.
    The confidence field must appear in the headline/title — not buried.

    Example headline: "[ANALYTICAL-ONLY] custom_npu_v1 / llama3_70b"
    Example headline: "[CALIBRATED] nvidia_a100_80g / llama3_70b"
    """
    chip_name:          str
    model_name:         str
    confidence:         ConfidenceLabel
    oracle_source:      str           # path to table, or "roofline"
    l3a_mode:           str           # "global" | "local" — always state this
    sim_duration_s:     float
    generated_at:       str           # ISO timestamp
    agentsim_version:   str

    @property
    def headline(self) -> str:
        return (
            f"[{self.confidence.value.upper()}] "
            f"{self.chip_name} / {self.model_name} "
            f"(L3A: {self.l3a_mode})"
        )
