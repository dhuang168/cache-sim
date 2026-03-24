"""
sweep/request_sweep.py

⚠️  SWEEP TOOL ONLY — NOT AUTHORITATIVE SIMULATION
    All outputs from this module are labeled "sweep-estimate".
    Do not use these results as final simulation outputs.
    Run core/des/engine.py for authoritative results.

This module is the demoted SimPy request-level simulator.
It was previously sim/request_sim.py and positioned as the primary
simulation engine. Under the final plan, it is a fast parameter
screening tool only.

CONSTRAINTS (enforced, not just documented):
  - SimPy CANNOT own cache state (no TierStore instantiation)
  - SimPy CANNOT run eviction logic
  - SimPy CANNOT compute tier occupancy over time
  - SimPy MUST share SimConfig and WorkloadProfile with Core DES
  - SimPy MUST label all output as "sweep-estimate"
  - SimPy calls SweepEstimator for latency (simplified, not stateful)

Typical workflow:
  1. Run sweep over 100s of configs → identify interesting region
  2. Run Core DES on top-N configs → authoritative results
  3. Compare sweep vs DES to validate sweep accuracy

MIT License.
"""

from __future__ import annotations
import simpy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import random

from agentsim.integration.chips.profiles import (
    ChipProfile, ModelConfig, ConfidenceLabel
)


# ---------------------------------------------------------------------------
# SweepEstimator ABC — the only latency interface SimPy may call
# ---------------------------------------------------------------------------

class SweepEstimator(ABC):
    """
    Simplified latency estimator for sweep tool use.

    What this IS:
      - A fast approximation for parameter screening
      - Stateless per-call (no tier occupancy tracking)

    What this IS NOT:
      - The authoritative simulation oracle
      - A replacement for core/des/oracle.py
      - A cache state machine

    SimPy MUST NOT:
      - Instantiate TierStore directly
      - Call EvictionPolicyBase implementations
      - Track per-session cache residency as simulation truth
    """

    @abstractmethod
    def estimate_turn_latency_ms(
        self,
        input_tokens:    int,
        output_tokens:   int,
        cache_hit:       bool,
        cache_tier:      str,   # "L1" | "L2" | "L3A" | "cold"
        chip:            ChipProfile,
        model:           ModelConfig,
    ) -> tuple[float, float, ConfidenceLabel]:
        """
        Returns (ttft_ms, total_ms, confidence).
        confidence is always ANALYTICAL_ONLY for sweep estimates.
        """
        ...


class SimpleRooflineSweepEstimator(SweepEstimator):
    """
    Simplified roofline estimator for sweep use.
    Does not track state. Does not consult oracle tables.
    Always returns ConfidenceLabel.ANALYTICAL_ONLY.
    """

    MEM_EFF  = 0.85
    COMP_EFF = 0.55

    # Cache tier transfer time multipliers (relative to L1)
    TIER_LATENCY_FACTOR = {
        "L1":   1.0,
        "L2":   3.0,    # DRAM is ~3× slower than HBM for KV load
        "L3A":  50.0,   # SSD is ~50× slower
        "cold": 100.0,  # full recompute cost proxy
    }

    def estimate_turn_latency_ms(
        self,
        input_tokens:    int,
        output_tokens:   int,
        cache_hit:       bool,
        cache_tier:      str,
        chip:            ChipProfile,
        model:           ModelConfig,
    ) -> tuple[float, float, ConfidenceLabel]:

        l1 = chip.l1
        if l1 is None:
            raise ValueError(f"Chip {chip.name} has no L1 tier")

        bw = l1.bandwidth_bps * self.MEM_EFF

        # Effective uncached tokens
        if cache_hit and cache_tier in ("L1", "L2"):
            uncached = max(0, int(input_tokens * 0.10))  # ~10% structural recompute
        elif cache_hit and cache_tier == "L3A":
            uncached = max(0, int(input_tokens * 0.10))
        else:
            uncached = input_tokens  # cold miss — full recompute

        # Prefill: O(n²) attention dominant at large contexts
        prefill_flops = 4 * uncached * uncached * model.hidden_dim * model.num_layers
        compute_s     = prefill_flops / (chip.compute_tflops_bf16 * 1e12 * self.COMP_EFF)
        mem_s         = (model.total_bytes_per_token_kv * uncached) / bw
        prefill_s     = max(compute_s, mem_s)

        # KV load time (if cache hit)
        factor = self.TIER_LATENCY_FACTOR.get(cache_tier, 100.0)
        kv_load_s = (model.kv_bytes_for_tokens(input_tokens - uncached) / bw) * factor \
                    if cache_hit else 0.0

        ttft_ms = (prefill_s + kv_load_s) * 1000

        # Decode: memory-bandwidth bound
        decode_step_bytes = model.kv_bytes_for_tokens(input_tokens) + \
                            2 * model.hidden_dim ** 2 * model.num_layers * model.dtype_bytes
        decode_s = (decode_step_bytes / bw) * output_tokens
        total_ms = ttft_ms + decode_s * 1000

        return ttft_ms, total_ms, ConfidenceLabel.ANALYTICAL_ONLY


# ---------------------------------------------------------------------------
# Simplified cache state for sweep — NOT authoritative
# This is a probabilistic estimator, not a real cache manager.
# It does NOT track evictions, tier pressure, or object residency.
# ---------------------------------------------------------------------------

@dataclass
class SweepCacheEstimate:
    """
    Probabilistic cache state for sweep tool.
    Does NOT implement eviction or tier accounting.
    Only tracks session last-access time for TTL estimation.
    """
    session_id:       str
    last_access_ms:   float = 0.0
    hit_probability:  float = 0.95  # assumed steady-state hit probability

    ANTHROPIC_TTL_MS = 300_000.0  # 5 minutes

    def estimate_hit(
        self,
        sim_time_ms:   float,
        rng:           random.Random,
    ) -> tuple[bool, str]:
        """
        Returns (cache_hit, tier).
        Probabilistic — not a real cache lookup.
        """
        age_ms = sim_time_ms - self.last_access_ms

        # TTL expiry
        if age_ms >= self.ANTHROPIC_TTL_MS:
            self.last_access_ms = sim_time_ms
            return False, "cold"

        self.last_access_ms = sim_time_ms

        if rng.random() < self.hit_probability:
            # Assume L1 hit for fast turns, L2 for slower
            tier = "L1" if age_ms < 30_000 else "L2"
            return True, tier
        return False, "cold"


# ---------------------------------------------------------------------------
# Sweep metrics — labeled sweep-estimate, never authoritative
# ---------------------------------------------------------------------------

@dataclass
class SweepMetrics:
    source:        str = "sweep-estimate"  # always — identifies non-authoritative
    confidence:    ConfidenceLabel = ConfidenceLabel.ANALYTICAL_ONLY

    ttft_ms_list:  list[float] = field(default_factory=list)
    total_ms_list: list[float] = field(default_factory=list)
    cache_hits:    int = 0
    cache_misses:  int = 0
    expiry_events: int = 0

    def record_turn(
        self,
        ttft_ms:   float,
        total_ms:  float,
        hit:       bool,
        expired:   bool,
    ) -> None:
        self.ttft_ms_list.append(ttft_ms)
        self.total_ms_list.append(total_ms)
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        if expired:
            self.expiry_events += 1

    def summary(self) -> dict:
        n = len(self.ttft_ms_list)
        if n == 0:
            return {"source": self.source, "confidence": self.confidence, "n": 0}

        sorted_ttft = sorted(self.ttft_ms_list)
        return {
            "source":       self.source,
            "confidence":   self.confidence.value,
            "n_turns":      n,
            "hit_rate":     self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            "expiry_rate":  self.expiry_events / max(1, n),
            "ttft_p50_ms":  sorted_ttft[int(n * 0.50)],
            "ttft_p90_ms":  sorted_ttft[int(n * 0.90)],
            "ttft_p99_ms":  sorted_ttft[int(n * 0.99)],
            "ttft_mean_ms": sum(self.ttft_ms_list) / n,
        }


# ---------------------------------------------------------------------------
# SimPy sweep runner
# ---------------------------------------------------------------------------

class RequestSweep:
    """
    SimPy-based fast parameter sweep.

    ⚠️  SWEEP TOOL — outputs are "sweep-estimate", not authoritative.

    Use this to:
      - Screen 100s of (chip, workload, policy) combinations quickly
      - Identify configurations worth running through Core DES
      - Get directional hit-rate and TTFT estimates in seconds

    Do NOT use this to:
      - Publish simulation results
      - Compare framework policies with precision
      - Study eviction behavior or tier occupancy
    """

    def __init__(
        self,
        chip:           ChipProfile,
        model:          ModelConfig,
        estimator:      Optional[SweepEstimator] = None,
        num_sessions:   int   = 200,
        arrival_rate_qps: float = 1.0,
        max_sim_time_ms: float = 3_600_000,
        seed:           int   = 42,
    ):
        self.chip        = chip
        self.model       = model
        self.estimator   = estimator or SimpleRooflineSweepEstimator()
        self.num_sessions = num_sessions
        self.arrival_rate = arrival_rate_qps
        self.max_sim_time = max_sim_time_ms
        self.rng         = random.Random(seed)
        self.metrics     = SweepMetrics()

    def run(self) -> SweepMetrics:
        env = simpy.Environment()
        env.process(self._arrival_loop(env))
        env.run(until=self.max_sim_time)
        return self.metrics

    def _arrival_loop(self, env):
        for _ in range(self.num_sessions):
            env.process(self._run_session(env))
            gap_ms = self.rng.expovariate(self.arrival_rate) * 1000
            yield env.timeout(gap_ms)

    def _run_session(self, env):
        session_id = f"s{self.rng.randint(0, 999999):06d}"
        cache_est  = SweepCacheEstimate(session_id=session_id)
        num_turns  = self.rng.randint(3, 20)

        # Grow input tokens each turn (coding workload pattern)
        input_tokens = self.rng.randint(20_000, 40_000)

        for _ in range(num_turns):
            sim_time = env.now
            hit, tier = cache_est.estimate_hit(sim_time, self.rng)
            expired   = (not hit) and (sim_time - cache_est.last_access_ms >= SweepCacheEstimate.ANTHROPIC_TTL_MS)

            output_tokens = self.rng.randint(200, 1500)

            ttft_ms, total_ms, _ = self.estimator.estimate_turn_latency_ms(
                input_tokens  = input_tokens,
                output_tokens = output_tokens,
                cache_hit     = hit,
                cache_tier    = tier,
                chip          = self.chip,
                model         = self.model,
            )

            self.metrics.record_turn(ttft_ms, total_ms, hit, expired)

            yield env.timeout(total_ms)

            # Think time
            think_ms = self.rng.lognormvariate(math.log(45), 1.4) * 1000
            yield env.timeout(think_ms)

            # Grow context each turn
            input_tokens += self.rng.randint(2000, 10000)


import math  # needed for lognormvariate call above
