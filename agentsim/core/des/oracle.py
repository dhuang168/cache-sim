"""
core/des/oracle.py

Prefill and decode latency oracle for the Core DES engine.

Two modes:
  1. PiecewiseOracle  — piecewise-linear interpolation from measured tables
                        ConfidenceLabel.CALIBRATED or SEMI_CALIBRATED
  2. RooflineOracle   — analytical roofline fallback
                        ConfidenceLabel.ANALYTICAL_ONLY

Every call returns (latency_us: int, confidence: ConfidenceLabel).
The confidence label must appear in all output reports.

Replaces the HardwareModel roofline in core/hardware_model.py.
That file is now superseded — see integration/chips/profiles.py for
ChipProfile and ModelConfig.

MIT License.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import json
import math
import os

import numpy as np
from agentsim.core.contracts import ConfidenceLabel, CacheOracleBase


# ---------------------------------------------------------------------------
# Piecewise oracle — primary oracle for calibrated targets
# ---------------------------------------------------------------------------

class PiecewiseOracle(CacheOracleBase):
    """
    Piecewise-linear interpolation from measured benchmark tables.
    Used for CALIBRATED and SEMI_CALIBRATED hardware targets.

    Table format (JSON):
    {
      "metadata": {
        "chip": "nvidia_a100_80g",
        "model": "llama3_70b_fp16",
        "source": "measured",          # "measured" | "interpolated"
        "confidence": "calibrated"
      },
      "points": [
        {"tokens": 512,    "latency_us": 45000},
        {"tokens": 1024,   "latency_us": 112000},
        ...
        {"tokens": 100000, "latency_us": 120700000}
      ],
      "extrapolation": "quadratic"    # used beyond the last measured point
    }

    Data from cache-sim prototype benchmarks/latency_tables/:
      10k tokens → 2.2s, 30k → 14.9s, 50k → 37.0s, 100k → 120.7s
    """

    def __init__(self, table_path: str):
        with open(table_path) as f:
            raw = json.load(f)

        self.metadata    = raw["metadata"]
        self.chip_name   = self.metadata["chip"]
        self.model_name  = self.metadata["model"]
        self.confidence  = ConfidenceLabel(self.metadata["confidence"])
        self.extrap      = raw.get("extrapolation", "quadratic")

        points = raw["points"]
        points.sort(key=lambda p: p["tokens"])
        self._tokens  = [p["tokens"]     for p in points]
        self._latency = [p["latency_us"] for p in points]

    def prefill_latency_us(
        self,
        uncached_tokens: int,
        chip_name:       str,
        model_name:      str,
    ) -> tuple[int, ConfidenceLabel]:

        if uncached_tokens <= 0:
            return 0, self.confidence

        lat = self._interpolate(uncached_tokens)
        return int(lat), self.confidence

    def decode_latency_us(
        self,
        output_tokens:  int,
        batch_size:     int,
        kv_tokens:      int,
        chip_name:      str,
        model_name:     str,
    ) -> tuple[int, ConfidenceLabel]:
        # Decode uses the sqrt(batch) degradation model from cache-sim
        # Each step latency ≈ base_step × sqrt(active_sequences)
        # base_step is derived from the oracle table at single-token prefill
        base_step_us, _ = self.prefill_latency_us(1, chip_name, model_name)
        total_us = int(base_step_us * output_tokens * math.sqrt(max(1, batch_size)))
        return total_us, self.confidence

    def kv_transfer_latency_us(
        self,
        bytes_to_transfer:  int,
        tier_bandwidth_bps: int,
        latency_floor_us:   int,
    ) -> int:
        transfer_us = int(bytes_to_transfer / tier_bandwidth_bps * 1_000_000)
        return max(latency_floor_us, transfer_us)

    def _interpolate(self, tokens: int) -> float:
        xs = self._tokens
        ys = self._latency

        # Below minimum measured point — linear extrapolation down
        if tokens <= xs[0]:
            ratio = tokens / xs[0]
            return ys[0] * ratio

        # Within range — piecewise linear
        for i in range(len(xs) - 1):
            if xs[i] <= tokens <= xs[i + 1]:
                t = (tokens - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + t * (ys[i + 1] - ys[i])

        # Beyond maximum — quadratic or linear extrapolation
        if self.extrap == "quadratic":
            # O(n²) scaling anchored to last measured point
            last_n = xs[-1]
            last_t = ys[-1]
            return last_t * (tokens / last_n) ** 2
        else:
            # Linear extrapolation from last two points
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return ys[-1] + slope * (tokens - xs[-1])


# ---------------------------------------------------------------------------
# Roofline oracle — analytical fallback for ANALYTICAL_ONLY targets
# ---------------------------------------------------------------------------

class RooflineOracle(CacheOracleBase):
    """
    Analytical roofline model.
    Used when no measured oracle table exists for the target chip.

    CONFIDENCE: ANALYTICAL_ONLY — always.
    Results from this oracle must be labeled clearly in all outputs.
    Do not compare RooflineOracle results directly with PiecewiseOracle
    results without acknowledging the confidence difference.

    This was previously HardwareModel in core/hardware_model.py.
    It is now explicitly a fallback, not the primary oracle.
    """

    COMPUTE_EFFICIENCY = 0.55
    MEMORY_EFFICIENCY  = 0.85

    def __init__(self, chip: ChipProfile, model: ModelConfig):
        self.chip  = chip
        self.model = model

    def prefill_latency_us(
        self,
        uncached_tokens: int,
        chip_name:       str,
        model_name:      str,
    ) -> tuple[int, ConfidenceLabel]:

        if uncached_tokens <= 0:
            return 0, ConfidenceLabel.ANALYTICAL_ONLY

        # Attention FLOPs
        attn_flops = 4 * uncached_tokens * uncached_tokens * self.model.hidden_dim
        ffn_flops  = 8 * uncached_tokens * self.model.hidden_dim
        total_flops = (attn_flops + ffn_flops) * self.model.num_layers

        compute_tflops = self.chip.compute_tflops_bf16 * 1e12
        compute_s = total_flops / (compute_tflops * self.COMPUTE_EFFICIENCY)

        # Memory: model weight bytes touched
        l1 = self.chip.l1
        if l1 is None:
            raise ValueError(f"Chip {self.chip.name} has no L1 tier")
        bw = l1.bandwidth_bps * self.MEMORY_EFFICIENCY
        weight_bytes = (
            2 * self.model.hidden_dim ** 2
            * self.model.num_layers
            * self.model.dtype_bytes
        )
        memory_s = weight_bytes / bw

        prefill_s = max(compute_s, memory_s)
        return int(prefill_s * 1_000_000), ConfidenceLabel.ANALYTICAL_ONLY

    def decode_latency_us(
        self,
        output_tokens:  int,
        batch_size:     int,
        kv_tokens:      int,
        chip_name:      str,
        model_name:     str,
    ) -> tuple[int, ConfidenceLabel]:

        l1 = self.chip.l1
        if l1 is None:
            raise ValueError(f"Chip {self.chip.name} has no L1 tier")

        bw = l1.bandwidth_bps * self.MEMORY_EFFICIENCY
        kv_bytes_total = self.model.kv_bytes_for_tokens(kv_tokens)
        weight_bytes = (
            2 * self.model.hidden_dim ** 2
            * self.model.num_layers
            * self.model.dtype_bytes
            * batch_size
        )
        bytes_per_step = weight_bytes + kv_bytes_total
        step_s  = bytes_per_step / bw
        total_s = step_s * output_tokens

        return int(total_s * 1_000_000), ConfidenceLabel.ANALYTICAL_ONLY

    def kv_transfer_latency_us(
        self,
        bytes_to_transfer:  int,
        tier_bandwidth_bps: int,
        latency_floor_us:   int,
    ) -> int:
        bw = tier_bandwidth_bps * self.MEMORY_EFFICIENCY
        transfer_us = int(bytes_to_transfer / bw * 1_000_000)
        return max(latency_floor_us, transfer_us)


# ---------------------------------------------------------------------------
# OracleFactory — selects the right oracle for a chip
# ---------------------------------------------------------------------------

class OracleFactory:
    """
    Returns the appropriate oracle for a given chip profile.
    If the chip has a calibrated oracle table → PiecewiseOracle.
    Otherwise → RooflineOracle (analytical-only).

    The confidence label is always available on the returned oracle's
    first call return value. It must be included in all report outputs.
    """

    def __init__(self, oracle_root: str = "benchmarks/oracle_tables"):
        self.oracle_root = oracle_root

    def get(self, chip: ChipProfile, model: ModelConfig) -> CacheOracleBase:
        if chip.oracle_table is not None:
            table_path = os.path.join(self.oracle_root, chip.oracle_table) \
                         if not os.path.isabs(chip.oracle_table) \
                         else chip.oracle_table
            if os.path.exists(table_path):
                return PiecewiseOracle(table_path)

        # No table — fall back to roofline with ANALYTICAL_ONLY label
        return RooflineOracle(chip, model)


# ---------------------------------------------------------------------------
# Simple oracles ported from sim/oracle.py (backward-compatible interface)
# These are used by the ported engine for direct compatibility.
# ---------------------------------------------------------------------------

class SimplePrefillOracle:
    """Piecewise-linear prefill oracle using numpy interp (from sim/oracle.py)."""

    def __init__(self, table_path: str):
        with open(table_path) as f:
            data = json.load(f)
        self._tokens = np.array(data["tokens"], dtype=float)
        self._latency_us = np.array(data["latency_us"], dtype=float)
        assert np.all(np.diff(self._tokens) > 0)

    def prefill_latency_us(self, uncached_tokens: int) -> int:
        return int(np.interp(uncached_tokens, self._tokens, self._latency_us))


class SimpleDecodeOracle:
    """Sqrt batch degradation decode model (from sim/oracle.py)."""
    BASE_TOKEN_LATENCY_US = 30

    def decode_latency_us(self, output_tokens: int, active_sequences: int) -> int:
        batch_factor = max(1.0, np.sqrt(active_sequences))
        per_token = self.BASE_TOKEN_LATENCY_US * batch_factor
        return int(output_tokens * per_token)


# ---------------------------------------------------------------------------
# Free functions (from sim/oracle.py) — used by engine directly
# ---------------------------------------------------------------------------

def transfer_time_us(size_bytes: int, tier) -> int:
    """T_trans = latency_floor_us + (size_bytes / bandwidth_bytes_per_s) * 1e6"""
    return int(tier.latency_floor_us + (size_bytes / tier.bandwidth_bytes_per_s) * 1_000_000)


def kv_transfer_time_us(kv_bytes: int, bandwidth_bps: int, latency_floor_us: int) -> int:
    """Time to transfer KV cache from prefill node to decode node."""
    return latency_floor_us + int((kv_bytes / bandwidth_bps) * 1_000_000)


def is_cache_worthwhile(kv_bytes: int, tier, uncached_tokens: int, prefill_oracle) -> bool:
    """Returns True if restoring from cache is faster than recomputing."""
    t_trans = transfer_time_us(kv_bytes, tier)
    t_recalc = prefill_oracle.prefill_latency_us(uncached_tokens)
    return t_trans < t_recalc
