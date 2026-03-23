from __future__ import annotations
import json
import numpy as np

from sim.config import TierConfig


class PrefillOracle:
    """
    Piecewise-linear interpolation over benchmark measurements.
    Input: number of tokens to prefill (uncached suffix only).
    Output: latency in microseconds.
    """

    def __init__(self, table_path: str):
        with open(table_path) as f:
            data = json.load(f)
        self._tokens = np.array(data["tokens"], dtype=float)
        self._latency_us = np.array(data["latency_us"], dtype=float)
        assert np.all(np.diff(self._tokens) > 0), "token breakpoints must be strictly increasing"

    def prefill_latency_us(self, uncached_tokens: int) -> int:
        """Returns interpolated prefill latency. Clamps to table bounds."""
        return int(np.interp(uncached_tokens, self._tokens, self._latency_us))


class DecodeOracle:
    """
    2D lookup: (output_tokens, n_active_sequences) -> latency_us.
    Decode throughput degrades nonlinearly with batch size due to
    memory bandwidth saturation.
    """

    # Base per-token decode latency (us) for single sequence on A100
    BASE_TOKEN_LATENCY_US = 30

    def decode_latency_us(self, output_tokens: int, active_sequences: int) -> int:
        """
        Model: per-token latency scales with sqrt of active sequences
        (memory bandwidth sharing). Total = tokens * per_token_latency.
        """
        batch_factor = max(1.0, np.sqrt(active_sequences))
        per_token = self.BASE_TOKEN_LATENCY_US * batch_factor
        return int(output_tokens * per_token)


def transfer_time_us(size_bytes: int, tier: TierConfig) -> int:
    """
    T_trans = latency_floor_us + (size_bytes / bandwidth_bytes_per_s) * 1e6
    """
    return int(tier.latency_floor_us + (size_bytes / tier.bandwidth_bytes_per_s) * 1_000_000)


def kv_transfer_time_us(kv_bytes: int, bandwidth_bps: int, latency_floor_us: int) -> int:
    """Time to transfer KV cache from prefill node to decode node (e.g., via RDMA)."""
    return latency_floor_us + int((kv_bytes / bandwidth_bps) * 1_000_000)


def is_cache_worthwhile(
    kv_bytes: int,
    tier: TierConfig,
    uncached_tokens: int,
    prefill_oracle: PrefillOracle,
) -> bool:
    """Returns True if restoring from cache is faster than recomputing."""
    t_trans = transfer_time_us(kv_bytes, tier)
    t_recalc = prefill_oracle.prefill_latency_us(uncached_tokens)
    return t_trans < t_recalc
