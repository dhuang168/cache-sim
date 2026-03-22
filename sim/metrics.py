from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Literal

import numpy as np

SavingsClass = Literal[
    "CACHE_HIT_L1",
    "CACHE_HIT_L2_WORTHWHILE",
    "CACHE_HIT_L3A_WORTHWHILE",
    "CACHE_HIT_L3A_BREAK_EVEN",
    "COLD_MISS",
]


def _sig_round(x: float, sig: int = 4) -> float:
    """Round to N significant figures."""
    if x == 0 or np.isnan(x) or np.isinf(x):
        return float(x)
    precision = sig - 1 - int(np.floor(np.log10(abs(x))))
    if precision < 0:
        precision = 0
    return round(float(x), precision)


@dataclass
class MetricsCollector:
    # TTFT by tier source (microseconds)
    ttft_us: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    # Savings event counts
    savings_events: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Recompute fraction distribution
    recompute_fraction: list[float] = field(default_factory=list)

    # Tier occupancy time-series (sampled at EPOCH_REPORT events)
    tier_occupancy_pct: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # GPU queue depth time-series
    prefill_queue_depth: list[int] = field(default_factory=list)
    decode_queue_depth: list[int] = field(default_factory=list)
    prefill_slot_blocked_us: int = 0

    # Queue wait time (microseconds) — time from entering pending queue to slot obtained
    queue_wait_us: list[int] = field(default_factory=list)

    # Prefill duration (microseconds) — actual prefill compute time per request
    prefill_duration_us: list[int] = field(default_factory=list)

    # Latency impact metrics (time-series, sampled at epoch)
    slot_utilization_pct: list[float] = field(default_factory=list)
    l3a_object_count: list[int] = field(default_factory=list)
    cold_evictions_per_epoch: list[int] = field(default_factory=list)

    # Sharing
    tokens_served_from_shared_prefix: int = 0
    total_tokens_served: int = 0

    # Memory pollution
    memory_pollution_bytes: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # Eviction counts
    l1_to_l2_evictions: int = 0          # pressure-driven (occupancy or placement)
    l1_to_l2_ttl_migrations: int = 0     # TTL-driven tier migration (not pressure)
    l2_to_l3a_evictions: int = 0
    session_cold_evictions: int = 0

    # Multi-node metrics
    per_node_queue_depth: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    per_node_l1_occupancy_pct: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    per_node_l2_occupancy_pct: dict[int, list[float]] = field(default_factory=lambda: defaultdict(list))
    per_node_prefill_count: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    affinity_dispatches: int = 0
    non_affinity_dispatches: int = 0
    cross_node_transfers: int = 0

    # Block sharing metrics
    shared_block_groups: int = 0
    shared_block_memory_saved_bytes: int = 0
    shared_block_ref_count_max: int = 0

    # Cross-worker duplication metrics (sampled at epoch)
    duplicate_block_bytes: list[int] = field(default_factory=list)  # redundant bytes across workers per epoch
    max_replication_factor: list[int] = field(default_factory=list)  # most-replicated object per epoch
    shared_prefix_worker_distribution: dict[str, int] = field(default_factory=dict)  # cache_key → n_workers having a copy

    # Global L3A bandwidth contention
    l3a_concurrent_reads: list[int] = field(default_factory=list)  # concurrent L3A reads per epoch
    l3a_bandwidth_contention_events: int = 0  # times contention factor > 1

    # Total sim time for rate calculations
    effective_sim_us: int = 0

    @property
    def sharing_factor(self) -> float:
        if self.total_tokens_served == 0:
            return 1.0
        unique = self.total_tokens_served - self.tokens_served_from_shared_prefix
        return self.total_tokens_served / max(1, unique)

    def ttft_percentiles(
        self, tier: str, percentiles: list[int] | None = None
    ) -> dict[int, float]:
        if percentiles is None:
            percentiles = [50, 95, 99]
        data = self.ttft_us.get(tier, [])
        if not data:
            return {p: float("nan") for p in percentiles}
        return {p: float(np.percentile(data, p)) for p in percentiles}

    def report(self) -> dict:
        """Produce final report dict."""
        r = _sig_round

        # Tier saturation
        tier_sat = {}
        for tier_name in ["L1", "L2", "L3A"]:
            occ = self.tier_occupancy_pct.get(tier_name, [])
            tier_sat[tier_name] = r(np.mean(occ) if occ else 0.0)

        # TTFT distributions
        ttft_ms = {}
        for source in ["L1_hit", "L2_hit", "L3A_hit", "cold_miss"]:
            pcts = self.ttft_percentiles(source)
            ttft_ms[source] = {
                f"p{p}": r(v / 1000.0) for p, v in pcts.items()
            }

        # Cache hit rate
        total_events = sum(self.savings_events.values())
        hit_rate = {}
        if total_events > 0:
            hit_rate["L1"] = r(self.savings_events.get("CACHE_HIT_L1", 0) / total_events)
            l2_hits = self.savings_events.get("CACHE_HIT_L2_WORTHWHILE", 0)
            hit_rate["L2"] = r(l2_hits / total_events)
            l3a_hits = (
                self.savings_events.get("CACHE_HIT_L3A_WORTHWHILE", 0)
                + self.savings_events.get("CACHE_HIT_L3A_BREAK_EVEN", 0)
            )
            hit_rate["L3A"] = r(l3a_hits / total_events)
            hit_rate["miss"] = r(self.savings_events.get("COLD_MISS", 0) / total_events)
        else:
            hit_rate = {"L1": 0.0, "L2": 0.0, "L3A": 0.0, "miss": 1.0}

        # Savings class distribution
        savings_dist = {}
        if total_events > 0:
            for cls, count in self.savings_events.items():
                savings_dist[cls] = r(count / total_events)

        # Eviction rates
        eff_s = max(1, self.effective_sim_us) / 1_000_000
        eviction_rate = {
            "L1_to_L2_pressure": r(self.l1_to_l2_evictions / eff_s),
            "L1_to_L2_ttl": r(self.l1_to_l2_ttl_migrations / eff_s),
            "L2_to_L3A": r(self.l2_to_l3a_evictions / eff_s),
        }

        # Memory pollution in GB
        pollution_gb = {
            tier_name: r(self.memory_pollution_bytes.get(tier_name, 0) / 1e9)
            for tier_name in ["L1", "L2", "L3A"]
        }

        # Prefill blocked pct
        total_us = max(1, self.effective_sim_us)
        blocked_pct = r(self.prefill_slot_blocked_us / total_us * 100)

        # Queue wait stats
        queue_wait_ms = {}
        if self.queue_wait_us:
            for p in [50, 95, 99]:
                queue_wait_ms[f"p{p}"] = r(float(np.percentile(self.queue_wait_us, p)) / 1000.0)
            queue_wait_ms["mean"] = r(float(np.mean(self.queue_wait_us)) / 1000.0)

        # Prefill duration stats
        prefill_duration_ms = {}
        if self.prefill_duration_us:
            for p in [50, 95, 99]:
                prefill_duration_ms[f"p{p}"] = r(float(np.percentile(self.prefill_duration_us, p)) / 1000.0)
            prefill_duration_ms["mean"] = r(float(np.mean(self.prefill_duration_us)) / 1000.0)

        # Slot utilization
        mean_slot_util = r(float(np.mean(self.slot_utilization_pct))) if self.slot_utilization_pct else 0.0

        # Multi-node dispatch stats
        dispatch_stats = {
            "affinity_dispatches": self.affinity_dispatches,
            "non_affinity_dispatches": self.non_affinity_dispatches,
            "cross_node_transfers": self.cross_node_transfers,
        }

        result = {
            "tier_saturation_pct": tier_sat,
            "ttft_ms": ttft_ms,
            "cache_hit_rate": hit_rate,
            "sharing_factor": r(self.sharing_factor),
            "memory_pollution_gb": pollution_gb,
            "eviction_rate_per_s": eviction_rate,
            "session_cold_evictions": self.session_cold_evictions,
            "savings_class_distribution": savings_dist,
            "prefill_slot_blocked_pct": blocked_pct,
        }

        if queue_wait_ms:
            result["queue_wait_ms"] = queue_wait_ms

        if prefill_duration_ms:
            result["prefill_duration_ms"] = prefill_duration_ms
            result["mean_slot_utilization_pct"] = mean_slot_util

        # Block sharing stats
        if self.shared_block_memory_saved_bytes > 0:
            result["sharing"] = {
                "groups": self.shared_block_groups,
                "memory_saved_gb": r(self.shared_block_memory_saved_bytes / 1e9),
                "max_ref_count": self.shared_block_ref_count_max,
            }
            if self.duplicate_block_bytes:
                result["sharing"]["duplicate_bytes_last_gb"] = r(self.duplicate_block_bytes[-1] / 1e9)
            if self.max_replication_factor:
                result["sharing"]["max_replication"] = max(self.max_replication_factor)

        if self.affinity_dispatches + self.non_affinity_dispatches > 0:
            result["dispatch_stats"] = dispatch_stats

        return result
