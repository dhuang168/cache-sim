"""Analysis helpers for cache_sim."""
from __future__ import annotations

import numpy as np

from sim.config import SimConfig
from sim.engine import SimEngine


def find_sustaining_qps(
    base_config: SimConfig,
    sla_queue_wait_p95_ms: float = 500.0,
    rate_range: tuple[float, float] = (0.1, 5.0),
    tolerance: float = 0.05,
    max_iterations: int = 10,
) -> tuple[float, dict]:
    """
    Binary search for the max arrival rate multiplier where p95 queue wait
    stays under the SLA threshold.

    Args:
        base_config: Base config (arrival rates will be scaled).
        sla_queue_wait_p95_ms: SLA threshold for p95 queue wait in ms.
        rate_range: (low_mult, high_mult) range to search.
        tolerance: Stop when range narrows to this fraction.
        max_iterations: Max binary search steps.

    Returns:
        (max_multiplier, metrics_report) at the sustaining rate.
    """
    sla_us = sla_queue_wait_p95_ms * 1000.0
    lo, hi = rate_range
    best_mult = lo
    best_report = {}

    for _ in range(max_iterations):
        mid = (lo + hi) / 2.0
        cfg = _scale_arrival_rates(base_config, mid)
        m = SimEngine(cfg).run()

        if m.queue_wait_us:
            p95_wait = float(np.percentile(m.queue_wait_us, 95))
        else:
            p95_wait = 0.0

        if p95_wait <= sla_us:
            best_mult = mid
            best_report = m.report()
            best_report["_rate_multiplier"] = mid
            best_report["_queue_wait_p95_ms"] = p95_wait / 1000.0
            lo = mid
        else:
            hi = mid

        if (hi - lo) / max(lo, 0.01) < tolerance:
            break

    return best_mult, best_report


def _scale_arrival_rates(config: SimConfig, multiplier: float) -> SimConfig:
    """Return a copy of config with all arrival_rate_peak scaled by multiplier."""
    import copy
    cfg = copy.deepcopy(config)
    for p in cfg.profiles:
        p.arrival_rate_peak *= multiplier
    return cfg
