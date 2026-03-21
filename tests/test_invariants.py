"""
Four mandatory invariant tests. Must all pass before any exploratory run.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import SimConfig
from sim.engine import SimEngine

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "default.json")


def _short_config() -> SimConfig:
    """Load default config with shortened sim for fast testing."""
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 60.0
    config.warmup_s = 5.0
    config.epoch_report_interval_s = 5.0
    config.sim_start_time_s = 36000.0  # 10 AM — ensures all profiles have nonzero arrival rate
    return config


def test_zero_ttl_collapses_l2():
    """
    With TTL=0, all L2 objects hibernate immediately.
    L2 occupancy must stay at 0 after warmup.
    """
    config = _short_config()
    config.ttl_l2_s = 0.0
    metrics = SimEngine(config).run()
    l2_occ = metrics.tier_occupancy_pct.get("L2", [0.0])
    assert max(l2_occ) < 1.0, f"L2 occupancy should be ~0% with TTL=0, got max {max(l2_occ)}%"


def test_infinite_l1_no_evictions():
    """
    With L1 capacity = effectively infinite,
    L1 eviction count must be 0.
    """
    config = _short_config()
    config.tiers[0].capacity_bytes = 2 ** 50  # effectively infinite
    metrics = SimEngine(config).run()
    assert metrics.l1_to_l2_evictions == 0, (
        f"Expected 0 L1 evictions with infinite capacity, got {metrics.l1_to_l2_evictions}"
    )


def test_no_shared_prefix_sharing_factor_one():
    """
    With all profiles having shared_system_prefix_tokens=0,
    sharing_factor must equal 1.0 (no cross-session reuse).
    """
    config = _short_config()
    for profile in config.profiles:
        profile.shared_system_prefix_tokens = 0
    metrics = SimEngine(config).run()
    assert abs(metrics.sharing_factor - 1.0) < 0.01, (
        f"Expected sharing_factor ~1.0 without shared prefixes, got {metrics.sharing_factor}"
    )


def test_zero_bandwidth_penalty_prefers_restore():
    """
    With all tier bandwidths set to sys.maxsize (effectively infinite),
    every L2 and L3a hit must be classified as WORTHWHILE (never BREAK_EVEN).
    """
    config = _short_config()
    for tier in config.tiers:
        tier.bandwidth_bytes_per_s = sys.maxsize
        tier.latency_floor_us = 0
    metrics = SimEngine(config).run()
    be = metrics.savings_events.get("CACHE_HIT_L3A_BREAK_EVEN", 0)
    assert be == 0, (
        f"Expected 0 BREAK_EVEN events with infinite bandwidth, got {be}"
    )
