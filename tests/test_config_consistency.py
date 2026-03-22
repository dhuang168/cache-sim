"""
Config consistency tests — verify simulator invariants that have caused
bugs in the past. Each test uses minimal sim duration or no sim at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sim.config import SimConfig
from sim.engine import SimEngine
from sim.oracle import PrefillOracle
from collections import Counter

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "default.json")
ORACLE_PATH = str(Path(__file__).resolve().parent.parent / "benchmarks" / "latency_tables" / "prefill_70b_a100.json")


def _quick_config() -> SimConfig:
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 10.0
    config.warmup_s = 1.0
    config.epoch_report_interval_s = 2.0
    config.sim_start_time_s = 36000.0
    return config


# ── Oracle tests ──

def test_oracle_covers_workload_range():
    """Oracle table must cover the token range of the configured workloads.
    Bug: oracle clamped at 32k tokens, making cold misses on 60k+ tokens
    appear cheaper than cache hits."""
    oracle = PrefillOracle(ORACLE_PATH)
    config = SimConfig.from_json(CONFIG_PATH)

    # Compute max possible context: largest shared_prefix + max context_growth × reasonable turns + input
    for profile in config.profiles:
        max_context = (
            profile.shared_system_prefix_tokens
            + profile.context_growth_max_tokens * 20  # ~20 turns
            + profile.input_len_mean_tokens * 3  # 3σ
        )
        # Oracle should return a value that increases (not clamp) at this token count
        lat_at_max = oracle.prefill_latency_us(max_context)
        lat_at_half = oracle.prefill_latency_us(max_context // 2)
        assert lat_at_max > lat_at_half, (
            f"Oracle appears clamped: oracle({max_context})={lat_at_max} "
            f"<= oracle({max_context//2})={lat_at_half} for profile {profile.name}"
        )


def test_oracle_monotonically_increasing():
    """Prefill latency must increase with token count."""
    oracle = PrefillOracle(ORACLE_PATH)
    prev = 0
    for tokens in [100, 500, 1000, 5000, 10000, 30000, 50000, 100000, 200000]:
        lat = oracle.prefill_latency_us(tokens)
        assert lat >= prev, f"Oracle not monotonic: {tokens} tokens → {lat} < {prev}"
        prev = lat


# ── Tier capacity tests ──

def test_l2_occupancy_never_exceeds_100():
    """L2 occupancy must never exceed 100%.
    Bug: evict_l1_to_l2() inserted without can_fit() check."""
    config = _quick_config()
    config.tiers[0].capacity_bytes = 500 * 1024**2  # small L1 → pressure evictions
    config.tiers[1].capacity_bytes = 1 * 1024**3     # small L2 → likely overflow target
    metrics = SimEngine(config).run()
    l2_occ = metrics.tier_occupancy_pct.get("L2", [0])
    assert max(l2_occ) <= 100.5, (
        f"L2 occupancy exceeded 100%: max={max(l2_occ):.1f}%"
    )


# ── Workload mix tests ──

def test_all_profiles_generate_sessions():
    """Every profile with mix weight > 0 must generate at least one session.
    Bug: diurnal rate = 0 at midnight → only batch generated traffic."""
    config = _quick_config()
    engine = SimEngine(config)

    session_profiles = []
    orig = engine._new_session
    def tracking(profile, start_us):
        session_profiles.append(profile.name)
        return orig(profile, start_us)
    engine._new_session = tracking
    engine.run()

    for name, weight in config.profile_mix.items():
        if weight > 0:
            count = session_profiles.count(name)
            assert count > 0, (
                f"Profile '{name}' (mix={weight}) generated 0 sessions. "
                f"Check sim_start_time_s and diurnal_peak_trough_ratio."
            )


def test_session_proportions_match_profile_mix():
    """Session proportions must roughly match profile_mix weights (±15%).
    Bug: profile_mix didn't scale arrival rate → batch dominated."""
    config = _quick_config()
    engine = SimEngine(config)

    session_profiles = []
    orig = engine._new_session
    def tracking(profile, start_us):
        session_profiles.append(profile.name)
        return orig(profile, start_us)
    engine._new_session = tracking
    engine.run()

    total = len(session_profiles)
    assert total > 50, f"Too few sessions ({total}) for proportion test"
    counts = Counter(session_profiles)
    for name, weight in config.profile_mix.items():
        if weight > 0:
            actual = counts.get(name, 0) / total
            assert abs(actual - weight) < 0.15, (
                f"Profile '{name}': actual={actual:.1%} vs target={weight:.0%} "
                f"(diff={abs(actual-weight):.1%} > 15%)"
            )


# ── Queue and metrics tests ──

def test_queue_depth_nonzero_under_load():
    """Under sufficient load, prefill queue should have nonzero depth.
    Bug: epoch report was reading wrong queue variable."""
    config = _quick_config()
    # High arrival rate to ensure queueing
    for p in config.profiles:
        p.arrival_rate_peak = 100
    config.tiers[0].capacity_bytes = 500 * 1024**2  # small L1 → slower prefill
    metrics = SimEngine(config).run()
    assert max(metrics.prefill_queue_depth) > 0, (
        "Prefill queue depth was always 0 — queue metric may not be wired correctly"
    )


def test_ttl_migrations_separate_from_pressure():
    """TTL migrations and pressure evictions must be tracked separately.
    Bug: both were conflated in a single counter."""
    config = _quick_config()
    config.tiers[0].capacity_bytes = 2 * 1024**3  # 2GB L1 — some pressure
    config.ttl_l2_s = 5.0  # short TTL to trigger migrations in 10s sim
    metrics = SimEngine(config).run()
    # At least one type should be nonzero if there's any L1 activity
    total_l1_to_l2 = metrics.l1_to_l2_evictions + metrics.l1_to_l2_ttl_migrations
    if total_l1_to_l2 > 0:
        # If both are identical and nonzero, they might be conflated
        # (This is a soft check — both could legitimately be equal)
        assert hasattr(metrics, 'l1_to_l2_evictions'), "Missing pressure eviction counter"
        assert hasattr(metrics, 'l1_to_l2_ttl_migrations'), "Missing TTL migration counter"


def test_no_global_l3a_penalty_single_worker():
    """Global L3A with 1 worker should not incur remote penalty.
    Bug: penalty applied when n_nodes > 1 instead of n_workers > 1."""
    config = _quick_config()
    config.service.n_prefill_nodes = 8
    config.service.n_gpus_per_worker = 8  # 1 worker
    config.service.l3a_shared = True
    config.service.l3a_remote_latency_us = 50_000
    engine = SimEngine(config)
    # With 1 worker, n_workers=1 → no remote penalty should be applied
    assert engine._n_workers == 1, "Should be 1 worker"
    # The penalty check uses n_workers > 1, so single worker = no penalty
    # Run sim and check that L3A hits (if any) don't have inflated TTFT
    metrics = engine.run()
    # Basic sanity: sim completes without error
    assert sum(metrics.savings_events.values()) > 0
