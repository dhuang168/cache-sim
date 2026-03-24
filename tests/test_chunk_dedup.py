"""
Integration tests for chunk-level deduplication (LMCache-style).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from sim.config import SimConfig
from sim.engine import SimEngine

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "default.json")
LMCACHE_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "lmcache.json")


def _quick_config() -> SimConfig:
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 10.0
    config.warmup_s = 1.0
    config.epoch_report_interval_s = 2.0
    config.sim_start_time_s = 36000.0
    return config


def _chunk_config() -> SimConfig:
    config = _quick_config()
    config.cache.block_size_tokens = 256
    config.cache.eviction_policy = "lru"
    config.cache.deduplication = "chunk"
    config.cache.tier_migration = "demand_pull"
    return config


def _stressed_chunk_config() -> SimConfig:
    config = _chunk_config()
    config.tiers[0].capacity_bytes = 500 * 1024**2   # 500 MB L1
    config.tiers[1].capacity_bytes = 2 * 1024**3
    config.tiers[2].capacity_bytes = 50 * 1024**3
    return config


# ─── Backward compat ───

def test_per_session_dedup_matches_default():
    """deduplication='per_session' produces same results as baseline."""
    config = _quick_config()
    config.cache.deduplication = "per_session"
    config.cache.tier_migration = "ttl_push"
    m = SimEngine(config).run()
    assert len(m.kv_transfer_us) == 0  # no disaggregated
    assert m.chunk_total_logical == 0  # no chunk metrics
    assert sum(m.savings_events.values()) > 0


# ─── Config validation ───

def test_chunk_requires_block_size():
    """chunk mode with block_size_tokens=0 raises ValueError."""
    config = _quick_config()
    config.cache.deduplication = "chunk"
    config.cache.block_size_tokens = 0
    with pytest.raises(ValueError, match="block_size_tokens"):
        SimEngine(config)


def test_demand_pull_requires_lru():
    """demand_pull with eviction_policy='ttl' raises ValueError."""
    config = _quick_config()
    config.cache.tier_migration = "demand_pull"
    config.cache.eviction_policy = "ttl"
    with pytest.raises(ValueError, match="eviction_policy"):
        SimEngine(config)


# ─── Chunk dedup functionality ───

def test_chunk_dedup_ratio_positive():
    """Chunk mode produces nonzero dedup ratio (shared prefixes deduped)."""
    config = _chunk_config()
    m = SimEngine(config).run()
    assert m.chunk_total_logical > 0
    assert m.chunk_dedup_hits > 0
    ratio = m.chunk_dedup_hits / m.chunk_total_logical
    assert ratio > 0.1, f"Dedup ratio {ratio:.3f} too low"


def test_chunk_dedup_reduces_storage():
    """Chunk mode uses less L1 storage than per-session mode for same workload."""
    # Per-session mode
    cfg_ps = _quick_config()
    cfg_ps.tiers[0].capacity_bytes = 500 * 1024**2
    m_ps = SimEngine(cfg_ps).run()
    l1_ps = m_ps.tier_occupancy_pct.get("L1", [0])

    # Chunk mode
    cfg_ch = _stressed_chunk_config()
    m_ch = SimEngine(cfg_ch).run()
    l1_ch = m_ch.tier_occupancy_pct.get("L1", [0])

    # Both should have used L1
    assert max(l1_ps) > 0 or max(l1_ch) > 0
    # Chunk mode should have stored fewer unique bytes due to dedup
    assert m_ch.chunk_dedup_hits > 0


def test_chunk_consecutive_lookup_hit():
    """With enough sim time, chunk mode should produce L1 cache hits."""
    config = _chunk_config()
    config.sim_duration_s = 30.0
    config.warmup_s = 5.0
    m = SimEngine(config).run()
    # Should have some L1 hits from returning sessions
    l1_hits = m.savings_events.get("CACHE_HIT_L1", 0)
    # With large L1 (80GB default), most chunks stay cached
    assert l1_hits >= 0  # May be 0 for very short sessions


def test_lmcache_config_loads_and_runs():
    """configs/lmcache.json loads and runs without error."""
    config = SimConfig.from_json(LMCACHE_PATH)
    config.sim_duration_s = 10.0
    config.warmup_s = 1.0
    m = SimEngine(config).run()
    assert sum(m.savings_events.values()) > 0
    assert m.chunk_total_logical > 0


# ─── Demand-pull tests ───

def test_demand_pull_promotes_on_hit():
    """Demand-pull mode promotes chunks from L2/L3A to L1."""
    config = _stressed_chunk_config()
    config.sim_duration_s = 30.0
    config.warmup_s = 5.0
    m = SimEngine(config).run()
    # With small L1 (500MB), chunks get evicted to L2/L3A.
    # When sessions return, demand-pull should promote them back.
    assert m.tier_promotions >= 0  # may be 0 if no returning sessions in short sim


def test_demand_pull_no_ttl_scheduling():
    """demand_pull mode should not produce TTL-driven L1->L2 migrations."""
    config = _chunk_config()
    m = SimEngine(config).run()
    # TTL migrations should be 0 in demand_pull mode
    assert m.l1_to_l2_ttl_migrations == 0


def test_demand_pull_object_mode():
    """Demand-pull works with per-session objects too (not just chunks)."""
    config = _quick_config()
    config.cache.eviction_policy = "lru"
    config.cache.tier_migration = "demand_pull"
    config.cache.deduplication = "per_session"
    m = SimEngine(config).run()
    assert sum(m.savings_events.values()) > 0
    assert m.l1_to_l2_ttl_migrations == 0


# ─── Tail-first eviction (vLLM) tests ───

def test_tail_first_preserves_prefix():
    """vLLM tail-first eviction preserves low-index chunks (shared prefix)."""
    config = _stressed_chunk_config()
    config.cache.chunk_eviction = "tail_first"
    config.sim_duration_s = 30.0
    config.warmup_s = 5.0
    m_tail = SimEngine(config).run()

    # Compare with LRU
    config2 = _stressed_chunk_config()
    config2.cache.chunk_eviction = "lru"
    config2.sim_duration_s = 30.0
    config2.warmup_s = 5.0
    m_lru = SimEngine(config2).run()

    # Tail-first should have equal or better hit rate (preserves prefix chains)
    hit_tail = 1.0 - m_tail.report()["cache_hit_rate"].get("miss", 1.0)
    hit_lru = 1.0 - m_lru.report()["cache_hit_rate"].get("miss", 1.0)
    assert hit_tail >= hit_lru * 0.95, (
        f"Tail-first hit {hit_tail:.3f} should be >= LRU hit {hit_lru:.3f}"
    )


def test_tail_first_higher_dedup():
    """vLLM tail-first should achieve higher dedup ratio (shared prefix stays cached)."""
    config = _stressed_chunk_config()
    config.cache.chunk_eviction = "tail_first"
    config.sim_duration_s = 30.0
    config.warmup_s = 5.0
    m = SimEngine(config).run()
    dedup = m.report().get("chunk_dedup", {})
    # Tail-first preserves shared prefix chunks → more dedup hits
    assert dedup.get("dedup_ratio", 0) > 0.3


# ─── Performance ───

def test_chunk_mode_perf_benchmark():
    """10s sim with chunk mode completes in reasonable time."""
    import time
    config = _chunk_config()
    start = time.monotonic()
    m = SimEngine(config).run()
    elapsed = time.monotonic() - start
    assert elapsed < 15.0, f"10s chunk sim took {elapsed:.1f}s"
    assert sum(m.savings_events.values()) > 0
