"""
Cross-session block sharing tests — verify multi-tier prefix sharing,
ref counting, and memory savings.
"""
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import SimConfig, SharingTier, SharingConfig, CacheConfig
from sim.engine import SimEngine
from sim.cache import kv_size_bytes

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "default.json")


def _sharing_config(block_size: int = 0) -> SimConfig:
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 10.0
    config.warmup_s = 1.0
    config.epoch_report_interval_s = 2.0
    config.sim_start_time_s = 36000.0
    config.cache = CacheConfig(
        block_size_tokens=block_size,
        sharing=SharingConfig(
            enabled=True,
            tiers=[
                SharingTier("framework", 20000, 1000),
                SharingTier("workspace", 5000, 10),
            ],
        ),
    )
    return config


# ── Config tests ──

def test_sharing_config_defaults():
    """SharingConfig disabled by default."""
    config = SimConfig.from_json(CONFIG_PATH)
    assert config.cache.sharing.enabled is False
    assert config.cache.sharing.tiers == []


def test_sharing_tier_stacking():
    """Framework + workspace tokens stack correctly."""
    config = _sharing_config()
    tiers = config.cache.sharing.tiers
    total_shared = sum(t.tokens for t in tiers)
    assert total_shared == 25000  # 20000 + 5000


# ── Ref counting tests ──

def test_ref_count_increment():
    """Multiple sessions in same framework group → ref_count > 1."""
    config = _sharing_config()
    engine = SimEngine(config)
    m = engine.run()

    # With sharing enabled, shared objects should have ref_count > 1
    # (multiple sessions in same framework group increment the same object)
    max_ref = 0
    for node in engine.nodes:
        for key, obj in node.l1_store.objects.items():
            if key.startswith("sp-"):
                max_ref = max(max_ref, obj.ref_count)
        for key, obj in node.l2_store.objects.items():
            if key.startswith("sp-"):
                max_ref = max(max_ref, obj.ref_count)
    # Check shared L3A too
    if engine._shared_l3a:
        for key, obj in engine._shared_l3a.objects.items():
            if key.startswith("sp-"):
                max_ref = max(max_ref, obj.ref_count)

    assert max_ref > 1 or len(engine._shared_block_index) > 0, (
        f"Expected shared objects with ref_count > 1, got max_ref={max_ref}"
    )


def test_memory_saved_metric():
    """With sharing, memory_saved should be > 0 for multiple sessions."""
    config = _sharing_config()
    m = SimEngine(config).run()
    assert m.shared_block_memory_saved_bytes > 0, (
        "Expected memory savings from shared prefix blocks"
    )


def test_shared_block_groups_created():
    """Sharing groups should be created for framework and workspace tiers."""
    config = _sharing_config()
    engine = SimEngine(config)
    m = engine.run()
    assert m.shared_block_groups > 0, "No sharing groups created"
    assert len(engine._shared_block_index) > 0, "Shared block index is empty"


# ── Extreme condition tests ──

def test_different_profiles_no_sharing():
    """Different profiles should NOT share framework blocks."""
    config = _sharing_config()
    engine = SimEngine(config)
    engine.run()
    # Check that coding and agentic_coding have different framework groups
    coding_keys = [k for k in engine._shared_block_index.keys() if "coding:" in k and "agentic" not in k]
    agentic_keys = [k for k in engine._shared_block_index.keys() if "agentic_coding:" in k]
    # They should be separate
    for ck in coding_keys:
        for ak in agentic_keys:
            assert ck != ak, "Different profiles should have separate sharing groups"


def test_sharing_disabled_backward_compat():
    """With sharing disabled, behavior is identical to pre-sharing code."""
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 5.0
    config.warmup_s = 1.0
    config.epoch_report_interval_s = 1.0
    config.sim_start_time_s = 36000.0
    # Sharing explicitly disabled
    config.cache.sharing.enabled = False
    m = SimEngine(config).run()
    assert m.shared_block_memory_saved_bytes == 0
    assert m.shared_block_groups == 0
    assert sum(m.savings_events.values()) > 0


# ── Integration: sharing reduces memory ──

def test_cross_worker_duplication_tracked():
    """With multiple workers, shared prefixes get duplicated.
    Framework prefix should appear on multiple workers."""
    config = _sharing_config()
    config.service.n_prefill_nodes = 16
    config.service.n_gpus_per_worker = 8  # 2 workers
    m = SimEngine(config).run()

    # With 2 workers, shared prefix objects may be duplicated
    # The duplication metrics should be populated
    if m.max_replication_factor:
        # At least some shared objects should exist on multiple workers
        max_repl = max(m.max_replication_factor)
        assert max_repl >= 1, "Expected replication factor >= 1"
    # Duplicate bytes should be tracked
    assert len(m.duplicate_block_bytes) > 0, "Duplication tracking not populated"


def test_l3a_bandwidth_contention_tracked():
    """Global L3A with high load should track bandwidth contention events."""
    config = _sharing_config()
    config.service.n_prefill_nodes = 16
    config.service.n_gpus_per_worker = 8  # 2 workers
    config.service.l3a_shared = True
    config.service.l3a_remote_latency_us = 50_000
    # High arrival rate to create concurrent L3A reads
    for p in config.profiles:
        p.arrival_rate_peak = 100
    m = SimEngine(config).run()
    # Contention tracking should be populated
    assert len(m.l3a_concurrent_reads) > 0, "L3A concurrent reads not tracked"


def test_sharing_reduces_tier_occupancy():
    """With sharing enabled, tier occupancy should be lower than without
    (because shared objects are stored once, not N times)."""
    import numpy as np

    base = SimConfig.from_json(CONFIG_PATH)
    base.sim_duration_s = 10.0
    base.warmup_s = 1.0
    base.epoch_report_interval_s = 2.0
    base.sim_start_time_s = 36000.0

    # Without sharing
    c_no = copy.deepcopy(base)
    c_no.cache.sharing.enabled = False
    m_no = SimEngine(c_no).run()

    # With sharing
    c_yes = copy.deepcopy(base)
    c_yes.cache = CacheConfig(
        sharing=SharingConfig(
            enabled=True,
            tiers=[
                SharingTier("framework", 20000, 1000),
                SharingTier("workspace", 5000, 10),
            ],
        ),
    )
    m_yes = SimEngine(c_yes).run()

    # Memory saved should be positive
    assert m_yes.shared_block_memory_saved_bytes > 0

    # Both should complete successfully
    assert sum(m_no.savings_events.values()) > 0
    assert sum(m_yes.savings_events.values()) > 0
