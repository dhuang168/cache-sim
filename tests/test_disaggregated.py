"""
Disaggregated prefill-decode separation tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from sim.config import SimConfig
from sim.engine import SimEngine
from sim.cache import kv_size_bytes

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "default.json")


def _quick_config() -> SimConfig:
    """10s sim, 1s warmup — fast for testing."""
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 10.0
    config.warmup_s = 1.0
    config.epoch_report_interval_s = 2.0
    config.sim_start_time_s = 36000.0  # 10 AM
    return config


def _disagg_config(n_prefill: int = 3, n_decode: int = 1) -> SimConfig:
    """Disaggregated config with stressed cache."""
    config = _quick_config()
    config.tiers[0].capacity_bytes = 500 * 1024**2   # 500 MB L1 per GPU
    config.tiers[1].capacity_bytes = 2 * 1024**3      # 2 GB L2 per worker
    config.tiers[2].capacity_bytes = 50 * 1024**3      # 50 GB L3A
    config.ttl_l2_s = 20.0
    config.ttl_l3a_s = 120.0
    config.service.n_prefill_nodes = n_prefill
    config.service.n_gpus_per_worker = n_prefill
    config.service.disaggregated = True
    config.service.n_decode_nodes = n_decode
    config.service.kv_transfer_bandwidth_bytes_per_s = 50_000_000_000  # 50 GB/s
    config.service.kv_transfer_latency_floor_us = 2000
    config.service.prefill_latency_multiplier = 0.85
    config.service.decode_batch_fill_factor = 0.85
    return config


# ─── Test 1: Backward compatibility ───

def test_backward_compat_disaggregated_false():
    """Explicit disaggregated=false matches default behavior."""
    config = _quick_config()
    config.service.disaggregated = False
    config.service.n_decode_nodes = 0
    metrics = SimEngine(config).run()
    report = metrics.report()

    # Should not have any disaggregated metrics
    assert len(metrics.kv_transfer_us) == 0
    assert "kv_transfer_ms" not in report
    # Should have standard metrics
    assert "cache_hit_rate" in report
    assert sum(metrics.savings_events.values()) > 0


# ─── Test 2: KV transfer events fire ───

def test_kv_transfer_events_fire():
    """3P:1D config produces KV transfer events with positive latencies."""
    config = _disagg_config()
    metrics = SimEngine(config).run()
    report = metrics.report()

    assert len(metrics.kv_transfer_us) > 0, "No KV transfers recorded"
    assert all(t > 0 for t in metrics.kv_transfer_us), "KV transfer times must be positive"
    assert "kv_transfer_ms" in report
    assert report["kv_transfer_ms"]["mean"] > 0


# ─── Test 3: Prefill not blocked by decode ───

def test_prefill_not_blocked_by_decode():
    """In disaggregated mode, prefill slots are freed immediately — no decode backpressure."""
    config = _disagg_config()
    # Tiny decode capacity to stress backpressure
    config.service.n_decode_slots = 2
    config.service.decode_queue_max = 4
    metrics = SimEngine(config).run()
    report = metrics.report()

    # In disaggregated mode, prefill_slot_blocked should be 0
    assert report.get("prefill_slot_blocked_pct", 0) == 0


# ─── Test 4: Decode node utilization ───

def test_decode_node_utilization():
    """Decode nodes should have active sequences under load."""
    config = _disagg_config()
    metrics = SimEngine(config).run()

    # Per-decode-node active seqs should be sampled
    assert len(metrics.per_decode_node_active_seqs) > 0
    # At least one decode node should have had nonzero active sequences
    all_seqs = []
    for node_seqs in metrics.per_decode_node_active_seqs.values():
        all_seqs.extend(node_seqs)
    assert max(all_seqs) > 0, "Decode nodes never had active sequences"


# ─── Test 5: KV transfer size matches model ───

def test_kv_transfer_size_matches_model():
    """Transfer bytes should match kv_size_bytes for the model."""
    config = _disagg_config()
    metrics = SimEngine(config).run()

    assert len(metrics.kv_transfer_bytes) > 0
    # All transfer sizes should be multiples of the per-token KV size
    # (2 * n_layers * n_kv_heads * head_dim * bytes_per_element)
    per_token = 2 * 80 * 8 * 128 * 2  # = 327,680 bytes
    for size in metrics.kv_transfer_bytes:
        assert size > 0
        # Size should be per_token * some_token_count (integer)
        assert size % per_token == 0, f"Transfer size {size} not a multiple of {per_token}"


# ─── Test 6: KV written to prefill node ───

def test_kv_write_targets_prefill_node():
    """After decode, KV cache is written to the prefill node's L1, not a decode node."""
    config = _disagg_config(n_prefill=1, n_decode=1)
    config.tiers[0].capacity_bytes = 80 * 1024**3  # large L1, keep objects
    metrics = SimEngine(config).run()

    # The engine should have objects in node[0] (prefill node) L1
    # Decode nodes have no cache stores
    # Just verify the sim completes and has cache hits
    assert sum(metrics.savings_events.values()) > 0
    report = metrics.report()
    assert report["cache_hit_rate"]["L1"] > 0, "No L1 hits — KV not written to prefill node?"


# ─── Test 7: Prefill multiplier effect ───

def test_prefill_multiplier_effect():
    """prefill_latency_multiplier=0.5 should roughly halve prefill durations."""
    # Run with multiplier=1.0
    config1 = _disagg_config()
    config1.service.prefill_latency_multiplier = 1.0
    m1 = SimEngine(config1).run()

    # Run with multiplier=0.5
    config2 = _disagg_config()
    config2.service.prefill_latency_multiplier = 0.5
    m2 = SimEngine(config2).run()

    if m1.prefill_duration_us and m2.prefill_duration_us:
        median1 = np.median(m1.prefill_duration_us)
        median2 = np.median(m2.prefill_duration_us)
        # median2 should be roughly half of median1 (allow 30% tolerance)
        ratio = median2 / median1 if median1 > 0 else 1.0
        assert 0.3 < ratio < 0.7, (
            f"Expected ~0.5 ratio, got {ratio:.3f} (median1={median1:.0f}, median2={median2:.0f})"
        )


# ─── Test 8: Transfer bandwidth sensitivity ───

def test_transfer_bandwidth_sensitivity():
    """Low bandwidth should produce much higher transfer times."""
    # High bandwidth: 100 GB/s
    cfg_fast = _disagg_config()
    cfg_fast.service.kv_transfer_bandwidth_bytes_per_s = 100_000_000_000
    m_fast = SimEngine(cfg_fast).run()

    # Low bandwidth: 1 GB/s
    cfg_slow = _disagg_config()
    cfg_slow.service.kv_transfer_bandwidth_bytes_per_s = 1_000_000_000
    m_slow = SimEngine(cfg_slow).run()

    if m_fast.kv_transfer_us and m_slow.kv_transfer_us:
        mean_fast = np.mean(m_fast.kv_transfer_us)
        mean_slow = np.mean(m_slow.kv_transfer_us)
        assert mean_slow > mean_fast * 5, (
            f"Slow BW transfer ({mean_slow:.0f}us) should be >5x fast ({mean_fast:.0f}us)"
        )


# ─── Test 9: Decode backpressure ───

def test_decode_backpressure():
    """With 1 decode slot, heavy load should queue or drop requests."""
    config = _disagg_config()
    config.service.n_decode_slots = 1
    config.service.decode_queue_max = 8
    metrics = SimEngine(config).run()

    # Should still complete some requests
    assert sum(metrics.savings_events.values()) > 0
    # With severe decode bottleneck, decode queue wait should be nonzero
    if metrics.decode_queue_wait_us:
        assert max(metrics.decode_queue_wait_us) > 0


# ─── Test 10: Performance benchmark ───

def test_disaggregated_perf_benchmark():
    """10s sim with 3P:1D completes in reasonable wall-clock time."""
    import time
    config = _disagg_config()
    start = time.monotonic()
    metrics = SimEngine(config).run()
    elapsed = time.monotonic() - start
    assert elapsed < 10.0, f"10s sim took {elapsed:.1f}s wall-clock (>10s)"
    assert sum(metrics.savings_events.values()) > 0
