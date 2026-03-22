"""
Block-level allocation tests — verify token block math, fragmentation,
and cache hit granularity at different block sizes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.cache import (
    token_blocks, cached_tokens_at_block_boundary, block_fragmentation,
    kv_size_bytes,
)
from sim.config import SimConfig, ModelConfig

MODEL = ModelConfig(
    model_id="test-70b", n_layers=80, n_kv_heads=8, head_dim=128, bytes_per_element=2
)
CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "default.json")


# ── Token block count ──

def test_block_count_16_tokens():
    """100 tokens at block_size=16 → 7 blocks (6×16 + 1×4)."""
    assert token_blocks(100, 16) == 7


def test_block_count_256_tokens():
    """1000 tokens at block_size=256 → 4 blocks (3×256 + 1×232)."""
    assert token_blocks(1000, 256) == 4


def test_block_count_4096_tokens():
    """50000 tokens at block_size=4096 → 13 blocks."""
    assert token_blocks(50000, 4096) == 13


def test_block_count_exact_fit():
    """Exact multiple → no partial block."""
    assert token_blocks(256, 16) == 16
    assert token_blocks(1024, 256) == 4


def test_block_count_legacy():
    """block_size=0 → legacy mode, always 1 block."""
    assert token_blocks(100, 0) == 1
    assert token_blocks(100000, 0) == 1


# ── Block boundary rounding ──

def test_cached_at_boundary_16():
    """95 cached tokens at block_size=16 → 80 (5 full blocks)."""
    assert cached_tokens_at_block_boundary(95, 16) == 80


def test_cached_at_boundary_256():
    """500 cached tokens at block_size=256 → 256 (1 full block)."""
    assert cached_tokens_at_block_boundary(500, 256) == 256


def test_cached_at_boundary_4096():
    """5000 cached tokens at block_size=4096 → 4096 (1 full block)."""
    assert cached_tokens_at_block_boundary(5000, 4096) == 4096


def test_cached_at_boundary_exact():
    """Exact boundary → no loss."""
    assert cached_tokens_at_block_boundary(256, 16) == 256
    assert cached_tokens_at_block_boundary(1024, 256) == 1024


def test_cached_at_boundary_legacy():
    """Legacy mode (block_size=0) → no rounding."""
    assert cached_tokens_at_block_boundary(95, 0) == 95


def test_cached_at_boundary_small_count():
    """Fewer tokens than block_size → 0 cached (no full block)."""
    assert cached_tokens_at_block_boundary(15, 16) == 0
    assert cached_tokens_at_block_boundary(255, 256) == 0


# ── Fragmentation ──

def test_fragmentation_16_blocks():
    """Fragmentation with 16-token blocks."""
    useful, wasted = block_fragmentation(100, 16, MODEL)
    assert useful == kv_size_bytes(100, MODEL)
    # 7 blocks × 16 tokens = 112 tokens allocated, 12 wasted
    assert wasted == kv_size_bytes(12, MODEL)


def test_fragmentation_256_blocks():
    """Fragmentation with 256-token blocks."""
    useful, wasted = block_fragmentation(1000, 256, MODEL)
    assert useful == kv_size_bytes(1000, MODEL)
    # 4 blocks × 256 = 1024 tokens, 24 wasted
    assert wasted == kv_size_bytes(24, MODEL)


def test_fragmentation_exact_fit():
    """Exact fit → zero fragmentation."""
    useful, wasted = block_fragmentation(256, 16, MODEL)
    assert wasted == 0


def test_fragmentation_legacy():
    """Legacy mode → zero fragmentation."""
    useful, wasted = block_fragmentation(1000, 0, MODEL)
    assert useful == kv_size_bytes(1000, MODEL)
    assert wasted == 0


# ── Integration: block size affects cache hit granularity ──

def test_block_size_affects_hit_granularity():
    """Larger block sizes lose more cached tokens at boundaries.
    Example: 10000 context × 0.95 stability = 9500 cached tokens.
    At block_size=16: 9500 // 16 × 16 = 9488 cached (12 lost)
    At block_size=256: 9500 // 256 × 256 = 9472 cached (28 lost)
    At block_size=4096: 9500 // 4096 × 4096 = 8192 cached (1308 lost)"""
    raw_cached = 9500
    assert cached_tokens_at_block_boundary(raw_cached, 16) == 9488
    assert cached_tokens_at_block_boundary(raw_cached, 256) == 9472
    assert cached_tokens_at_block_boundary(raw_cached, 4096) == 8192
    # Larger blocks lose more — this is the granularity tradeoff


def test_block_larger_than_context():
    """Block size larger than total context → 0 cached tokens (no full block)."""
    assert cached_tokens_at_block_boundary(500, 1000000) == 0


# ── Integration: sim with block mode ──

def test_16_token_blocks_sim():
    """Run sim with block_size=16, verify it completes and produces results."""
    from sim.engine import SimEngine
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 5.0
    config.warmup_s = 1.0
    config.epoch_report_interval_s = 1.0
    config.sim_start_time_s = 36000.0
    config.cache.block_size_tokens = 16
    m = SimEngine(config).run()
    assert sum(m.savings_events.values()) > 0


def test_256_token_blocks_sim():
    """Run sim with block_size=256."""
    from sim.engine import SimEngine
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 5.0
    config.warmup_s = 1.0
    config.epoch_report_interval_s = 1.0
    config.sim_start_time_s = 36000.0
    config.cache.block_size_tokens = 256
    m = SimEngine(config).run()
    assert sum(m.savings_events.values()) > 0


def test_4096_token_blocks_sim():
    """Run sim with block_size=4096 (page-aligned)."""
    from sim.engine import SimEngine
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 5.0
    config.warmup_s = 1.0
    config.epoch_report_interval_s = 1.0
    config.sim_start_time_s = 36000.0
    config.cache.block_size_tokens = 4096
    m = SimEngine(config).run()
    assert sum(m.savings_events.values()) > 0


def test_block_size_affects_hit_rate():
    """Larger block sizes should have equal or lower cache hit effectiveness.
    Because block boundary rounding loses more cached tokens."""
    from sim.engine import SimEngine
    import copy

    base = SimConfig.from_json(CONFIG_PATH)
    base.sim_duration_s = 5.0
    base.warmup_s = 1.0
    base.epoch_report_interval_s = 1.0
    base.sim_start_time_s = 36000.0

    recompute_fractions = {}
    for block_size in [0, 16, 256, 4096]:
        c = copy.deepcopy(base)
        c.cache.block_size_tokens = block_size
        m = SimEngine(c).run()
        if m.recompute_fraction:
            import numpy as np
            recompute_fractions[block_size] = np.mean(m.recompute_fraction)

    # Larger blocks → more recompute (or equal) because more tokens lost at boundary
    if 0 in recompute_fractions and 4096 in recompute_fractions:
        assert recompute_fractions[4096] >= recompute_fractions[0] - 0.01, (
            f"4096-token blocks should have >= recompute fraction vs legacy. "
            f"4096={recompute_fractions[4096]:.3f} vs legacy={recompute_fractions[0]:.3f}"
        )
