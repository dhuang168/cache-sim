"""Tests for KV size computation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import ModelConfig
from sim.cache import kv_size_bytes, allocated_blocks, block_waste_ratio


def _model_70b_fp16() -> ModelConfig:
    return ModelConfig(
        model_id="llama3-70b",
        n_layers=80,
        n_kv_heads=8,
        head_dim=128,
        bytes_per_element=2,
    )


def test_kv_size_1000_tokens():
    """kv_size_bytes(1000, model_70b_fp16) must be ~327,680 bytes."""
    model = _model_70b_fp16()
    size = kv_size_bytes(1000, model)
    expected = 327_680_000
    # 2 * 80 * 8 * 128 * 2 * 1000 = 327,680,000
    assert size == expected, f"Expected {expected}, got {size}"


def test_kv_size_scales_linearly():
    model = _model_70b_fp16()
    s1 = kv_size_bytes(1000, model)
    s2 = kv_size_bytes(2000, model)
    assert s2 == 2 * s1


def test_allocated_blocks_ceiling():
    assert allocated_blocks(100, 64) == 2
    assert allocated_blocks(64, 64) == 1
    assert allocated_blocks(1, 64) == 1
    assert allocated_blocks(0, 64) == 0


def test_block_waste_ratio():
    ratio = block_waste_ratio(100, 64)
    # 2 blocks * 64 = 128 bytes allocated, waste = 28/128 = 0.21875
    assert abs(ratio - 0.21875) < 1e-9
