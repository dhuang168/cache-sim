"""Tests for latency oracle."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.oracle import PrefillOracle, DecodeOracle, transfer_time_us
from sim.config import TierConfig

TABLE_PATH = str(Path(__file__).resolve().parent.parent / "benchmarks" / "latency_tables" / "prefill_70b_a100.json")


def test_prefill_oracle_exact_points():
    oracle = PrefillOracle(TABLE_PATH)
    assert oracle.prefill_latency_us(512) == 45000
    assert oracle.prefill_latency_us(1024) == 90000


def test_prefill_oracle_interpolation():
    oracle = PrefillOracle(TABLE_PATH)
    lat = oracle.prefill_latency_us(768)
    # Linear interpolation between 512->45000 and 1024->90000
    assert 45000 < lat < 90000


def test_prefill_oracle_clamp():
    oracle = PrefillOracle(TABLE_PATH)
    # Below minimum should clamp
    assert oracle.prefill_latency_us(1) == 45000
    # Above maximum should clamp
    assert oracle.prefill_latency_us(500000) == 600000000


def test_transfer_time():
    tier = TierConfig(
        name="L2", capacity_bytes=4_000_000_000_000,
        bandwidth_bytes_per_s=64_000_000_000,
        latency_floor_us=10, block_size_bytes=33_554_432,
    )
    t = transfer_time_us(64_000_000, tier)  # 64MB
    # 10 + (64e6 / 64e9) * 1e6 = 10 + 1000 = 1010
    assert t == 1010


def test_decode_oracle():
    oracle = DecodeOracle()
    lat = oracle.decode_latency_us(100, 1)
    assert lat > 0
    # More sequences = higher latency
    lat2 = oracle.decode_latency_us(100, 100)
    assert lat2 > lat
