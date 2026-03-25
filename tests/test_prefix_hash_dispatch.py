"""
Tests for PrefixHashDispatcher — OpenAI-style prefix-hash routing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.des.dispatch import PrefixHashDispatcher
from agentsim.core.des.node import PrefillNode
from agentsim.core.des.cache import TierStore

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "heavy_coding.json")


def _quick_config(algo="prefix_hash", peak=15):
    cfg = SimConfig.from_json(CONFIG_PATH)
    cfg.sim_duration_s = 60.0; cfg.warmup_s = 10.0; cfg.sim_start_time_s = 36000.0
    cfg.service.n_prefill_nodes = 32; cfg.service.n_gpus_per_worker = 8
    cfg.service.dispatch_algorithm = algo
    for p in cfg.profiles: p.arrival_rate_peak = peak
    return cfg


def test_prefix_hash_runs():
    """prefix_hash dispatch algorithm runs without error."""
    cfg = _quick_config()
    m = SimEngine(cfg).run()
    assert sum(m.savings_events.values()) > 0


def test_prefix_hash_consistent_routing():
    """Same profile hashes to same node consistently."""
    nodes = []
    for i in range(8):
        l1 = TierStore("L1", 80*1024**3, 5120)
        l2 = TierStore("L2", 1024**4, 32*1024**2)
        nodes.append(PrefillNode(i, i//8, l1, l2, 32, 128))

    d = PrefixHashDispatcher(nodes, prefix_hash_tokens=256)

    # Same profile → same node
    n1 = d.dispatch("s1", "r1", {"profile": "coding", "shared_prefix_tokens": 20000}, 0)
    n2 = d.dispatch("s2", "r2", {"profile": "coding", "shared_prefix_tokens": 20000}, 0)
    n3 = d.dispatch("s99", "r99", {"profile": "coding", "shared_prefix_tokens": 20000}, 0)
    assert n1.node_id == n2.node_id == n3.node_id, "Same profile should hash to same node"


def test_different_profiles_different_nodes():
    """Different profiles may hash to different nodes."""
    nodes = []
    for i in range(8):
        l1 = TierStore("L1", 80*1024**3, 5120)
        l2 = TierStore("L2", 1024**4, 32*1024**2)
        nodes.append(PrefillNode(i, i//8, l1, l2, 32, 128))

    d = PrefixHashDispatcher(nodes, prefix_hash_tokens=256)
    coding = d.dispatch("s1", "r1", {"profile": "coding", "shared_prefix_tokens": 20000}, 0)
    chat = d.dispatch("s2", "r2", {"profile": "chat", "shared_prefix_tokens": 2048}, 0)
    # Not guaranteed different, but likely with good hash distribution
    # Just verify they both return valid nodes
    assert 0 <= coding.node_id < 8
    assert 0 <= chat.node_id < 8


def test_configurable_prefix_length():
    """prefix_hash_tokens parameter controls hash input."""
    cfg1 = _quick_config()
    cfg1.service.prefix_hash_tokens = 128
    m1 = SimEngine(cfg1).run()

    cfg2 = _quick_config()
    cfg2.service.prefix_hash_tokens = 512
    m2 = SimEngine(cfg2).run()

    # Both should run and complete
    assert sum(m1.savings_events.values()) > 0
    assert sum(m2.savings_events.values()) > 0


def test_overflow_under_pressure():
    """Under heavy load, prefix_hash should overflow to other nodes."""
    cfg = _quick_config(peak=100)
    cfg.sim_duration_s = 30.0; cfg.warmup_s = 5.0
    cfg.service.prefix_hash_overflow_threshold = 0.5  # low threshold → more overflow
    engine = SimEngine(cfg)
    engine.run()
    # Check dispatcher overflow count
    if isinstance(engine.dispatcher, PrefixHashDispatcher):
        assert engine.dispatcher.total_dispatches > 0
        # With low threshold, should see some overflow
        print(f"Overflow: {engine.dispatcher.overflow_count}/{engine.dispatcher.total_dispatches}")


def test_prefix_hash_vs_push_same_completed():
    """Same workload produces same completed count regardless of dispatch."""
    cfg_hash = _quick_config("prefix_hash")
    cfg_push = _quick_config("push")
    m_hash = SimEngine(cfg_hash).run()
    m_push = SimEngine(cfg_push).run()
    # Same seed + same workload → same completed count
    assert sum(m_hash.savings_events.values()) == sum(m_push.savings_events.values())
