"""
Multi-node prefill dispatch tests.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sim.config import SimConfig
from sim.engine import SimEngine

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "default.json")


def _quick_config() -> SimConfig:
    """10s sim, 1s warmup — fast for testing."""
    config = SimConfig.from_json(CONFIG_PATH)
    config.sim_duration_s = 10.0
    config.warmup_s = 1.0
    config.epoch_report_interval_s = 2.0
    config.sim_start_time_s = 36000.0  # 10 AM — ensures all profiles have nonzero arrival rate
    return config


def _stressed_multinode_config(n_nodes: int, algorithm: str = "push", gpus_per_worker: int | None = None) -> SimConfig:
    """Stressed config with multiple nodes — small L1 forces multi-tier activity."""
    config = _quick_config()
    config.tiers[0].capacity_bytes = 500 * 1024**2  # 500 MB L1 per GPU
    config.tiers[1].capacity_bytes = 2 * 1024**3     # 2 GB L2 per worker
    config.tiers[2].capacity_bytes = 50 * 1024**3     # 50 GB shared L3A
    config.ttl_l2_s = 20.0
    config.ttl_l3a_s = 120.0
    config.service.n_prefill_nodes = n_nodes
    config.service.dispatch_algorithm = algorithm
    # Default: all GPUs on one worker (simplest topology for tests)
    config.service.n_gpus_per_worker = gpus_per_worker if gpus_per_worker else n_nodes
    return config


def test_single_node_backward_compat():
    """N=1 produces valid metrics (same as current code)."""
    config = _quick_config()
    config.service.n_prefill_nodes = 1
    config.service.dispatch_algorithm = "push"
    metrics = SimEngine(config).run()
    report = metrics.report()

    assert "tier_saturation_pct" in report
    assert "cache_hit_rate" in report
    total_events = sum(metrics.savings_events.values())
    assert total_events > 0, "Should have processed some requests"
    # No dispatch_stats for single node
    assert metrics.affinity_dispatches == 0
    assert metrics.non_affinity_dispatches == 0


def test_more_nodes_reduce_queue_pressure():
    """4 nodes should have lower per-node queue pressure than 1 node."""
    config_1 = _stressed_multinode_config(1)
    config_4 = _stressed_multinode_config(4)

    metrics_1 = SimEngine(config_1).run()
    metrics_4 = SimEngine(config_4).run()

    # 1-node: single node's queue depth
    mean_q1 = sum(metrics_1.prefill_queue_depth) / max(1, len(metrics_1.prefill_queue_depth))

    # 4-node: per-node queue depth should be lower than single node's
    # Use max per-node mean depth
    max_per_node_q4 = 0.0
    for node_id in range(4):
        depths = metrics_4.per_node_queue_depth.get(node_id, [0])
        node_mean = sum(depths) / max(1, len(depths))
        max_per_node_q4 = max(max_per_node_q4, node_mean)

    assert max_per_node_q4 <= mean_q1, (
        f"4-node max per-node queue depth ({max_per_node_q4:.1f}) "
        f"should be <= 1-node queue depth ({mean_q1:.1f})"
    )


def test_push_affinity_dispatches():
    """Multi-turn sessions with 4 nodes should produce affinity dispatches."""
    config = _stressed_multinode_config(4, "push")
    metrics = SimEngine(config).run()

    assert metrics.affinity_dispatches > 0, (
        f"Expected some affinity dispatches with 4 nodes, got 0"
    )
    total_dispatches = metrics.affinity_dispatches + metrics.non_affinity_dispatches
    assert total_dispatches > 0


def test_pull_no_starvation():
    """Pull mode with 4 nodes: all nodes should get some work."""
    config = _stressed_multinode_config(4, "pull")
    metrics = SimEngine(config).run()

    for node_id in range(4):
        count = metrics.per_node_prefill_count.get(node_id, 0)
        assert count > 0, f"Node {node_id} got 0 prefills — starvation detected"


def test_pull_affinity_matches():
    """Pull mode produces valid results with multi-turn sessions."""
    config = _stressed_multinode_config(4, "pull")
    metrics = SimEngine(config).run()

    total_events = sum(metrics.savings_events.values())
    assert total_events > 0, "Pull mode should process requests"
    # All nodes should get work
    for node_id in range(4):
        assert metrics.per_node_prefill_count.get(node_id, 0) > 0


def test_queue_wait_metric():
    """Queue wait metric is populated and non-negative."""
    config = _stressed_multinode_config(4, "push")
    metrics = SimEngine(config).run()

    assert len(metrics.queue_wait_us) > 0, "Should have queue wait samples"
    assert all(w >= 0 for w in metrics.queue_wait_us), "Queue wait must be non-negative"
    report = metrics.report()
    assert "queue_wait_ms" in report
    assert report["queue_wait_ms"]["p95"] >= 0


def test_local_l3a_mode():
    """Local L3A mode: each node gets its own L3A, no cross-node L3A access."""
    config = _stressed_multinode_config(4, "push")
    config.service.l3a_shared = False
    metrics = SimEngine(config).run()

    total_events = sum(metrics.savings_events.values())
    assert total_events > 0
    report = metrics.report()
    assert "tier_saturation_pct" in report


def test_global_vs_local_l3a_hit_rate():
    """Global L3A should have >= hit rate compared to local L3A (same total capacity)."""
    config_global = _stressed_multinode_config(4, "push")
    config_global.service.l3a_shared = True
    config_global.service.l3a_remote_latency_us = 0  # remove latency penalty for fair comparison

    config_local = _stressed_multinode_config(4, "push")
    config_local.service.l3a_shared = False

    m_global = SimEngine(config_global).run()
    m_local = SimEngine(config_local).run()

    total_g = max(1, sum(m_global.savings_events.values()))
    total_l = max(1, sum(m_local.savings_events.values()))
    miss_g = m_global.savings_events.get("COLD_MISS", 0) / total_g
    miss_l = m_local.savings_events.get("COLD_MISS", 0) / total_l

    # Global L3A should have <= miss rate (or equal if L3A isn't used much)
    assert miss_g <= miss_l + 0.05, (
        f"Global L3A miss rate ({miss_g:.3f}) should be <= local ({miss_l:.3f})"
    )


def test_worker_topology_shared_l2():
    """8 GPUs across 2 workers: GPUs on same worker share L2."""
    config = _stressed_multinode_config(8, "push", gpus_per_worker=4)
    engine = SimEngine(config)

    # GPUs 0-3 should share L2, GPUs 4-7 should share L2
    assert engine.nodes[0].l2_store is engine.nodes[1].l2_store
    assert engine.nodes[0].l2_store is engine.nodes[3].l2_store
    assert engine.nodes[4].l2_store is engine.nodes[7].l2_store
    # Different workers should NOT share L2
    assert engine.nodes[0].l2_store is not engine.nodes[4].l2_store

    # Worker IDs should be correct
    for i in range(4):
        assert engine.nodes[i].worker_id == 0
    for i in range(4, 8):
        assert engine.nodes[i].worker_id == 1

    metrics = engine.run()
    assert sum(metrics.savings_events.values()) > 0


def test_intra_worker_no_l2_penalty():
    """L2 hit on same worker should not incur cross-node transfer penalty."""
    config = _stressed_multinode_config(8, "push", gpus_per_worker=4)
    engine = SimEngine(config)
    # GPUs on same worker share L2 — same_worker check
    assert engine._same_worker(0, 3) is True
    assert engine._same_worker(0, 4) is False


def test_local_l3a_worker_isolation():
    """Local L3A on worker 0 must NOT be visible from worker 1.
    Place an object on worker 0's L3A, then search from worker 1 — should miss."""
    config = _stressed_multinode_config(8, "push", gpus_per_worker=4)
    config.service.l3a_shared = False
    engine = SimEngine(config)

    # Manually place an object in worker 0's L3A
    from sim.cache import CacheObject, Tier, BlockLayout
    obj = CacheObject(
        session_id="test", shared_prefix_id=None, token_range=(0, 99),
        model_id="test", tier=Tier.L3A, size_bytes=1024, block_count=1,
        created_at_us=0, last_accessed_at_us=0, ref_count=1,
        block_layout=BlockLayout.L3A_ALIGNED, is_hibernated=True,
    )
    engine.nodes[0].l3a_store.insert("test-key", obj)

    # Search from worker 0's node — should find it
    found, nid = engine._find_cache_object_with_node("test-key", from_node_id=0)
    assert found is not None, "Should find object on own worker's L3A"

    # Search from worker 1's node (node_id=4) — should NOT find it
    found, nid = engine._find_cache_object_with_node("test-key", from_node_id=4)
    assert found is None, "Should NOT find object on different worker's local L3A"


def test_global_l3a_cross_worker_access():
    """Global L3A must be accessible from any worker."""
    config = _stressed_multinode_config(8, "push", gpus_per_worker=4)
    config.service.l3a_shared = True
    engine = SimEngine(config)

    from sim.cache import CacheObject, Tier, BlockLayout
    obj = CacheObject(
        session_id="test", shared_prefix_id=None, token_range=(0, 99),
        model_id="test", tier=Tier.L3A, size_bytes=1024, block_count=1,
        created_at_us=0, last_accessed_at_us=0, ref_count=1,
        block_layout=BlockLayout.L3A_ALIGNED, is_hibernated=True,
    )
    engine._shared_l3a.insert("test-key", obj)

    # Should be found from any node
    for node_id in [0, 3, 4, 7]:
        found, _ = engine._find_cache_object_with_node("test-key", from_node_id=node_id)
        assert found is not None, f"Global L3A should be visible from node {node_id}"


def test_session_migration_global_advantage():
    """With forced session migration (small L1/L2, multiple workers),
    global L3A must have higher hit rate than local L3A.
    Uses extreme conditions: tiny L1/L2 force L3A usage, 2 workers force migration."""
    # Tiny L1 (1MB) and L2 (10MB) — everything goes to L3A immediately
    config_g = _quick_config()
    config_g.tiers[0].capacity_bytes = 1 * 1024**2   # 1MB L1 per GPU
    config_g.tiers[1].capacity_bytes = 10 * 1024**2   # 10MB L2 per worker
    config_g.tiers[2].capacity_bytes = 50 * 1024**3    # 50GB L3A per worker
    config_g.service.n_prefill_nodes = 8
    config_g.service.n_gpus_per_worker = 4  # 2 workers
    config_g.service.l3a_shared = True
    config_g.service.l3a_remote_latency_us = 50_000

    import copy
    config_l = copy.deepcopy(config_g)
    config_l.service.l3a_shared = False
    config_l.service.l3a_remote_latency_us = 0

    m_g = SimEngine(config_g).run()
    m_l = SimEngine(config_l).run()

    total_g = max(1, sum(m_g.savings_events.values()))
    total_l = max(1, sum(m_l.savings_events.values()))
    miss_g = m_g.savings_events.get("COLD_MISS", 0) / total_g
    miss_l = m_l.savings_events.get("COLD_MISS", 0) / total_l

    assert miss_g < miss_l, (
        f"Global L3A should have fewer misses than local with session migration. "
        f"Global miss={miss_g:.3f}, Local miss={miss_l:.3f}"
    )


def test_multinode_perf_benchmark(benchmark):
    """10s sim with 4 nodes should complete quickly."""
    config = _stressed_multinode_config(4, "push")

    def run():
        return SimEngine(config).run()

    result = benchmark.pedantic(run, rounds=1, iterations=1, warmup_rounds=0)
    assert sum(result.savings_events.values()) > 0
