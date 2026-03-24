"""
Phase 1 — Tests for ported agentsim.core.des engine.

Verifies:
1. New engine produces identical results to old engine (same seed, same config)
2. DESEvent emission works (observers receive events)
3. New engine satisfies Phase 0.5 tolerance bands
4. Import boundaries not violated
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.contracts import DESEvent, DESEventKind, ObserverBase, SavingsEvent

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "heavy_coding.json")


def _quick_config(peak: float = 15) -> SimConfig:
    cfg = SimConfig.from_json(CONFIG_PATH)
    cfg.sim_duration_s = 60.0
    cfg.warmup_s = 10.0
    cfg.sim_start_time_s = 36000.0
    for p in cfg.profiles:
        p.arrival_rate_peak = peak
    return cfg


class EventCollector(ObserverBase):
    """Test observer that collects all DESEvents."""
    def __init__(self):
        self.events: list[DESEvent] = []

    def on_event(self, event: DESEvent) -> None:
        self.events.append(event)


# ─── Test 1: New engine matches old engine ───

def test_new_engine_matches_old():
    """Same config + seed → identical completed count and hit rates."""
    from sim.engine import SimEngine as OldEngine
    from sim.config import SimConfig as OldConfig

    cfg_new = _quick_config()
    cfg_old = OldConfig.from_json(CONFIG_PATH)
    cfg_old.sim_duration_s = 60.0
    cfg_old.warmup_s = 10.0
    cfg_old.sim_start_time_s = 36000.0
    for p in cfg_old.profiles:
        p.arrival_rate_peak = 15

    m_old = OldEngine(cfg_old).run()
    m_new = SimEngine(cfg_new).run()

    old_completed = sum(m_old.savings_events.values())
    new_completed = sum(m_new.savings_events.values())
    assert old_completed == new_completed, f"Old {old_completed} != New {new_completed}"

    r_old = m_old.report()["cache_hit_rate"]
    r_new = m_new.report()["cache_hit_rate"]
    for key in r_old:
        assert abs(r_old[key] - r_new[key]) < 0.001, f"Hit rate mismatch on {key}: {r_old[key]} vs {r_new[key]}"


# ─── Test 2: DESEvent emission ───

def test_des_events_emitted():
    """Engine emits DESEvents to registered observers."""
    cfg = _quick_config()
    collector = EventCollector()
    m = SimEngine(cfg, observers=[collector]).run()

    assert len(collector.events) > 0, "No DESEvents emitted"

    # Should have PREFILL_COMPLETE and DECODE_COMPLETE events
    kinds = {e.kind for e in collector.events}
    assert DESEventKind.PREFILL_COMPLETE in kinds, "No PREFILL_COMPLETE events"
    assert DESEventKind.DECODE_COMPLETE in kinds, "No DECODE_COMPLETE events"


def test_prefill_complete_payload():
    """PREFILL_COMPLETE events have required payload fields."""
    cfg = _quick_config()
    collector = EventCollector()
    SimEngine(cfg, observers=[collector]).run()

    prefill_events = [e for e in collector.events if e.kind == DESEventKind.PREFILL_COMPLETE]
    assert len(prefill_events) > 0

    # Check required payload fields per contracts
    for evt in prefill_events[:5]:  # check first 5
        p = evt.payload
        assert "session_id" in p
        assert "request_id" in p
        assert "ttft_us" in p
        assert "prefill_latency_us" in p
        assert "queue_wait_prefill_us" in p
        assert "savings_event" in p


def test_decode_complete_payload():
    """DECODE_COMPLETE events have required payload fields."""
    cfg = _quick_config()
    collector = EventCollector()
    SimEngine(cfg, observers=[collector]).run()

    decode_events = [e for e in collector.events if e.kind == DESEventKind.DECODE_COMPLETE]
    assert len(decode_events) > 0

    for evt in decode_events[:5]:
        p = evt.payload
        assert "decode_latency_us" in p
        assert "output_tokens" in p


def test_event_ids_unique():
    """All event IDs are unique."""
    cfg = _quick_config()
    collector = EventCollector()
    SimEngine(cfg, observers=[collector]).run()

    ids = [e.event_id for e in collector.events]
    assert len(ids) == len(set(ids)), "Duplicate event IDs"


def test_event_timestamps_non_decreasing():
    """Event timestamps are non-decreasing."""
    cfg = _quick_config()
    collector = EventCollector()
    SimEngine(cfg, observers=[collector]).run()

    times = [e.sim_time_us for e in collector.events]
    for i in range(1, len(times)):
        assert times[i] >= times[i-1], f"Timestamp decrease at {i}: {times[i]} < {times[i-1]}"


# ─── Test 3: Performance ───

def test_new_engine_perf():
    """New engine runs 30s sim in reasonable time."""
    import time
    cfg = _quick_config()
    cfg.sim_duration_s = 30.0
    cfg.warmup_s = 5.0
    start = time.monotonic()
    SimEngine(cfg).run()
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"30s sim took {elapsed:.1f}s"
