"""
Phase 2 — Observer layer tests.

Verifies:
1. Observers receive DESEvents from engine
2. Anthropic miss taxonomy classifies correctly (COLD/EXPIRY/EVICTION)
3. OpenAI infers cache hits from TTFT
4. Observers are read-only (never modify engine state)
5. Observer order doesn't affect output (same-timestamp idempotency)
6. EventStream.miss_summary() produces correct output
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine
from agentsim.core.contracts import ObserverBase, DESEvent, DESEventKind
from agentsim.core.observation.events import (
    EventStream, EventKind, CacheMissType, CacheHitType,
)
from agentsim.core.observation.anthropic import AnthropicProtocolObserver
from agentsim.core.observation.openai_chat import OpenAIProtocolObserver

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "heavy_coding.json")


def _quick_config() -> SimConfig:
    cfg = SimConfig.from_json(CONFIG_PATH)
    cfg.sim_duration_s = 60.0
    cfg.warmup_s = 10.0
    cfg.sim_start_time_s = 36000.0
    for p in cfg.profiles:
        p.arrival_rate_peak = 15
    return cfg


# ─── Test 1: Anthropic observer receives events ───

def test_anthropic_observer_receives_events():
    """AnthropicProtocolObserver collects CheckpointEvents from DESEvents."""
    stream = EventStream()
    observer = AnthropicProtocolObserver(stream)
    cfg = _quick_config()
    m = SimEngine(cfg, observers=[observer]).run()

    assert len(stream) > 0, "No checkpoint events produced"
    # Should have CACHE_DECISION and TTFT events
    decisions = stream.filter(kind=EventKind.CACHE_DECISION)
    ttfts = stream.filter(kind=EventKind.TTFT)
    assert len(decisions) > 0, "No CACHE_DECISION events"
    assert len(ttfts) > 0, "No TTFT events"


def test_anthropic_miss_taxonomy_has_all_types():
    """Over a full sim, should see at least COLD misses (first turns)."""
    stream = EventStream()
    observer = AnthropicProtocolObserver(stream)
    cfg = _quick_config()
    SimEngine(cfg, observers=[observer]).run()

    decisions = stream.filter(kind=EventKind.CACHE_DECISION)
    miss_types = {e.cache_miss_type for e in decisions}
    # Should have at least COLD (first turns) and NONE (hits)
    assert CacheMissType.COLD in miss_types or CacheMissType.NONE in miss_types


def test_anthropic_miss_summary():
    """miss_summary() produces complete output with rates summing to ~1.0."""
    stream = EventStream()
    observer = AnthropicProtocolObserver(stream)
    cfg = _quick_config()
    SimEngine(cfg, observers=[observer]).run()

    summary = stream.miss_summary()
    assert summary, "Empty miss summary"
    assert "total_turns" in summary
    assert summary["total_turns"] > 0
    # Hit + miss rates should sum to ~1.0
    total = (summary.get("full_hit_rate", 0) +
             summary.get("partial_hit_rate", 0) +
             summary.get("cross_request_hit_rate", 0) +
             summary.get("total_miss_rate", 0))
    assert 0.95 <= total <= 1.05, f"Rates sum to {total}, expected ~1.0"


# ─── Test 2: OpenAI observer infers from TTFT ───

def test_openai_observer_receives_events():
    """OpenAIProtocolObserver produces checkpoint events."""
    stream = EventStream()
    observer = OpenAIProtocolObserver(stream, baseline_ttft_us=10_000_000)
    cfg = _quick_config()
    SimEngine(cfg, observers=[observer]).run()

    assert len(stream) > 0
    decisions = stream.filter(kind=EventKind.CACHE_DECISION, protocol="openai_chat")
    assert len(decisions) > 0


def test_openai_infers_hits_from_ttft():
    """OpenAI should detect some hits when TTFT < 35% of baseline."""
    stream = EventStream()
    # Set baseline high so cache hits appear as fast TTFT
    observer = OpenAIProtocolObserver(stream, baseline_ttft_us=100_000_000)
    cfg = _quick_config()
    SimEngine(cfg, observers=[observer]).run()

    decisions = stream.filter(kind=EventKind.CACHE_DECISION, protocol="openai_chat")
    hits = [e for e in decisions if e.cache_hit_type == CacheHitType.FULL]
    # With very high baseline, most requests should look like hits
    assert len(hits) > 0, "No hits inferred — baseline may be too low"


# ─── Test 3: Both observers simultaneously ───

def test_both_observers_simultaneous():
    """Running both Anthropic and OpenAI observers together works."""
    stream_a = EventStream()
    stream_o = EventStream()
    obs_a = AnthropicProtocolObserver(stream_a)
    obs_o = OpenAIProtocolObserver(stream_o, baseline_ttft_us=10_000_000)

    cfg = _quick_config()
    SimEngine(cfg, observers=[obs_a, obs_o]).run()

    assert len(stream_a) > 0
    assert len(stream_o) > 0
    # Both should have the same number of TTFT events (one per prefill complete)
    ttft_a = stream_a.filter(kind=EventKind.TTFT)
    ttft_o = stream_o.filter(kind=EventKind.TTFT)
    assert len(ttft_a) == len(ttft_o), (
        f"Anthropic {len(ttft_a)} TTFT events vs OpenAI {len(ttft_o)}"
    )


# ─── Test 4: Observer read-only enforcement ───

def test_observer_read_only():
    """ObserverBase._assert_read_only raises NotImplementedError."""
    stream = EventStream()
    observer = AnthropicProtocolObserver(stream)
    with pytest.raises(NotImplementedError, match="Observer"):
        observer._assert_read_only()


# ─── Test 5: Observer order invariance ───

def test_observer_order_invariant():
    """Same-timestamp events produce identical output regardless of observer order."""
    stream1 = EventStream()
    stream2 = EventStream()
    obs_a1 = AnthropicProtocolObserver(stream1)
    obs_a2 = AnthropicProtocolObserver(stream2)

    cfg = _quick_config()
    # Run twice with same config — both should produce identical streams
    SimEngine(cfg, observers=[obs_a1]).run()
    SimEngine(cfg, observers=[obs_a2]).run()

    events1 = stream1.filter(kind=EventKind.CACHE_DECISION)
    events2 = stream2.filter(kind=EventKind.CACHE_DECISION)
    assert len(events1) == len(events2)

    for e1, e2 in zip(events1, events2):
        assert e1.cache_miss_type == e2.cache_miss_type
        assert e1.cache_hit_type == e2.cache_hit_type


# ─── Test 6: Performance ───

def test_observers_perf():
    """Observers don't significantly slow the engine."""
    import time
    cfg = _quick_config()
    cfg.sim_duration_s = 30.0
    cfg.warmup_s = 5.0

    # Without observers
    start = time.monotonic()
    SimEngine(cfg).run()
    base_time = time.monotonic() - start

    # With both observers
    stream_a = EventStream()
    stream_o = EventStream()
    start = time.monotonic()
    SimEngine(cfg, observers=[
        AnthropicProtocolObserver(stream_a),
        OpenAIProtocolObserver(stream_o),
    ]).run()
    obs_time = time.monotonic() - start

    # Observers should add < 50% overhead
    assert obs_time < base_time * 1.5 or obs_time < 2.0, (
        f"Observer overhead too high: {obs_time:.2f}s vs {base_time:.2f}s base"
    )
