"""
Anthropic Protocol Observer — consumes DESEvents, produces CheckpointEvents.

Implements the Anthropic miss taxonomy:
  COLD:     first turn of session, no cached prefix
  EXPIRY:   had cache, idle > 5 min TTL → invalidated
  EVICTION: had cache, idle < 5 min, but evicted under memory pressure
  TRANSFER: cache hit but required cross-worker KV transfer

This observer is READ-ONLY — it never modifies simulation state.
"""
from __future__ import annotations
from typing import Optional

from agentsim.core.contracts import ObserverBase, DESEvent, DESEventKind
from agentsim.core.observation.events import (
    CheckpointEvent, EventKind, EventStream,
    CacheMissType, CacheHitType,
)


class AnthropicProtocolObserver(ObserverBase):
    """
    Consumes DESEvents from the engine, produces Anthropic-protocol
    CheckpointEvents with cache miss classification.
    """

    TTL_US = 300 * 1_000_000  # 5-minute idle TTL in microseconds

    def __init__(self, stream: EventStream):
        self.stream = stream
        # Per-session tracking for miss classification
        self._session_turn_count: dict[str, int] = {}
        self._session_last_write_us: dict[str, int] = {}  # last sim_time of a cache write
        self._session_last_event_us: dict[str, int] = {}  # last sim_time of any event

    def on_event(self, event: DESEvent) -> None:
        if event.kind == DESEventKind.PREFILL_COMPLETE:
            self._on_prefill_complete(event)
        elif event.kind == DESEventKind.DECODE_COMPLETE:
            self._on_decode_complete(event)

    def _on_prefill_complete(self, event: DESEvent) -> None:
        p = event.payload
        session_id = p.get("session_id", "")
        request_id = p.get("request_id", "")
        ttft_us = p.get("ttft_us", 0)
        prefill_us = p.get("prefill_latency_us", 0)
        cached_tokens = p.get("cached_tokens", 0)
        uncached_tokens = p.get("uncached_tokens", 0)
        cache_tier = p.get("cache_tier", "cold_miss")

        # Track turn count
        turn = self._session_turn_count.get(session_id, 0)
        self._session_turn_count[session_id] = turn + 1

        # Determine idle time since last event
        last_event_us = self._session_last_event_us.get(session_id, 0)
        idle_us = event.sim_time_us - last_event_us if last_event_us > 0 else 0

        # Classify miss type
        is_hit = "hit" in cache_tier.lower() and cached_tokens > 0
        had_previous_write = session_id in self._session_last_write_us

        if is_hit and cached_tokens > 0 and uncached_tokens <= 1:
            miss_type = CacheMissType.NONE
            hit_type = CacheHitType.FULL
        elif is_hit and cached_tokens > 0:
            miss_type = CacheMissType.NONE
            hit_type = CacheHitType.PARTIAL
        elif turn == 0 and not had_previous_write:
            miss_type = CacheMissType.COLD
            hit_type = CacheHitType.NONE
        elif had_previous_write and idle_us >= self.TTL_US:
            miss_type = CacheMissType.EXPIRY
            hit_type = CacheHitType.NONE
        elif had_previous_write and idle_us < self.TTL_US:
            miss_type = CacheMissType.EVICTION
            hit_type = CacheHitType.NONE
        else:
            miss_type = CacheMissType.COLD
            hit_type = CacheHitType.NONE

        # Emit CACHE_DECISION checkpoint
        self.stream.append(CheckpointEvent(
            session_id=session_id,
            turn_id=turn,
            event_id=self.stream.next_id(),
            kind=EventKind.CACHE_DECISION,
            protocol="anthropic",
            sim_time_ms=event.sim_time_us / 1000.0,
            cache_read_tokens=cached_tokens,
            cache_written_tokens=uncached_tokens,
            cache_miss_type=miss_type,
            cache_hit_type=hit_type,
            input_tokens=cached_tokens + uncached_tokens,
            prefill_latency_ms=prefill_us / 1000.0,
        ))

        # Emit TTFT checkpoint
        self.stream.append(CheckpointEvent(
            session_id=session_id,
            turn_id=turn,
            event_id=self.stream.next_id(),
            kind=EventKind.TTFT,
            protocol="anthropic",
            sim_time_ms=event.sim_time_us / 1000.0,
            prefill_latency_ms=ttft_us / 1000.0,
        ))

        # Update session tracking
        if cached_tokens > 0 or uncached_tokens > 0:
            self._session_last_write_us[session_id] = event.sim_time_us
        self._session_last_event_us[session_id] = event.sim_time_us

    def _on_decode_complete(self, event: DESEvent) -> None:
        p = event.payload
        session_id = p.get("session_id", "")
        turn = self._session_turn_count.get(session_id, 1) - 1

        self.stream.append(CheckpointEvent(
            session_id=session_id,
            turn_id=turn,
            event_id=self.stream.next_id(),
            kind=EventKind.TURN_COMPLETE,
            protocol="anthropic",
            sim_time_ms=event.sim_time_us / 1000.0,
            output_tokens=p.get("output_tokens", 0),
        ))
        self._session_last_event_us[session_id] = event.sim_time_us
