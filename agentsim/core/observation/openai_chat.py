"""
OpenAI Chat Protocol Observer — infers cache hits from TTFT anomaly.

OpenAI Chat Completions API does not expose cache tokens.
Detection is inference-based: if TTFT < 35% of baseline → probable cache hit.
Cannot distinguish EXPIRY from COLD — both appear as PROBABLE_MISS.

This observer is READ-ONLY — it never modifies simulation state.
"""
from __future__ import annotations
from typing import Optional

from agentsim.core.contracts import ObserverBase, DESEvent, DESEventKind, ConfidenceLabel
from agentsim.core.observation.events import (
    CheckpointEvent, EventKind, EventStream,
    CacheMissType, CacheHitType,
)


class OpenAIProtocolObserver(ObserverBase):
    """
    Consumes DESEvents, infers cache hit/miss from TTFT ratio.
    Confidence: SEMI_CALIBRATED (inference-based, not explicit).
    """

    HIT_THRESHOLD_RATIO = 0.35  # TTFT < 35% of baseline → probable hit

    def __init__(self, stream: EventStream, baseline_ttft_us: float = 10_000_000):
        """baseline_ttft_us: expected cold-miss TTFT in microseconds."""
        self.stream = stream
        self.baseline_ttft_us = baseline_ttft_us
        self._session_turn_count: dict[str, int] = {}

    def on_event(self, event: DESEvent) -> None:
        if event.kind == DESEventKind.PREFILL_COMPLETE:
            self._on_prefill_complete(event)
        elif event.kind == DESEventKind.DECODE_COMPLETE:
            self._on_decode_complete(event)

    def _on_prefill_complete(self, event: DESEvent) -> None:
        p = event.payload
        session_id = p.get("session_id", "")
        ttft_us = p.get("ttft_us", 0)

        turn = self._session_turn_count.get(session_id, 0)
        self._session_turn_count[session_id] = turn + 1

        # Infer cache hit from TTFT ratio
        ratio = ttft_us / max(1, self.baseline_ttft_us)
        if ratio <= self.HIT_THRESHOLD_RATIO:
            hit_type = CacheHitType.FULL
            miss_type = CacheMissType.NONE
        else:
            hit_type = CacheHitType.NONE
            miss_type = CacheMissType.COLD  # can't distinguish expiry vs cold

        # Emit CACHE_DECISION (inferred)
        self.stream.append(CheckpointEvent(
            session_id=session_id,
            turn_id=turn,
            event_id=self.stream.next_id(),
            kind=EventKind.CACHE_DECISION,
            protocol="openai_chat",
            sim_time_ms=event.sim_time_us / 1000.0,
            cache_miss_type=miss_type,
            cache_hit_type=hit_type,
            raw_event={"inferred": True, "ttft_ratio": round(ratio, 4)},
        ))

        # Emit TTFT
        self.stream.append(CheckpointEvent(
            session_id=session_id,
            turn_id=turn,
            event_id=self.stream.next_id(),
            kind=EventKind.TTFT,
            protocol="openai_chat",
            sim_time_ms=event.sim_time_us / 1000.0,
            prefill_latency_ms=ttft_us / 1000.0,
        ))

    def _on_decode_complete(self, event: DESEvent) -> None:
        p = event.payload
        session_id = p.get("session_id", "")
        turn = self._session_turn_count.get(session_id, 1) - 1

        self.stream.append(CheckpointEvent(
            session_id=session_id,
            turn_id=turn,
            event_id=self.stream.next_id(),
            kind=EventKind.TURN_COMPLETE,
            protocol="openai_chat",
            sim_time_ms=event.sim_time_us / 1000.0,
            output_tokens=p.get("output_tokens", 0),
        ))
