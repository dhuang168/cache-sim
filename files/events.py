"""
core/observation/events.py    ← CORRECT FINAL LOCATION
(currently at core/events.py — move during Phase 2)

⚠️  ONE POSITIONING CHANGE REQUIRED (Phase 2):

This file's content is correct and preserved.
The miss taxonomy (COLD/EXPIRY/EVICTION/TRANSFER), CheckpointEvent,
EventStream, and miss_summary() are all aligned with the final plan.

ONE THING THAT MUST CHANGE in Phase 2:
  AnthropicEventMapper and OpenAIChatEventMapper currently receive
  explicit method calls from external callers:

    mapper.on_message_start(session_id, sim_time_ms, ...)  ← WRONG
    mapper.on_first_content_delta(...)                      ← WRONG

  Final plan requires mappers to implement ObserverBase:

    class AnthropicEventMapper(ObserverBase):
        def on_event(self, event: DESEvent) -> None:
            if event.kind == DESEventKind.PREFILL_COMPLETE:
                self._on_prefill_complete(event)
            ...

  The logic inside each method is correct — only the call direction
  changes. Mappers are called BY the DES engine, not by external code.
  Mappers are read-only — they never modify simulation state.

Until Phase 2: this file is usable for Goal 2 development and testing.
The explicit-call API is acceptable as a temporary interface.

LAYER: Observation Layer (Layer 2) — downstream only.
DEPENDENCY: May read DESEvent payloads. Must never write to Core DES state.
MIT License.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Cache miss taxonomy
# ---------------------------------------------------------------------------

class CacheMissType(str, Enum):
    NONE       = "NONE"       # hit — no miss
    COLD       = "COLD"       # never seen this prefix before
    EXPIRY     = "EXPIRY"     # seen before but Anthropic 5-min TTL expired
    EVICTION   = "EVICTION"   # seen before but evicted under memory pressure
    TRANSFER   = "TRANSFER"   # hit but required cross-node KV transfer (latency)


class CacheHitType(str, Enum):
    NONE       = "NONE"       # miss — no hit
    FULL       = "FULL"       # entire prefix served from cache
    PARTIAL    = "PARTIAL"    # prefix grew — partial hit + new write
    CROSS_REQ  = "CROSS_REQ"  # hit from a DIFFERENT session (shared prefix)


# ---------------------------------------------------------------------------
# Checkpoint event kinds
# ---------------------------------------------------------------------------

class EventKind(str, Enum):
    # --- Goal 1 + Goal 2 + Goal 3 ---
    SESSION_START        = "SESSION_START"       # session begins
    SESSION_END          = "SESSION_END"         # session complete

    TURN_START           = "TURN_START"          # request fired

    # --- Goal 2: Protocol-level checkpoints ---
    CACHE_DECISION       = "CACHE_DECISION"      # cache_read / cache_write known
                                                  # (Anthropic: from message_start.usage)
                                                  # (OpenAI:    inferred from TTFT)
    TTFT                 = "TTFT"                # first content token received
    BATCH_BOUNDARY       = "BATCH_BOUNDARY"      # request joined an in-flight batch
    EVICTION_TRIGGER     = "EVICTION_TRIGGER"    # KV cache pressure — eviction fired
    STREAMING_STALL      = "STREAMING_STALL"     # inter-token gap exceeded threshold
    TURN_COMPLETE        = "TURN_COMPLETE"       # last token / finish_reason received

    # --- Anthropic-specific ---
    EXPIRY_BOUNDARY      = "EXPIRY_BOUNDARY"     # 5-min TTL crossed between turns
    PREFIX_REUSE         = "PREFIX_REUSE"        # cross-session prefix hit detected

    # --- Goal 3: Framework interop ---
    KV_TRANSFER_START    = "KV_TRANSFER_START"   # P/D disagg: KV leaving prefill node
    KV_TRANSFER_COMPLETE = "KV_TRANSFER_COMPLETE"# KV arrived at decode node
    FRAMEWORK_ERROR      = "FRAMEWORK_ERROR"     # framework returned unexpected event


# ---------------------------------------------------------------------------
# The unified checkpoint event
# ---------------------------------------------------------------------------

@dataclass
class CheckpointEvent:
    # --- Identity ---
    session_id:    str
    turn_id:       int
    event_id:      int            # monotonic counter within simulation

    # --- What kind of event ---
    kind:          EventKind
    protocol:      Literal[
        "anthropic",
        "openai_chat",
        "openai_responses",
        "internal",               # simulator-internal, not from a real API
    ]

    # --- Timing ---
    sim_time_ms:   float          # SimPy simulation time
    wall_time_ms:  Optional[float] = None  # real wall clock (Goal 3 only)

    # --- Cache state at this checkpoint ---
    cache_read_tokens:    int = 0      # tokens served from existing cache
    cache_written_tokens: int = 0      # tokens written to new cache entry
    cache_miss_type:      CacheMissType  = CacheMissType.NONE
    cache_hit_type:       CacheHitType   = CacheHitType.NONE

    # --- Latency intervals (filled in as events arrive) ---
    prefill_latency_ms:    Optional[float] = None   # TURN_START → TTFT
    decode_throughput_tps: Optional[float] = None   # tokens/s during decode
    inter_token_gap_ms:    Optional[float] = None   # for STREAMING_STALL
    kv_transfer_ms:        Optional[float] = None   # for KV_TRANSFER_COMPLETE

    # --- Token counts ---
    input_tokens:          Optional[int]   = None
    output_tokens:         Optional[int]   = None
    new_tokens:            Optional[int]   = None   # = input - cached

    # --- Hardware context ---
    chip_name:             Optional[str]   = None
    hbm_utilization_pct:   Optional[float] = None
    ddr_utilization_pct:   Optional[float] = None

    # --- Framework context (Goal 3) ---
    framework:             Optional[str]   = None   # "vllm" | "sglang" | "native"
    batch_size:            Optional[int]   = None

    # --- Raw protocol payload (for debugging Goal 2 / Goal 3) ---
    raw_event:             Optional[dict]  = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Event stream — collects all events for one simulation run
# ---------------------------------------------------------------------------

class EventStream:
    """
    Append-only log of all checkpoint events.
    Supports filtering and miss analysis.
    """

    def __init__(self):
        self._events: list[CheckpointEvent] = []
        self._counter = 0

    def append(self, event: CheckpointEvent) -> None:
        self._events.append(event)

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def __len__(self) -> int:
        return len(self._events)

    def filter(
        self,
        kind:       Optional[EventKind] = None,
        session_id: Optional[str]       = None,
        protocol:   Optional[str]       = None,
    ) -> list[CheckpointEvent]:
        result = self._events
        if kind:
            result = [e for e in result if e.kind == kind]
        if session_id:
            result = [e for e in result if e.session_id == session_id]
        if protocol:
            result = [e for e in result if e.protocol == protocol]
        return result

    # ------------------------------------------------------------------
    # Cache miss analysis — the core Goal 2 output
    # ------------------------------------------------------------------

    def miss_summary(self) -> dict:
        """
        Summarize all cache decisions observed in the stream.
        This is the primary Goal 2 deliverable.
        """
        decisions = self.filter(kind=EventKind.CACHE_DECISION)
        if not decisions:
            return {}

        total        = len(decisions)
        cold_misses  = sum(1 for e in decisions if e.cache_miss_type == CacheMissType.COLD)
        expiry_misses= sum(1 for e in decisions if e.cache_miss_type == CacheMissType.EXPIRY)
        evict_misses = sum(1 for e in decisions if e.cache_miss_type == CacheMissType.EVICTION)
        full_hits    = sum(1 for e in decisions if e.cache_hit_type  == CacheHitType.FULL)
        partial_hits = sum(1 for e in decisions if e.cache_hit_type  == CacheHitType.PARTIAL)
        cross_req    = sum(1 for e in decisions if e.cache_hit_type  == CacheHitType.CROSS_REQ)

        total_read_tokens    = sum(e.cache_read_tokens    for e in decisions)
        total_written_tokens = sum(e.cache_written_tokens for e in decisions)

        return {
            "total_turns":            total,
            "full_hit_rate":          full_hits    / total,
            "partial_hit_rate":       partial_hits / total,
            "cross_request_hit_rate": cross_req    / total,
            "cold_miss_rate":         cold_misses  / total,
            "expiry_miss_rate":       expiry_misses/ total,
            "eviction_miss_rate":     evict_misses / total,
            "total_miss_rate":        (cold_misses + expiry_misses + evict_misses) / total,
            "total_tokens_read_from_cache":    total_read_tokens,
            "total_tokens_written_to_cache":   total_written_tokens,
            "cache_efficiency_ratio":
                total_read_tokens / max(1, total_read_tokens + total_written_tokens),
        }

    def ttft_percentiles(self) -> dict:
        """P50, P90, P99 TTFT across all turns."""
        ttfts = [
            e.prefill_latency_ms
            for e in self.filter(kind=EventKind.TTFT)
            if e.prefill_latency_ms is not None
        ]
        if not ttfts:
            return {}
        ttfts.sort()
        n = len(ttfts)
        return {
            "p50_ms":  ttfts[int(n * 0.50)],
            "p90_ms":  ttfts[int(n * 0.90)],
            "p99_ms":  ttfts[int(n * 0.99)],
            "mean_ms": sum(ttfts) / n,
        }

    def expiry_events(self) -> list[CheckpointEvent]:
        """All turns where the Anthropic 5-min TTL was crossed."""
        return self.filter(kind=EventKind.EXPIRY_BOUNDARY)

    def stall_events(self) -> list[CheckpointEvent]:
        """All streaming stall events (inter-token gap exceeded threshold)."""
        return self.filter(kind=EventKind.STREAMING_STALL)


# ---------------------------------------------------------------------------
# Protocol → CheckpointEvent mappers
# These parse real or simulated SSE event dicts into CheckpointEvents
# ---------------------------------------------------------------------------

class AnthropicEventMapper:
    """
    Maps Anthropic Messages API SSE events to CheckpointEvents.

    Cache miss detection logic:
        COLD:    first turn of session AND cache_read == 0
        EXPIRY:  cache_read == 0 AND previous turn had cache_write > 0
                 AND think_time >= TTL
        PARTIAL: cache_read > 0 AND cache_write > 0
        FULL:    cache_read > 0 AND cache_write == 0
    """

    def __init__(self, stream: EventStream, ttl_s: float = 300.0):
        self.stream = stream
        self.ttl_s  = ttl_s
        # Track per-session state for expiry detection
        self._session_last_write:     dict[str, int]   = {}  # session_id → tokens written
        self._session_last_turn_time: dict[str, float] = {}  # session_id → sim_time_ms

    def on_message_start(
        self,
        session_id:          str,
        turn_id:             int,
        sim_time_ms:         float,
        input_tokens:        int,
        cache_read_tokens:   int,
        cache_written_tokens:int,
        think_time_ms:       float,
        raw:                 Optional[dict] = None,
    ) -> CheckpointEvent:
        """Called when message_start event arrives. This is the CACHE_DECISION."""

        is_first_turn = (turn_id == 0)
        prev_write    = self._session_last_write.get(session_id, 0)
        think_time_s  = think_time_ms / 1000.0

        # Classify miss type
        if cache_read_tokens > 0 and cache_written_tokens == 0:
            miss_type = CacheMissType.NONE
            hit_type  = CacheHitType.FULL

        elif cache_read_tokens > 0 and cache_written_tokens > 0:
            miss_type = CacheMissType.NONE
            hit_type  = CacheHitType.PARTIAL

        elif cache_read_tokens == 0 and is_first_turn:
            miss_type = CacheMissType.COLD
            hit_type  = CacheHitType.NONE

        elif cache_read_tokens == 0 and prev_write > 0 and think_time_s >= self.ttl_s:
            miss_type = CacheMissType.EXPIRY
            hit_type  = CacheHitType.NONE

        elif cache_read_tokens == 0 and prev_write > 0 and think_time_s < self.ttl_s:
            # Had cache before, think time was short, but missed → eviction
            miss_type = CacheMissType.EVICTION
            hit_type  = CacheHitType.NONE

        else:
            miss_type = CacheMissType.COLD
            hit_type  = CacheHitType.NONE

        # Update session tracking
        self._session_last_write[session_id]     = cache_written_tokens
        self._session_last_turn_time[session_id] = sim_time_ms

        event = CheckpointEvent(
            session_id           = session_id,
            turn_id              = turn_id,
            event_id             = self.stream.next_id(),
            kind                 = EventKind.CACHE_DECISION,
            protocol             = "anthropic",
            sim_time_ms          = sim_time_ms,
            cache_read_tokens    = cache_read_tokens,
            cache_written_tokens = cache_written_tokens,
            cache_miss_type      = miss_type,
            cache_hit_type       = hit_type,
            input_tokens         = input_tokens,
            raw_event            = raw,
        )
        self.stream.append(event)
        return event

    def on_first_content_delta(
        self,
        session_id:      str,
        turn_id:         int,
        sim_time_ms:     float,
        turn_start_ms:   float,
        raw:             Optional[dict] = None,
    ) -> CheckpointEvent:
        """Called when first content_block_delta arrives. This is TTFT."""
        event = CheckpointEvent(
            session_id         = session_id,
            turn_id            = turn_id,
            event_id           = self.stream.next_id(),
            kind               = EventKind.TTFT,
            protocol           = "anthropic",
            sim_time_ms        = sim_time_ms,
            prefill_latency_ms = sim_time_ms - turn_start_ms,
            raw_event          = raw,
        )
        self.stream.append(event)
        return event

    def on_message_stop(
        self,
        session_id:    str,
        turn_id:       int,
        sim_time_ms:   float,
        output_tokens: int,
        raw:           Optional[dict] = None,
    ) -> CheckpointEvent:
        """Called when message_stop arrives. This is TURN_COMPLETE."""
        event = CheckpointEvent(
            session_id    = session_id,
            turn_id       = turn_id,
            event_id      = self.stream.next_id(),
            kind          = EventKind.TURN_COMPLETE,
            protocol      = "anthropic",
            sim_time_ms   = sim_time_ms,
            output_tokens = output_tokens,
            raw_event     = raw,
        )
        self.stream.append(event)
        return event


class OpenAIChatEventMapper:
    """
    Maps OpenAI Chat Completions SSE events to CheckpointEvents.
    Cache signals are inferred from TTFT anomaly detection since
    OpenAI does not expose cache_read_tokens in the stream.
    """

    # TTFT significantly faster than baseline → probable cache hit
    HIT_THRESHOLD_RATIO = 0.35

    def __init__(self, stream: EventStream, baseline_ttft_ms: float = 1000.0):
        self.stream           = stream
        self.baseline_ttft_ms = baseline_ttft_ms

    def on_first_chunk(
        self,
        session_id:    str,
        turn_id:       int,
        sim_time_ms:   float,
        turn_start_ms: float,
        raw:           Optional[dict] = None,
    ) -> CheckpointEvent:
        ttft = sim_time_ms - turn_start_ms

        # Infer cache hit/miss from TTFT ratio
        ratio = ttft / max(1.0, self.baseline_ttft_ms)
        if ratio <= self.HIT_THRESHOLD_RATIO:
            hit_type  = CacheHitType.FULL
            miss_type = CacheMissType.NONE
        else:
            hit_type  = CacheHitType.NONE
            miss_type = CacheMissType.COLD   # can't distinguish expiry vs cold

        # Emit CACHE_DECISION (inferred)
        cache_event = CheckpointEvent(
            session_id      = session_id,
            turn_id         = turn_id,
            event_id        = self.stream.next_id(),
            kind            = EventKind.CACHE_DECISION,
            protocol        = "openai_chat",
            sim_time_ms     = sim_time_ms,
            cache_miss_type = miss_type,
            cache_hit_type  = hit_type,
            raw_event       = {"inferred": True, "ttft_ratio": ratio},
        )
        self.stream.append(cache_event)

        # Emit TTFT
        ttft_event = CheckpointEvent(
            session_id         = session_id,
            turn_id            = turn_id,
            event_id           = self.stream.next_id(),
            kind               = EventKind.TTFT,
            protocol           = "openai_chat",
            sim_time_ms        = sim_time_ms,
            prefill_latency_ms = ttft,
            raw_event          = raw,
        )
        self.stream.append(ttft_event)
        return ttft_event
