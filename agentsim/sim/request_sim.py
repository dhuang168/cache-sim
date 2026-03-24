"""
agentsim/sim/request_sim.py

Goal 1: Fast request-level SimPy simulator.
Drives agentic sessions at scale — billions of tokens,
tens of parallel workers.

Granularity: one SimPy event per turn (TURN_START + TURN_COMPLETE).
No token stepping. No streaming simulation.
Everything inside a turn is one analytical latency calculation.
MIT License.
"""

from __future__ import annotations
import simpy
import random
from dataclasses import dataclass
from typing import Optional

from agentsim.core.hardware_model import HardwareModel
from agentsim.core.session_model  import AgenticSessionGenerator, AgenticSession, Turn
from agentsim.core.events         import (
    EventStream, CheckpointEvent, EventKind,
    CacheMissType, CacheHitType, AnthropicEventMapper
)


# ---------------------------------------------------------------------------
# KV Cache State — tracks what's in HBM and DDR per session
# ---------------------------------------------------------------------------

@dataclass
class SessionCacheState:
    """Per-session KV cache tracking."""
    session_id:        str
    hbm_tokens:        int   = 0   # tokens currently in HBM
    ddr_tokens:        int   = 0   # tokens currently in DDR
    last_access_ms:    float = 0.0
    prefix_hash:       int   = 0   # simple hash for cross-session detection


class KVCacheManager:
    """
    Manages KV cache across all active sessions.
    Tracks HBM and DDR tier occupancy.
    Fires EVICTION_TRIGGER events when HBM is full.
    """

    def __init__(
        self,
        hw_model:     HardwareModel,
        stream:       EventStream,
        evict_to_ddr: bool = True,    # True = evict to DDR, False = drop
    ):
        self.hw          = hw_model
        self.stream      = stream
        self.evict_to_ddr= evict_to_ddr

        # Convert GB capacity to token capacity
        bytes_per_token  = hw_model.model.total_bytes_per_token_kv
        self.hbm_max_tokens = int(
            hw_model.chip.hbm_capacity_gb * 1e9 / bytes_per_token
        )
        self.ddr_max_tokens = int(
            (hw_model.chip.ddr_capacity_gb or 0) * 1e9 / bytes_per_token
        )

        self.hbm_used_tokens = 0
        self.ddr_used_tokens = 0
        self._sessions: dict[str, SessionCacheState] = {}

    @property
    def hbm_utilization(self) -> float:
        return self.hbm_used_tokens / max(1, self.hbm_max_tokens)

    @property
    def ddr_utilization(self) -> float:
        if self.ddr_max_tokens == 0:
            return 0.0
        return self.ddr_used_tokens / self.ddr_max_tokens

    def lookup(
        self,
        session_id:   str,
        prefix_tokens: int,
        sim_time_ms:  float,
        ttl_ms:       float = 300_000,
    ) -> tuple[int, str, CacheMissType, CacheHitType]:
        """
        Look up how many tokens of the prefix are cached and where.

        Returns:
            cached_tokens:  how many tokens were found
            location:       "hbm" | "ddr" | "cold"
            miss_type
            hit_type
        """
        state = self._sessions.get(session_id)

        if state is None:
            return 0, "cold", CacheMissType.COLD, CacheHitType.NONE

        age_ms = sim_time_ms - state.last_access_ms

        if age_ms >= ttl_ms:
            # TTL expired
            self._evict_session(session_id)
            return 0, "cold", CacheMissType.EXPIRY, CacheHitType.NONE

        cached = min(state.hbm_tokens + state.ddr_tokens, prefix_tokens)
        hbm_cached = min(state.hbm_tokens, prefix_tokens)

        if hbm_cached >= prefix_tokens:
            return cached, "hbm", CacheMissType.NONE, CacheHitType.FULL
        elif cached >= prefix_tokens:
            return cached, "ddr", CacheMissType.NONE, CacheHitType.PARTIAL
        elif cached > 0:
            return cached, "ddr", CacheMissType.NONE, CacheHitType.PARTIAL
        else:
            return 0, "cold", CacheMissType.COLD, CacheHitType.NONE

    def update(
        self,
        session_id:   str,
        new_tokens:   int,
        sim_time_ms:  float,
    ) -> None:
        """Record new KV tokens written after a turn completes."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionCacheState(session_id=session_id)

        state = self._sessions[session_id]
        state.last_access_ms = sim_time_ms

        # Try to fit in HBM
        hbm_available = self.hbm_max_tokens - self.hbm_used_tokens
        if new_tokens <= hbm_available:
            state.hbm_tokens     += new_tokens
            self.hbm_used_tokens += new_tokens
        else:
            # HBM full — need to evict before writing
            self._make_room_hbm(new_tokens, sim_time_ms)
            state.hbm_tokens     += new_tokens
            self.hbm_used_tokens += new_tokens

    def _make_room_hbm(self, needed_tokens: int, sim_time_ms: float) -> None:
        """Evict LRU sessions from HBM to make room."""
        by_age = sorted(
            self._sessions.values(),
            key=lambda s: s.last_access_ms
        )
        freed = 0
        for state in by_age:
            if freed >= needed_tokens:
                break
            evict_tokens = state.hbm_tokens
            state.hbm_tokens      = 0
            self.hbm_used_tokens -= evict_tokens
            freed                += evict_tokens

            if self.evict_to_ddr and self.ddr_max_tokens > 0:
                ddr_available = self.ddr_max_tokens - self.ddr_used_tokens
                move = min(evict_tokens, ddr_available)
                state.ddr_tokens     += move
                self.ddr_used_tokens += move

            # Record eviction event
            self.stream.append(CheckpointEvent(
                session_id      = state.session_id,
                turn_id         = -1,
                event_id        = self.stream.next_id(),
                kind            = EventKind.EVICTION_TRIGGER,
                protocol        = "internal",
                sim_time_ms     = sim_time_ms,
                cache_miss_type = CacheMissType.EVICTION,
                hbm_utilization_pct = self.hbm_utilization * 100,
                ddr_utilization_pct = self.ddr_utilization * 100,
            ))

    def _evict_session(self, session_id: str) -> None:
        state = self._sessions.pop(session_id, None)
        if state:
            self.hbm_used_tokens -= state.hbm_tokens
            self.ddr_used_tokens -= state.ddr_tokens


# ---------------------------------------------------------------------------
# SimPy simulation — request level
# ---------------------------------------------------------------------------

class RequestLevelSim:
    """
    Goal 1 simulator.
    Request-level granularity. No token stepping.
    One SimPy event per turn: TURN_START fires, analytical latency computed,
    TURN_COMPLETE fires, think time sampled, next TURN_START scheduled.
    """

    def __init__(
        self,
        hw_model:    HardwareModel,
        generator:   AgenticSessionGenerator,
        stream:      EventStream,
        num_sessions: int         = 1000,
        max_sim_time_ms: float    = 3_600_000,  # 1 hour of sim time
        framework:   str          = "native",
    ):
        self.hw         = hw_model
        self.generator  = generator
        self.stream     = stream
        self.num_sessions = num_sessions
        self.max_sim_time = max_sim_time_ms
        self.framework  = framework

        self.env     = simpy.Environment()
        self.kv_mgr  = KVCacheManager(hw_model=hw_model, stream=stream)
        self.ant_mapper = AnthropicEventMapper(stream=stream)

    def run(self) -> EventStream:
        """Run the full simulation. Returns the populated EventStream."""
        self.env.process(self._session_arrival_loop())
        self.env.run(until=self.max_sim_time)
        return self.stream

    # ------------------------------------------------------------------
    # SimPy processes
    # ------------------------------------------------------------------

    def _session_arrival_loop(self):
        """Poisson session arrivals."""
        for i in range(self.num_sessions):
            session = self.generator.generate_session()
            self.env.process(self._run_session(session))
            gap_ms = self.generator.next_arrival_gap_s() * 1000
            yield self.env.timeout(gap_ms)

    def _run_session(self, session: AgenticSession):
        """Drive one agentic session through all its turns."""
        sim_time = self.env.now

        # SESSION_START event
        self.stream.append(CheckpointEvent(
            session_id  = session.session_id,
            turn_id     = 0,
            event_id    = self.stream.next_id(),
            kind        = EventKind.SESSION_START,
            protocol    = "internal",
            sim_time_ms = sim_time,
            framework   = self.framework,
        ))

        prev_think_time_ms = 0.0

        for turn in session.turns:
            sim_time = self.env.now

            # 1. Cache lookup
            cached_tokens, cache_loc, miss_type, hit_type = self.kv_mgr.lookup(
                session_id    = session.session_id,
                prefix_tokens = turn.cached_prefix_len,
                sim_time_ms   = sim_time,
            )

            # 2. Detect expiry (think time crossed TTL)
            if miss_type == CacheMissType.EXPIRY:
                self.stream.append(CheckpointEvent(
                    session_id  = session.session_id,
                    turn_id     = turn.turn_id,
                    event_id    = self.stream.next_id(),
                    kind        = EventKind.EXPIRY_BOUNDARY,
                    protocol    = "anthropic",
                    sim_time_ms = sim_time,
                ))

            # 3. Emit CACHE_DECISION (Anthropic-style)
            cache_read    = cached_tokens
            cache_written = turn.input_tokens - cached_tokens if cached_tokens < turn.input_tokens else 0
            self.ant_mapper.on_message_start(
                session_id           = session.session_id,
                turn_id              = turn.turn_id,
                sim_time_ms          = sim_time,
                input_tokens         = turn.input_tokens,
                cache_read_tokens    = cache_read,
                cache_written_tokens = cache_written,
                think_time_ms        = prev_think_time_ms,
            )

            turn_start_ms = sim_time

            # 4. Compute latency analytically
            latency = self.hw.predict_turn(
                input_tokens   = turn.input_tokens,
                output_tokens  = turn.output_tokens,
                cached_tokens  = cached_tokens,
                cache_location = cache_loc,
                batch_size     = 1,
            )

            # 5. Advance SimPy time by prefill latency → TTFT
            yield self.env.timeout(latency.prefill_ms)
            self.ant_mapper.on_first_content_delta(
                session_id    = session.session_id,
                turn_id       = turn.turn_id,
                sim_time_ms   = self.env.now,
                turn_start_ms = turn_start_ms,
            )

            # 6. Advance by decode latency → TURN_COMPLETE
            yield self.env.timeout(latency.decode_ms)
            self.ant_mapper.on_message_stop(
                session_id    = session.session_id,
                turn_id       = turn.turn_id,
                sim_time_ms   = self.env.now,
                output_tokens = turn.output_tokens,
            )

            # 7. Update KV cache with new tokens
            self.kv_mgr.update(
                session_id  = session.session_id,
                new_tokens  = turn.input_tokens,
                sim_time_ms = self.env.now,
            )

            # 8. Sample think time → wait before next turn
            if turn.turn_id < len(session.turns) - 1:
                think_s  = self.generator.sample_think_time_s()
                think_ms = think_s * 1000
                prev_think_time_ms = think_ms
                yield self.env.timeout(think_ms)

            # 9. Spawn sub-agents concurrently if any
            for sub in session.sub_sessions:
                if session.sub_session_spawn_turn.get(turn.turn_id):
                    self.env.process(self._run_session(sub))

        # SESSION_END
        self.stream.append(CheckpointEvent(
            session_id  = session.session_id,
            turn_id     = -1,
            event_id    = self.stream.next_id(),
            kind        = EventKind.SESSION_END,
            protocol    = "internal",
            sim_time_ms = self.env.now,
        ))
