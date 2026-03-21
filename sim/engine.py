from __future__ import annotations
import heapq
from collections import deque
from pathlib import Path

import numpy as np

from sim.config import SimConfig
from sim.events import Event, EventType, RequestState, validate_transition
from sim.cache import (
    CacheObject, PrefixTrie, TierStore, Tier, BlockLayout,
    kv_size_bytes, allocated_blocks, TIER_TO_LAYOUT,
)
from sim.service import ServiceModel
from sim.workload import WorkloadSynthesizer
from sim.oracle import PrefillOracle, DecodeOracle, transfer_time_us, is_cache_worthwhile
from sim.eviction import EvictionEngine
from sim.metrics import MetricsCollector


# Resolve benchmark table path relative to package
_BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "latency_tables"


class SessionState:
    """Tracks per-session context growth and turn count."""

    def __init__(self, session_id: str, profile_name: str, start_us: int, duration_us: int):
        self.session_id = session_id
        self.profile_name = profile_name
        self.start_us = start_us
        self.end_us = start_us + duration_us
        self.turn = 0
        self.context_tokens = 0
        self.token_hash_count: int = 0  # length proxy — avoids building large lists


class SimEngine:
    """
    Main simulation loop. sim_clock is uint64 microseconds.
    Never read wall-clock time inside this class.
    """

    def __init__(self, config: SimConfig):
        if config.enable_suffix_cache:
            raise NotImplementedError("Suffix cache is a v2 feature")
        if config.enable_l3b_object_store:
            raise NotImplementedError("L3b object store is a v2 feature")

        self.config = config
        self.sim_clock_us: int = 0
        self._event_seq: int = 0
        self._heap: list[tuple] = []

        rng = np.random.default_rng(config.seed)
        self.rng = rng
        self.workload = WorkloadSynthesizer(config, rng)
        self.service = ServiceModel(config.service)
        self.metrics = MetricsCollector()

        # Tier stores
        self._tier_stores: dict[Tier, TierStore] = {}
        for i, tier_cfg in enumerate(config.tiers):
            tier_enum = [Tier.L1, Tier.L2, Tier.L3A][i]
            self._tier_stores[tier_enum] = TierStore(
                tier_cfg.name, tier_cfg.capacity_bytes, tier_cfg.block_size_bytes
            )

        self.eviction = EvictionEngine(config, self._tier_stores)

        # Oracles
        table_path = _BENCHMARK_DIR / f"prefill_70b_a100.json"
        if not table_path.exists():
            raise FileNotFoundError(f"Benchmark table missing: {table_path}")
        self.prefill_oracle = PrefillOracle(str(table_path))
        self.decode_oracle = DecodeOracle()

        # Prefix tries
        self.shared_prefix_trie = PrefixTrie()
        self.session_tries: dict[str, PrefixTrie] = {}

        # Session tracking
        self.sessions: dict[str, SessionState] = {}
        self.request_states: dict[str, RequestState] = {}
        self._pending_prefills: deque[tuple] = deque()  # (session_id, request_id, payload)
        self.request_arrival_us: dict[str, int] = {}
        self._id_counter: int = 0

        # Profile lookup cache (avoid linear scan per request)
        self._profile_map: dict[str, object] = {p.name: p for p in config.profiles}

        # Dispatch table (built once)
        self._dispatch_table = {
            EventType.REQUEST_ARRIVAL: self._on_arrival,
            EventType.PREFILL_START: self._on_prefill_start,
            EventType.PREFILL_COMPLETE: self._on_prefill_complete,
            EventType.DECODE_START: self._on_decode_start,
            EventType.DECODE_COMPLETE: self._on_decode_complete,
            EventType.TTL_FIRE: self._on_ttl_fire,
            EventType.TIER_EVICTION: self._on_tier_eviction,
            EventType.SESSION_RESUME: self._on_session_resume,
            EventType.EPOCH_REPORT: self._on_epoch_report,
        }

    def schedule(self, event: Event) -> None:
        assert event.time_us >= self.sim_clock_us, (
            f"Cannot schedule event in the past: {event.time_us} < {self.sim_clock_us}"
        )
        event.seq = self._event_seq
        heapq.heappush(self._heap, (event.time_us, self._event_seq, event))
        self._event_seq += 1

    def run(self) -> MetricsCollector:
        end_us = int(self.config.sim_duration_s * 1_000_000)
        warmup_us = int(self.config.warmup_s * 1_000_000)

        self._seed_arrivals()

        # Schedule periodic epoch reports
        epoch_interval_us = int(self.config.epoch_report_interval_s * 1_000_000)
        t = epoch_interval_us
        while t <= end_us:
            self.schedule(Event(
                time_us=t, seq=0, event_type=EventType.EPOCH_REPORT,
                session_id="__system__",
            ))
            t += epoch_interval_us

        while self._heap:
            _, _, event = heapq.heappop(self._heap)
            if event.time_us > end_us:
                break
            self.sim_clock_us = event.time_us
            collecting = self.sim_clock_us >= warmup_us
            self._dispatch(event, collecting)

        self.metrics.effective_sim_us = end_us - warmup_us
        return self.metrics

    def _dispatch(self, event: Event, collecting: bool) -> None:
        self._dispatch_table[event.event_type](event, collecting)

    # ─── Arrival and session management ───

    def _seed_arrivals(self) -> None:
        """Schedule one new-session arrival per profile to kick off the stream."""
        for profile in self.config.profiles:
            mix_frac = self.config.profile_mix.get(profile.name, 0)
            if mix_frac <= 0:
                continue
            self._schedule_next_new_session(profile, 0.0)

    def _new_session(self, profile, start_us: int) -> str:
        self._id_counter += 1
        session_id = f"s{self._id_counter}"
        dur_s = self.workload.sample_session_duration(profile)
        dur_us = int(dur_s * 1_000_000)
        self.sessions[session_id] = SessionState(
            session_id, profile.name, start_us, dur_us,
        )
        self.session_tries[session_id] = PrefixTrie()

        # Initialize with shared system prefix if applicable
        if profile.shared_system_prefix_tokens > 0:
            sess = self.sessions[session_id]
            sess.context_tokens = profile.shared_system_prefix_tokens
            sess.token_hash_count = profile.shared_system_prefix_tokens
        return session_id

    def _on_arrival(self, event: Event, collecting: bool) -> None:
        session_id = event.session_id
        sess = self.sessions.get(session_id)
        profile_name = (event.payload or {}).get("profile", "")
        profile = self._get_profile(profile_name)
        is_new_session_event = (event.payload or {}).get("new_session", False)

        # If this is a new-session stream event, create the session and
        # immediately schedule the next new-session event for this profile.
        if is_new_session_event:
            session_id = self._new_session(profile, self.sim_clock_us)
            sess = self.sessions[session_id]
            event.session_id = session_id
            # Keep the new-session stream going
            self._schedule_next_new_session(profile, self.sim_clock_us / 1_000_000)

        if sess is None or self.sim_clock_us > sess.end_us:
            # Session expired or unknown — just drop this request
            return

        self._id_counter += 1
        request_id = f"r{self._id_counter}"
        self.request_states[request_id] = RequestState.QUEUED
        self.request_arrival_us[request_id] = self.sim_clock_us

        # Grow context
        if profile.context_growth_max_tokens > 0:
            growth = self.rng.integers(
                profile.context_growth_min_tokens,
                profile.context_growth_max_tokens + 1,
            )
            sess.context_tokens += int(growth)

        input_len = self.workload.sample_input_length(profile)
        output_len = self.workload.sample_output_length(profile)
        sess.turn += 1

        # Track token count (PrefixTrie only needs lengths, not actual hashes)
        sess.token_hash_count += input_len

        # Schedule PREFILL_START immediately
        self.schedule(Event(
            time_us=self.sim_clock_us, seq=0,
            event_type=EventType.PREFILL_START,
            session_id=session_id,
            request_id=request_id,
            payload={
                "profile": profile_name,
                "input_tokens": input_len,
                "output_tokens": output_len,
                "total_context": sess.context_tokens,
                "turn": sess.turn,
            },
        ))

        # Schedule next arrival within this session
        iat = self.workload.sample_iat(profile)
        next_us = self.sim_clock_us + int(iat * 1_000_000)
        if next_us <= sess.end_us:
            self.schedule(Event(
                time_us=next_us, seq=0,
                event_type=EventType.REQUEST_ARRIVAL,
                session_id=session_id,
                payload={"profile": profile_name},
            ))

    def _schedule_next_new_session(self, profile, current_time_s: float) -> None:
        """Schedule the next new-session arrival for this profile's stream."""
        t = self.workload.sample_next_arrival_time(current_time_s, profile)
        t_us = int(t * 1_000_000)
        end_us = int(self.config.sim_duration_s * 1_000_000)
        if t_us <= end_us:
            self.schedule(Event(
                time_us=t_us, seq=0,
                event_type=EventType.REQUEST_ARRIVAL,
                session_id="__pending__",
                payload={"profile": profile.name, "new_session": True},
            ))

    # ─── Prefill ───

    def _on_prefill_start(self, event: Event, collecting: bool) -> None:
        request_id = event.request_id
        payload = event.payload or {}
        session_id = event.session_id
        sess = self.sessions.get(session_id)
        profile = self._get_profile(payload.get("profile", ""))

        # FSM transition: QUEUED -> CACHE_LOOKUP
        self._transition(request_id, RequestState.CACHE_LOOKUP)

        total_context = payload.get("total_context", 0)
        input_tokens = payload.get("input_tokens", 0)

        # Determine prefix stability for this turn
        estimated_turns = max(1, int(
            (sess.end_us - sess.start_us) / max(1, int(profile.iat_mean_s * 1_000_000))
        )) if sess else 1
        turn = payload.get("turn", 1)
        stability = self.workload.prefix_stability(profile, turn, estimated_turns)
        cached_tokens = int(total_context * stability)
        uncached_tokens = max(1, total_context - cached_tokens + input_tokens)

        # Cache lookup — pick the deepest match across both tries
        hit_tier = None
        hit_key = None
        best_depth = 0
        if sess and sess.token_hash_count > 0:
            # Check shared prefix trie
            sp_key, sp_depth = self.shared_prefix_trie.lookup(sess.token_hash_count)
            if sp_key and sp_depth > 0:
                sp_obj = self._find_cache_object(sp_key)
                if sp_obj:
                    best_depth = sp_depth
                    hit_key = sp_key
                    hit_tier = sp_obj.tier

            # Check session trie — prefer if deeper match
            s_trie = self.session_tries.get(session_id)
            if s_trie:
                s_key, s_depth = s_trie.lookup(sess.token_hash_count)
                if s_key and s_depth > best_depth:
                    s_obj = self._find_cache_object(s_key)
                    if s_obj:
                        hit_key = s_key
                        hit_tier = s_obj.tier
                        best_depth = s_depth

        # Classify hit
        if hit_tier == Tier.L1:
            self._transition(request_id, RequestState.HIT_L1)
            ttft_component = "L1_hit"
        elif hit_tier == Tier.L2:
            self._transition(request_id, RequestState.HIT_L2)
            ttft_component = "L2_hit"
        elif hit_tier == Tier.L3A:
            self._transition(request_id, RequestState.HIT_L3A)
            ttft_component = "L3A_hit"
        else:
            self._transition(request_id, RequestState.MISS)
            ttft_component = "cold_miss"
            uncached_tokens = total_context + input_tokens
            cached_tokens = 0

        # Compute prefill latency
        kv_bytes = kv_size_bytes(cached_tokens, self.config.model) if cached_tokens > 0 else 0

        if hit_tier and hit_tier != Tier.L1 and kv_bytes > 0:
            tier_cfg = self.config.tiers[[Tier.L1, Tier.L2, Tier.L3A].index(hit_tier)]
            t_transfer = transfer_time_us(kv_bytes, tier_cfg)
            worthwhile = is_cache_worthwhile(
                kv_bytes, tier_cfg, uncached_tokens, self.prefill_oracle
            )

            # Savings classification
            if hit_tier == Tier.L2:
                if collecting:
                    self.metrics.savings_events["CACHE_HIT_L2_WORTHWHILE"] += 1
            elif hit_tier == Tier.L3A:
                if collecting:
                    cls = "CACHE_HIT_L3A_WORTHWHILE" if worthwhile else "CACHE_HIT_L3A_BREAK_EVEN"
                    self.metrics.savings_events[cls] += 1

            # Prefill time = transfer + remaining uncached prefill
            prefill_us = t_transfer + self.prefill_oracle.prefill_latency_us(
                max(1, uncached_tokens)
            )
        elif hit_tier == Tier.L1 and kv_bytes > 0:
            if collecting:
                self.metrics.savings_events["CACHE_HIT_L1"] += 1
            prefill_us = self.prefill_oracle.prefill_latency_us(max(1, uncached_tokens))
        else:
            if collecting:
                self.metrics.savings_events["COLD_MISS"] += 1
            prefill_us = self.prefill_oracle.prefill_latency_us(max(1, uncached_tokens))

        # Update cache object access time
        if hit_key:
            obj = self._find_cache_object(hit_key)
            if obj:
                obj.last_accessed_at_us = self.sim_clock_us

        # Recompute fraction
        total_tokens = cached_tokens + uncached_tokens
        if collecting and total_tokens > 0:
            self.metrics.recompute_fraction.append(uncached_tokens / total_tokens)

        # Try to admit to prefill pool
        self._transition(request_id, RequestState.PREFILLING)

        payload["uncached_tokens"] = uncached_tokens
        payload["cached_tokens"] = cached_tokens
        payload["prefill_us"] = prefill_us
        payload["ttft_component"] = ttft_component
        payload["hit_key"] = hit_key

        if self.service.prefill_slots_free > 0:
            self.service.prefill_slots_free -= 1
            self.schedule(Event(
                time_us=self.sim_clock_us + prefill_us, seq=0,
                event_type=EventType.PREFILL_COMPLETE,
                session_id=event.session_id,
                request_id=request_id,
                payload=payload,
            ))
        elif len(self._pending_prefills) < self.config.service.prefill_queue_max:
            # Queue the prefill — will be scheduled when a slot frees up
            self._pending_prefills.append((event.session_id, request_id, payload))
        # If queue is also full, request is dropped (backpressure)

    def _on_prefill_complete(self, event: Event, collecting: bool) -> None:
        request_id = event.request_id
        payload = event.payload or {}

        # FSM: PREFILLING -> DECODE_QUEUED
        self._transition(request_id, RequestState.DECODE_QUEUED)

        # Record TTFT
        if collecting and request_id in self.request_arrival_us:
            ttft = self.sim_clock_us - self.request_arrival_us[request_id]
            component = payload.get("ttft_component", "cold_miss")
            self.metrics.ttft_us[component].append(ttft)

        # Sharing tracking
        if collecting:
            cached = payload.get("cached_tokens", 0)
            uncached = payload.get("uncached_tokens", 0)
            self.metrics.total_tokens_served += cached + uncached
            profile = self._get_profile(payload.get("profile", ""))
            if profile.shared_system_prefix_tokens > 0 and cached > 0:
                shared_contribution = min(cached, profile.shared_system_prefix_tokens)
                self.metrics.tokens_served_from_shared_prefix += shared_contribution

        # Free the prefill slot and drain pending queue
        self.service.prefill_slots_free += 1
        self._drain_pending_prefills()

        # Try to move to decode
        decode_event = self.service.complete_prefill(event, self.sim_clock_us)
        if decode_event:
            decode_event.payload = payload
            self.schedule(decode_event)
        else:
            if collecting:
                self.metrics.prefill_slot_blocked_us += 1  # will accumulate in epoch

    # ─── Decode ───

    def _on_decode_start(self, event: Event, collecting: bool) -> None:
        request_id = event.request_id
        self._transition(request_id, RequestState.DECODING)

        payload = event.payload or {}
        output_tokens = payload.get("output_tokens", 1)
        active_seqs = self.service.active_decode_sequences

        decode_us = self.decode_oracle.decode_latency_us(output_tokens, active_seqs)

        self.schedule(Event(
            time_us=self.sim_clock_us + decode_us, seq=0,
            event_type=EventType.DECODE_COMPLETE,
            session_id=event.session_id,
            request_id=request_id,
            payload=payload,
        ))

    def _on_decode_complete(self, event: Event, collecting: bool) -> None:
        request_id = event.request_id
        payload = event.payload or {}

        # FSM: DECODING -> KV_WRITE
        self._transition(request_id, RequestState.KV_WRITE)

        # Write KV to L1 cache
        session_id = event.session_id
        total_ctx = payload.get("total_context", 0)
        output_tokens = payload.get("output_tokens", 0)
        total_tokens = total_ctx + output_tokens

        if total_tokens > 0:
            size = kv_size_bytes(total_tokens, self.config.model)
            cache_key = f"kv-{session_id}-{request_id}"
            sess = self.sessions.get(session_id)
            profile = self._get_profile(payload.get("profile", ""))

            placed_tier = self._place_kv_object(
                cache_key, session_id, profile, size, total_tokens, collecting,
            )

            # Update tries regardless of which tier the object landed in
            if placed_tier and sess and sess.token_hash_count > 0:
                trie = self.session_tries.get(session_id)
                if trie:
                    trie.insert(
                        total_tokens,
                        cache_key, self.sim_clock_us,
                    )

                # Shared prefix trie
                if profile.shared_system_prefix_tokens > 0:
                    sp_len = profile.shared_system_prefix_tokens
                    sp_key = f"sp-{profile.name}"

                    existing_obj = self._find_cache_object(sp_key)
                    if existing_obj:
                        existing_obj.ref_count += 1
                        existing_obj.last_accessed_at_us = self.sim_clock_us
                    else:
                        sp_size = kv_size_bytes(sp_len, self.config.model)
                        self._place_kv_object(
                            sp_key, "__shared__", profile,
                            sp_size, sp_len, collecting,
                            shared_prefix_id=profile.name,
                        )
                        if self._find_cache_object(sp_key):
                            self.shared_prefix_trie.insert(
                                sp_len, sp_key, self.sim_clock_us,
                            )

        # FSM: KV_WRITE -> COMPLETE
        self._transition(request_id, RequestState.COMPLETE)

        # Free decode slot
        next_decode = self.service.complete_decode(event, self.sim_clock_us)
        if next_decode:
            next_decode.payload = next_decode.payload or {}
            self.schedule(next_decode)

        # Cleanup
        self.request_states.pop(request_id, None)
        self.request_arrival_us.pop(request_id, None)

    # ─── TTL and eviction ───

    def _on_ttl_fire(self, event: Event, collecting: bool) -> None:
        payload = event.payload or {}
        cache_key = payload.get("cache_key", "")
        target = payload.get("target", "")

        if target == "l2_hibernate":
            # Check if the object is still in L1 and hasn't been accessed recently
            l1 = self._tier_stores[Tier.L1]
            obj = l1.get(cache_key)
            if obj is None:
                return  # Already evicted or moved

            ttl_us = int(self.config.ttl_l2_s * 1_000_000)
            if self.sim_clock_us - obj.last_accessed_at_us < ttl_us:
                # Was accessed since TTL was set; reschedule
                remaining = ttl_us - (self.sim_clock_us - obj.last_accessed_at_us)
                self.schedule(Event(
                    time_us=self.sim_clock_us + remaining, seq=0,
                    event_type=EventType.TTL_FIRE,
                    session_id=event.session_id,
                    payload=payload,
                ))
                return

            # Move L1 -> L2
            evicted = self.eviction.evict_l1_to_l2(0)  # just move this one
            # Actually, directly move this specific object
            l1_obj = l1.remove(cache_key)
            if l1_obj:
                l2 = self._tier_stores[Tier.L2]
                n_blocks = allocated_blocks(l1_obj.size_bytes, l2.block_size_bytes)
                l1_obj.tier = Tier.L2
                l1_obj.block_count = n_blocks
                l1_obj.block_layout = TIER_TO_LAYOUT[Tier.L2]

                if not l2.can_fit(l1_obj.size_bytes):
                    # Evict from L2 to L3a
                    expired = self.eviction.find_ttl_expired_l2(self.sim_clock_us)
                    for ek in expired:
                        self.eviction.hibernate_l2_to_l3a(ek)
                        if collecting:
                            self.metrics.l2_to_l3a_evictions += 1

                if l2.can_fit(l1_obj.size_bytes):
                    l2.insert(cache_key, l1_obj)
                    if collecting:
                        self.metrics.l1_to_l2_ttl_migrations += 1

                    # If L2 TTL is 0, immediately hibernate to L3A
                    if self.config.ttl_l2_s <= 0:
                        self.eviction.hibernate_l2_to_l3a(cache_key)
                        if collecting:
                            self.metrics.l2_to_l3a_evictions += 1
                        return

                    # Schedule L3a hibernation TTL
                    l3a_ttl_us = int(self.config.ttl_l3a_s * 1_000_000)
                    self.schedule(Event(
                        time_us=self.sim_clock_us + l3a_ttl_us, seq=0,
                        event_type=EventType.TTL_FIRE,
                        session_id=event.session_id,
                        payload={"cache_key": cache_key, "target": "l3a_hibernate"},
                    ))

        elif target == "l3a_hibernate":
            l2 = self._tier_stores[Tier.L2]
            obj = l2.get(cache_key)
            if obj is None:
                return

            ttl_us = int(self.config.ttl_l3a_s * 1_000_000)
            if self.sim_clock_us - obj.last_accessed_at_us < ttl_us:
                remaining = ttl_us - (self.sim_clock_us - obj.last_accessed_at_us)
                self.schedule(Event(
                    time_us=self.sim_clock_us + remaining, seq=0,
                    event_type=EventType.TTL_FIRE,
                    session_id=event.session_id,
                    payload=payload,
                ))
                return

            success = self.eviction.hibernate_l2_to_l3a(cache_key)
            if collecting and success:
                self.metrics.l2_to_l3a_evictions += 1

    def _on_tier_eviction(self, event: Event, collecting: bool) -> None:
        """Handle explicit tier eviction events (L3a cleanup)."""
        if self.eviction.needs_l3a_cleanup():
            evicted = self.eviction.cleanup_l3a(
                self._tier_stores[Tier.L3A].block_size_bytes
            )
            if collecting:
                self.metrics.session_cold_evictions += len(evicted)

    def _on_session_resume(self, event: Event, collecting: bool) -> None:
        """Handle session resume from cache."""
        # Session resume is handled by the normal arrival + cache lookup path
        pass

    # ─── Epoch reporting ───

    def _on_epoch_report(self, event: Event, collecting: bool) -> None:
        if not collecting:
            return

        # Record tier occupancy
        for tier_enum, store in self._tier_stores.items():
            self.metrics.tier_occupancy_pct[store.name].append(
                store.occupancy_pct * 100.0
            )

        # Record queue depths
        self.metrics.prefill_queue_depth.append(len(self._pending_prefills))
        self.metrics.decode_queue_depth.append(len(self.service.decode_queue))

        # Record memory pollution
        for tier_enum, store in self._tier_stores.items():
            self.metrics.memory_pollution_bytes[store.name] = store.fragmentation_bytes()

        # Accumulate prefill blocked time
        self.metrics.prefill_slot_blocked_us += self.service.get_blocked_us(self.sim_clock_us)

        # Run TTL checks for L2 -> L3a hibernation
        expired_l2 = self.eviction.find_ttl_expired_l2(self.sim_clock_us)
        for key in expired_l2:
            self.eviction.hibernate_l2_to_l3a(key)
            self.metrics.l2_to_l3a_evictions += 1

        # Run TTL checks for L3a — only evict objects whose session has ended
        expired_l3a = self.eviction.find_ttl_expired_l3a(self.sim_clock_us)
        for key in expired_l3a:
            obj = self._tier_stores[Tier.L3A].get(key)
            if obj and obj.session_id != "__shared__":
                sess = self.sessions.get(obj.session_id)
                if sess and self.sim_clock_us <= sess.end_us:
                    continue  # Session still alive — keep the object
            self._tier_stores[Tier.L3A].remove(key)
            self.metrics.session_cold_evictions += 1

        # L1 pressure check
        if self.eviction.needs_l1_eviction():
            l1 = self._tier_stores[Tier.L1]
            target = int(l1.capacity_bytes * self.config.eviction_hbm_threshold * 0.9)
            to_free = l1.used_bytes - target
            if to_free > 0:
                evicted = self.eviction.evict_l1_to_l2(to_free)
                self.metrics.l1_to_l2_evictions += len(evicted)

    # ─── Helpers ───

    def _drain_pending_prefills(self) -> None:
        """Schedule pending prefills when prefill slots become available."""
        while self._pending_prefills and self.service.prefill_slots_free > 0:
            session_id, request_id, payload = self._pending_prefills.popleft()
            self.service.prefill_slots_free -= 1
            prefill_us = payload.get("prefill_us", 0)
            self.schedule(Event(
                time_us=self.sim_clock_us + prefill_us, seq=0,
                event_type=EventType.PREFILL_COMPLETE,
                session_id=session_id,
                request_id=request_id,
                payload=payload,
            ))

    def _place_kv_object(
        self,
        cache_key: str,
        session_id: str,
        profile,
        size: int,
        total_tokens: int,
        collecting: bool,
        shared_prefix_id: str | None = None,
    ) -> Tier | None:
        """
        Try to place a KV object into L1; if it doesn't fit, fall through
        to L2 (then L3A). Returns the tier it was placed in, or None.
        """
        if shared_prefix_id is None and profile.shared_system_prefix_tokens > 0:
            sp_id = profile.name
        else:
            sp_id = shared_prefix_id

        # Try L1
        l1 = self._tier_stores[Tier.L1]
        if not l1.can_fit(size):
            evicted = self.eviction.evict_l1_to_l2(
                allocated_blocks(size, l1.block_size_bytes) * l1.block_size_bytes
            )
            if collecting:
                self.metrics.l1_to_l2_evictions += len(evicted)

        if l1.can_fit(size):
            n_blocks = allocated_blocks(size, l1.block_size_bytes)
            obj = CacheObject(
                session_id=session_id,
                shared_prefix_id=sp_id,
                token_range=(0, total_tokens - 1),
                model_id=self.config.model.model_id,
                tier=Tier.L1,
                size_bytes=size,
                block_count=n_blocks,
                created_at_us=self.sim_clock_us,
                last_accessed_at_us=self.sim_clock_us,
                ref_count=1,
                block_layout=TIER_TO_LAYOUT[Tier.L1],
            )
            l1.insert(cache_key, obj)
            # Schedule TTL
            ttl_us = int(self.config.ttl_l2_s * 1_000_000)
            if ttl_us >= 0:
                self.schedule(Event(
                    time_us=self.sim_clock_us + ttl_us, seq=0,
                    event_type=EventType.TTL_FIRE,
                    session_id=session_id,
                    payload={"cache_key": cache_key, "target": "l2_hibernate"},
                ))
            return Tier.L1

        # L1 too small — try L2 directly
        l2 = self._tier_stores[Tier.L2]
        if not l2.can_fit(size):
            expired = self.eviction.find_ttl_expired_l2(self.sim_clock_us)
            for ek in expired:
                self.eviction.hibernate_l2_to_l3a(ek)
                if collecting:
                    self.metrics.l2_to_l3a_evictions += 1

        if l2.can_fit(size):
            n_blocks = allocated_blocks(size, l2.block_size_bytes)
            obj = CacheObject(
                session_id=session_id,
                shared_prefix_id=sp_id,
                token_range=(0, total_tokens - 1),
                model_id=self.config.model.model_id,
                tier=Tier.L2,
                size_bytes=size,
                block_count=n_blocks,
                created_at_us=self.sim_clock_us,
                last_accessed_at_us=self.sim_clock_us,
                ref_count=1,
                block_layout=TIER_TO_LAYOUT[Tier.L2],
            )
            l2.insert(cache_key, obj)
            # Schedule L3a hibernation TTL
            ttl_us = int(self.config.ttl_l3a_s * 1_000_000)
            if ttl_us >= 0:
                self.schedule(Event(
                    time_us=self.sim_clock_us + ttl_us, seq=0,
                    event_type=EventType.TTL_FIRE,
                    session_id=session_id,
                    payload={"cache_key": cache_key, "target": "l3a_hibernate"},
                ))
            return Tier.L2

        # L2 too small — try L3A
        l3a = self._tier_stores[Tier.L3A]
        if not l3a.can_fit(size):
            evicted = self.eviction.cleanup_l3a(
                allocated_blocks(size, l3a.block_size_bytes) * l3a.block_size_bytes
            )
            if collecting:
                self.metrics.session_cold_evictions += len(evicted)

        if l3a.can_fit(size):
            n_blocks = allocated_blocks(size, l3a.block_size_bytes)
            obj = CacheObject(
                session_id=session_id,
                shared_prefix_id=sp_id,
                token_range=(0, total_tokens - 1),
                model_id=self.config.model.model_id,
                tier=Tier.L3A,
                size_bytes=size,
                block_count=n_blocks,
                created_at_us=self.sim_clock_us,
                last_accessed_at_us=self.sim_clock_us,
                ref_count=1,
                block_layout=TIER_TO_LAYOUT[Tier.L3A],
                is_hibernated=True,
            )
            l3a.insert(cache_key, obj)
            return Tier.L3A

        return None  # Couldn't place anywhere

    def _transition(self, request_id: str, next_state: RequestState) -> None:
        current = self.request_states.get(request_id)
        if current is not None:
            validate_transition(current, next_state)
        self.request_states[request_id] = next_state

    def _get_profile(self, name: str):
        return self._profile_map.get(name, self.config.profiles[0])

    def _find_cache_object(self, key: str) -> CacheObject | None:
        for store in self._tier_stores.values():
            obj = store.get(key)
            if obj:
                return obj
        return None
