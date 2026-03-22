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
    cached_tokens_at_block_boundary,
)
from sim.service import ServiceModel
from sim.workload import WorkloadSynthesizer
from sim.oracle import PrefillOracle, DecodeOracle, transfer_time_us, is_cache_worthwhile
from sim.eviction import EvictionEngine
from sim.metrics import MetricsCollector
from sim.node import PrefillNode
from sim.dispatch import PushDispatcher, PullDispatcher


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

        n_nodes = config.service.n_prefill_nodes
        gpus_per_worker = config.service.n_gpus_per_worker
        self._l3a_shared = config.service.l3a_shared
        self._gpus_per_worker = gpus_per_worker

        # Compute number of workers
        if n_nodes == 1:
            n_workers = 1
        else:
            assert n_nodes % gpus_per_worker == 0, (
                f"n_prefill_nodes ({n_nodes}) must be divisible by n_gpus_per_worker ({gpus_per_worker})"
            )
            n_workers = n_nodes // gpus_per_worker
        self._n_workers = n_workers

        # L3A config value = per-worker SSD capacity
        # Global mode: pool all workers' SSDs → total = per_worker × n_workers
        # Local mode: each worker uses its own SSD → per_worker capacity
        l3a_cfg = config.tiers[2]
        if self._l3a_shared:
            global_cap = l3a_cfg.capacity_bytes * n_workers
            self._shared_l3a = TierStore(l3a_cfg.name, global_cap, l3a_cfg.block_size_bytes)
        else:
            self._shared_l3a = None  # no global L3A in local mode

        # Create per-worker shared L2 (DRAM) and local L3A (SSD) stores,
        # then per-GPU L1 (HBM) stores
        l1_cfg = config.tiers[0]
        l2_cfg = config.tiers[1]

        self._worker_l2_stores: list[TierStore] = []
        self._worker_l3a_stores: list[TierStore | None] = []
        for w in range(n_workers):
            l2 = TierStore(l2_cfg.name, l2_cfg.capacity_bytes, l2_cfg.block_size_bytes)
            self._worker_l2_stores.append(l2)
            if not self._l3a_shared:
                local_l3a = TierStore(l3a_cfg.name, l3a_cfg.capacity_bytes, l3a_cfg.block_size_bytes)
                self._worker_l3a_stores.append(local_l3a)
            else:
                self._worker_l3a_stores.append(None)

        self.nodes: list[PrefillNode] = []
        for i in range(n_nodes):
            worker_id = i // gpus_per_worker if n_nodes > 1 else 0
            l1 = TierStore(l1_cfg.name, l1_cfg.capacity_bytes, l1_cfg.block_size_bytes)
            worker_l2 = self._worker_l2_stores[worker_id]
            worker_l3a = self._worker_l3a_stores[worker_id]
            node = PrefillNode(
                node_id=i,
                worker_id=worker_id,
                l1_store=l1,
                l2_store=worker_l2,  # shared with other GPUs on same worker
                prefill_slots=config.service.n_prefill_slots,
                prefill_queue_max=config.service.prefill_queue_max,
                l3a_store=worker_l3a,  # shared with other GPUs on same worker
            )
            self.nodes.append(node)

        # Backward-compat: _tier_stores aliases node[0] L1/L2 + L3A
        self._tier_stores: dict[Tier, TierStore] = {
            Tier.L1: self.nodes[0].l1_store,
            Tier.L2: self.nodes[0].l2_store,
            Tier.L3A: self._shared_l3a if self._l3a_shared else self.nodes[0].l3a_store,
        }

        # Per-node eviction engines
        self._node_eviction: list[EvictionEngine] = []
        for node in self.nodes:
            l3a_for_node = self._shared_l3a if self._l3a_shared else node.l3a_store
            stores = {
                Tier.L1: node.l1_store,
                Tier.L2: node.l2_store,
                Tier.L3A: l3a_for_node,
            }
            self._node_eviction.append(EvictionEngine(config, stores))

        # Backward-compat alias
        self.eviction = self._node_eviction[0]

        # Dispatcher
        if config.service.dispatch_algorithm == "pull":
            self.dispatcher = PullDispatcher(self.nodes)
        else:
            self.dispatcher = PushDispatcher(self.nodes)

        # Backward-compat: single pending_prefills deque for N=1
        self._pending_prefills = self.nodes[0].pending_prefills

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
        self.request_arrival_us: dict[str, int] = {}
        self._request_queued_at_us: dict[str, int] = {}  # request_id -> time entered pending queue
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
            EventType.NODE_PULL_CHECK: self._on_node_pull_check,
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

    # ─── Node helpers ───

    def _get_node(self, node_id: int) -> PrefillNode:
        return self.nodes[node_id]

    def _get_node_eviction(self, node_id: int) -> EvictionEngine:
        return self._node_eviction[node_id]

    def _find_cache_object_with_node(self, key: str, from_node_id: int | None = None) -> tuple[CacheObject | None, int | None]:
        """Search all node L1s, all node L2s, then L3A.
        Returns (obj, node_id) for L1/L2 hits; (obj, None) for shared L3A;
        (obj, node_id) for local L3A hits.
        For local L3A, only searches the requesting node's worker SSD."""
        for node in self.nodes:
            obj = node.l1_store.get(key)
            if obj:
                return obj, node.node_id
            obj = node.l2_store.get(key)
            if obj:
                return obj, node.node_id
        if self._l3a_shared:
            obj = self._shared_l3a.get(key)
            if obj:
                return obj, None
        else:
            # Local L3A: only search the requesting node's worker SSD
            if from_node_id is not None:
                worker_id = self.nodes[from_node_id].worker_id
                # All nodes on same worker share the same l3a_store reference
                l3a = self.nodes[from_node_id].l3a_store
                if l3a:
                    obj = l3a.get(key)
                    if obj:
                        return obj, from_node_id
            else:
                # Fallback: search all workers' L3A (for global lookups like trie)
                checked_workers = set()
                for node in self.nodes:
                    if node.worker_id in checked_workers:
                        continue
                    checked_workers.add(node.worker_id)
                    if node.l3a_store:
                        obj = node.l3a_store.get(key)
                        if obj:
                            return obj, node.node_id
        return None, None

    def _find_cache_object(self, key: str) -> CacheObject | None:
        obj, _ = self._find_cache_object_with_node(key)
        return obj

    def _get_l3a_for_node(self, node_id: int) -> TierStore:
        """Return the L3A store for a given node (shared or local)."""
        if self._l3a_shared:
            return self._shared_l3a
        return self.nodes[node_id].l3a_store

    def _same_worker(self, node_id_a: int, node_id_b: int) -> bool:
        """Check if two GPU nodes are on the same worker (share L2/L3A)."""
        return self.nodes[node_id_a].worker_id == self.nodes[node_id_b].worker_id

    def _cross_node_transfer_us(self, size_bytes: int, from_node_id: int, to_node_id: int) -> int:
        """Transfer latency between two GPU nodes.
        Intra-worker (same host): small NVLink penalty for L1-to-L1, zero for shared L2/L3A.
        Inter-worker (different host): full network penalty."""
        if self._same_worker(from_node_id, to_node_id):
            # Intra-worker: NVLink between GPUs — use inter_node config but fast
            svc = self.config.service
            return svc.inter_node_latency_us  # just base latency, no bandwidth penalty
        svc = self.config.service
        return svc.inter_node_latency_us + int(
            (size_bytes / svc.inter_node_bandwidth_bytes_per_s) * 1_000_000
        )

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
        cached_tokens_raw = int(total_context * stability)
        # Round to block boundary — only full token blocks count as cached
        block_size_tok = self.config.cache.block_size_tokens
        cached_tokens = cached_tokens_at_block_boundary(cached_tokens_raw, block_size_tok)
        uncached_tokens = max(1, total_context - cached_tokens + input_tokens)

        # ─── Dispatch: pick a node ───
        n_nodes = self.config.service.n_prefill_nodes
        is_pull = isinstance(self.dispatcher, PullDispatcher)

        if is_pull:
            # Pull mode: enqueue to global queue, then trigger pulls
            # But first do cache lookup to populate payload
            pass  # node assignment happens after cache lookup below
        else:
            # Push mode: dispatch to a node
            node = self.dispatcher.dispatch(session_id, request_id, payload, self.sim_clock_us)
            payload["node_id"] = node.node_id

        # Cache lookup — pick the deepest match across both tries
        # For local L3A, only search the assigned node's worker SSD
        lookup_node_id = payload.get("node_id", None)  # set by push dispatch, None for pull
        hit_tier = None
        hit_key = None
        hit_node_id = None
        best_depth = 0
        if sess and sess.token_hash_count > 0:
            # Check shared prefix trie
            sp_key, sp_depth = self.shared_prefix_trie.lookup(sess.token_hash_count)
            if sp_key and sp_depth > 0:
                sp_obj, sp_nid = self._find_cache_object_with_node(sp_key, from_node_id=lookup_node_id)
                if sp_obj:
                    best_depth = sp_depth
                    hit_key = sp_key
                    hit_tier = sp_obj.tier
                    hit_node_id = sp_nid

            # Check session trie — prefer if deeper match
            s_trie = self.session_tries.get(session_id)
            if s_trie:
                s_key, s_depth = s_trie.lookup(sess.token_hash_count)
                if s_key and s_depth > best_depth:
                    s_obj, s_nid = self._find_cache_object_with_node(s_key, from_node_id=lookup_node_id)
                    if s_obj:
                        hit_key = s_key
                        hit_tier = s_obj.tier
                        hit_node_id = s_nid
                        best_depth = s_depth

        # For push mode, check if hit is on assigned node or remote
        assigned_node_id = payload.get("node_id", 0)

        # Track affinity dispatch
        if collecting and n_nodes > 1:
            if hit_node_id is not None and hit_node_id == assigned_node_id:
                self.metrics.affinity_dispatches += 1
            else:
                self.metrics.non_affinity_dispatches += 1

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

        # Cross-node transfer penalty
        cross_node_penalty_us = 0
        if (
            hit_tier in (Tier.L1, Tier.L2)
            and hit_node_id is not None
            and hit_node_id != assigned_node_id
            and n_nodes > 1
            and kv_bytes > 0
        ):
            if hit_tier == Tier.L2 and self._same_worker(hit_node_id, assigned_node_id):
                # L2 hit on same worker — no penalty (shared DRAM)
                pass
            else:
                cross_node_penalty_us = self._cross_node_transfer_us(
                    kv_bytes, hit_node_id, assigned_node_id
                )
                if collecting:
                    self.metrics.cross_node_transfers += 1

        if hit_tier and hit_tier != Tier.L1 and kv_bytes > 0:
            tier_cfg = self.config.tiers[[Tier.L1, Tier.L2, Tier.L3A].index(hit_tier)]
            t_transfer = transfer_time_us(kv_bytes, tier_cfg)
            # Add remote latency for global L3A access (only when multiple workers exist)
            if hit_tier == Tier.L3A and self._l3a_shared and self._n_workers > 1:
                t_transfer += self.config.service.l3a_remote_latency_us
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

            # Prefill time = transfer + remaining uncached prefill + cross-node penalty
            prefill_us = t_transfer + self.prefill_oracle.prefill_latency_us(
                max(1, uncached_tokens)
            ) + cross_node_penalty_us
        elif hit_tier == Tier.L1 and kv_bytes > 0:
            if collecting:
                self.metrics.savings_events["CACHE_HIT_L1"] += 1
            prefill_us = self.prefill_oracle.prefill_latency_us(
                max(1, uncached_tokens)
            ) + cross_node_penalty_us
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

        # FSM: -> PREFILLING
        self._transition(request_id, RequestState.PREFILLING)

        payload["uncached_tokens"] = uncached_tokens
        payload["cached_tokens"] = cached_tokens
        payload["prefill_us"] = prefill_us
        payload["ttft_component"] = ttft_component
        payload["hit_key"] = hit_key

        if is_pull:
            # Pull mode: assign to node via pull, or enqueue to global queue
            assigned = False
            for node in self.nodes:
                if node.prefill_slots_free > 0:
                    payload["node_id"] = node.node_id
                    node.prefill_slots_free -= 1
                    completion_us = self.sim_clock_us + prefill_us
                    node.add_completion(completion_us)
                    if collecting:
                        self.metrics.per_node_prefill_count[node.node_id] += 1
                        self.metrics.queue_wait_us.append(0)  # immediate slot
                    self.schedule(Event(
                        time_us=completion_us, seq=0,
                        event_type=EventType.PREFILL_COMPLETE,
                        session_id=event.session_id,
                        request_id=request_id,
                        payload=payload,
                    ))
                    assigned = True
                    break
            if not assigned:
                self._request_queued_at_us[request_id] = self.sim_clock_us
                self.dispatcher.enqueue(
                    event.session_id, request_id, payload, self.sim_clock_us,
                )
        else:
            # Push mode: use assigned node
            node = self._get_node(assigned_node_id)
            if node.prefill_slots_free > 0:
                node.prefill_slots_free -= 1
                completion_us = self.sim_clock_us + prefill_us
                node.add_completion(completion_us)
                if collecting:
                    self.metrics.per_node_prefill_count[node.node_id] += 1
                    self.metrics.queue_wait_us.append(0)  # immediate slot
                self.schedule(Event(
                    time_us=completion_us, seq=0,
                    event_type=EventType.PREFILL_COMPLETE,
                    session_id=event.session_id,
                    request_id=request_id,
                    payload=payload,
                ))
            elif len(node.pending_prefills) < node.prefill_queue_max:
                self._request_queued_at_us[request_id] = self.sim_clock_us
                node.pending_prefills.append((event.session_id, request_id, payload))
            # If queue is also full, request is dropped (backpressure)

    def _on_prefill_complete(self, event: Event, collecting: bool) -> None:
        request_id = event.request_id
        payload = event.payload or {}

        # FSM: PREFILLING -> DECODE_QUEUED
        self._transition(request_id, RequestState.DECODE_QUEUED)

        # Record TTFT and prefill duration
        if collecting and request_id in self.request_arrival_us:
            ttft = self.sim_clock_us - self.request_arrival_us[request_id]
            component = payload.get("ttft_component", "cold_miss")
            self.metrics.ttft_us[component].append(ttft)
        prefill_us = payload.get("prefill_us", 0)
        if collecting and prefill_us > 0:
            self.metrics.prefill_duration_us.append(prefill_us)

        # Sharing tracking
        if collecting:
            cached = payload.get("cached_tokens", 0)
            uncached = payload.get("uncached_tokens", 0)
            self.metrics.total_tokens_served += cached + uncached
            profile = self._get_profile(payload.get("profile", ""))
            if profile.shared_system_prefix_tokens > 0 and cached > 0:
                shared_contribution = min(cached, profile.shared_system_prefix_tokens)
                self.metrics.tokens_served_from_shared_prefix += shared_contribution

        # Free the prefill slot on the specific node and drain
        node_id = payload.get("node_id", 0)
        node = self._get_node(node_id)
        node.prefill_slots_free += 1

        # Remove completion time
        completion_us = event.time_us
        node.remove_completion(completion_us)

        # Drain pending
        if isinstance(self.dispatcher, PullDispatcher):
            self._pull_drain_node(node, collecting)
        else:
            self._push_drain_node(node, collecting)

        # Try to move to decode
        decode_event = self.service.complete_prefill(event, self.sim_clock_us)
        if decode_event:
            decode_event.payload = payload
            self.schedule(decode_event)
        else:
            if collecting:
                self.metrics.prefill_slot_blocked_us += 1

    def _push_drain_node(self, node: PrefillNode, collecting: bool) -> None:
        """Drain pending prefills from a node's local queue (push mode)."""
        while node.pending_prefills and node.prefill_slots_free > 0:
            session_id, request_id, payload = node.pending_prefills.popleft()
            node.prefill_slots_free -= 1
            prefill_us = payload.get("prefill_us", 0)
            completion_us = self.sim_clock_us + prefill_us
            node.add_completion(completion_us)
            if collecting:
                self.metrics.per_node_prefill_count[node.node_id] += 1
                queued_at = self._request_queued_at_us.pop(request_id, None)
                if queued_at is not None:
                    self.metrics.queue_wait_us.append(self.sim_clock_us - queued_at)
            self.schedule(Event(
                time_us=completion_us, seq=0,
                event_type=EventType.PREFILL_COMPLETE,
                session_id=session_id,
                request_id=request_id,
                payload=payload,
            ))

    def _pull_drain_node(self, node: PrefillNode, collecting: bool) -> None:
        """Pull jobs from global queue for this node (pull mode)."""
        while node.prefill_slots_free > 0:
            job = self.dispatcher.pull(node, self.sim_clock_us)
            if job is None:
                break
            session_id, request_id, payload = job
            payload["node_id"] = node.node_id
            node.prefill_slots_free -= 1
            prefill_us = payload.get("prefill_us", 0)
            completion_us = self.sim_clock_us + prefill_us
            node.add_completion(completion_us)
            if collecting:
                self.metrics.per_node_prefill_count[node.node_id] += 1
                queued_at = self._request_queued_at_us.pop(request_id, None)
                if queued_at is not None:
                    self.metrics.queue_wait_us.append(self.sim_clock_us - queued_at)
            self.schedule(Event(
                time_us=completion_us, seq=0,
                event_type=EventType.PREFILL_COMPLETE,
                session_id=session_id,
                request_id=request_id,
                payload=payload,
            ))

    def _on_node_pull_check(self, event: Event, collecting: bool) -> None:
        """Handle pull check event for a specific node."""
        node_id = (event.payload or {}).get("node_id", 0)
        node = self._get_node(node_id)
        self._pull_drain_node(node, collecting)

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
        node_id = payload.get("node_id", 0)

        if total_tokens > 0:
            size = kv_size_bytes(total_tokens, self.config.model)
            cache_key = f"kv-{session_id}-{request_id}"
            sess = self.sessions.get(session_id)
            profile = self._get_profile(payload.get("profile", ""))

            placed_tier = self._place_kv_object(
                cache_key, session_id, profile, size, total_tokens, collecting,
                node_id=node_id,
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
                            node_id=node_id,
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
        node_id = payload.get("node_id", 0)
        node = self._get_node(node_id)
        eviction = self._get_node_eviction(node_id)

        if target == "l2_hibernate":
            # Check if the object is still in the node's L1
            l1 = node.l1_store
            obj = l1.get(cache_key)
            if obj is None:
                return  # Already evicted or moved

            ttl_us = int(self.config.ttl_l2_s * 1_000_000)
            if self.sim_clock_us - obj.last_accessed_at_us < ttl_us:
                remaining = ttl_us - (self.sim_clock_us - obj.last_accessed_at_us)
                self.schedule(Event(
                    time_us=self.sim_clock_us + remaining, seq=0,
                    event_type=EventType.TTL_FIRE,
                    session_id=event.session_id,
                    payload=payload,
                ))
                return

            # Move L1 -> L2 (node-local)
            l1_obj = l1.remove(cache_key)
            if l1_obj:
                l2 = node.l2_store
                n_blocks = allocated_blocks(l1_obj.size_bytes, l2.block_size_bytes)
                l1_obj.tier = Tier.L2
                l1_obj.block_count = n_blocks
                l1_obj.block_layout = TIER_TO_LAYOUT[Tier.L2]

                if not l2.can_fit(l1_obj.size_bytes):
                    expired = eviction.find_ttl_expired_l2(self.sim_clock_us)
                    for ek in expired:
                        eviction.hibernate_l2_to_l3a(ek)
                        if collecting:
                            self.metrics.l2_to_l3a_evictions += 1

                if l2.can_fit(l1_obj.size_bytes):
                    l2.insert(cache_key, l1_obj)
                    if collecting:
                        self.metrics.l1_to_l2_ttl_migrations += 1

                    # If L2 TTL is 0, immediately hibernate to L3A
                    if self.config.ttl_l2_s <= 0:
                        eviction.hibernate_l2_to_l3a(cache_key)
                        if collecting:
                            self.metrics.l2_to_l3a_evictions += 1
                        return

                    # Schedule L3a hibernation TTL
                    l3a_ttl_us = int(self.config.ttl_l3a_s * 1_000_000)
                    self.schedule(Event(
                        time_us=self.sim_clock_us + l3a_ttl_us, seq=0,
                        event_type=EventType.TTL_FIRE,
                        session_id=event.session_id,
                        payload={"cache_key": cache_key, "target": "l3a_hibernate", "node_id": node_id},
                    ))

        elif target == "l3a_hibernate":
            l2 = node.l2_store
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

            success = eviction.hibernate_l2_to_l3a(cache_key)
            if collecting and success:
                self.metrics.l2_to_l3a_evictions += 1

    def _on_tier_eviction(self, event: Event, collecting: bool) -> None:
        """Handle explicit tier eviction events (L3a cleanup)."""
        for node_id, eviction in enumerate(self._node_eviction):
            if eviction.needs_l3a_cleanup():
                l3a = self._get_l3a_for_node(node_id)
                evicted = eviction.cleanup_l3a(l3a.block_size_bytes)
                if collecting:
                    self.metrics.session_cold_evictions += len(evicted)

    def _on_session_resume(self, event: Event, collecting: bool) -> None:
        """Handle session resume from cache."""
        pass

    # ─── Epoch reporting ───

    def _on_epoch_report(self, event: Event, collecting: bool) -> None:
        if not collecting:
            return

        # Record per-node and aggregate tier occupancy
        # L1 is per-GPU, L2 is per-worker (shared by GPUs on same host)
        agg_l1_used = 0
        agg_l1_cap = 0
        for node in self.nodes:
            l1_occ = node.l1_store.occupancy_pct * 100.0
            l2_occ = node.l2_store.occupancy_pct * 100.0
            self.metrics.per_node_l1_occupancy_pct[node.node_id].append(l1_occ)
            self.metrics.per_node_l2_occupancy_pct[node.node_id].append(l2_occ)
            self.metrics.per_node_queue_depth[node.node_id].append(len(node.pending_prefills))
            agg_l1_used += node.l1_store.used_bytes
            agg_l1_cap += node.l1_store.capacity_bytes

        # Aggregate L2 by worker (avoid double-counting shared L2 stores)
        agg_l2_used = sum(s.used_bytes for s in self._worker_l2_stores)
        agg_l2_cap = sum(s.capacity_bytes for s in self._worker_l2_stores)

        self.metrics.tier_occupancy_pct["L1"].append(
            (agg_l1_used / agg_l1_cap * 100.0) if agg_l1_cap > 0 else 0.0
        )
        self.metrics.tier_occupancy_pct["L2"].append(
            (agg_l2_used / agg_l2_cap * 100.0) if agg_l2_cap > 0 else 0.0
        )
        if self._l3a_shared:
            self.metrics.tier_occupancy_pct["L3A"].append(
                self._shared_l3a.occupancy_pct * 100.0
            )
        else:
            # Aggregate local L3A by worker (avoid double-counting)
            agg_l3a_used = sum(s.used_bytes for s in self._worker_l3a_stores if s)
            agg_l3a_cap = sum(s.capacity_bytes for s in self._worker_l3a_stores if s)
            self.metrics.tier_occupancy_pct["L3A"].append(
                (agg_l3a_used / agg_l3a_cap * 100.0) if agg_l3a_cap > 0 else 0.0
            )

        # Record queue depths (aggregate)
        total_pending = sum(len(n.pending_prefills) for n in self.nodes)
        if isinstance(self.dispatcher, PullDispatcher):
            total_pending += len(self.dispatcher.global_queue)
        self.metrics.prefill_queue_depth.append(total_pending)
        self.metrics.decode_queue_depth.append(len(self.service.decode_queue))

        # Record memory pollution (aggregate by worker for L2/L3A, by node for L1)
        agg_l1_frag = sum(n.l1_store.fragmentation_bytes() for n in self.nodes)
        agg_l2_frag = sum(s.fragmentation_bytes() for s in self._worker_l2_stores)
        self.metrics.memory_pollution_bytes["L1"] = agg_l1_frag
        self.metrics.memory_pollution_bytes["L2"] = agg_l2_frag
        if self._l3a_shared:
            self.metrics.memory_pollution_bytes["L3A"] = self._shared_l3a.fragmentation_bytes()
        else:
            self.metrics.memory_pollution_bytes["L3A"] = sum(
                s.fragmentation_bytes() for s in self._worker_l3a_stores if s
            )

        # Accumulate prefill blocked time
        self.metrics.prefill_slot_blocked_us += self.service.get_blocked_us(self.sim_clock_us)

        # Slot utilization: fraction of total prefill slots that are busy
        total_slots = sum(n.prefill_slots_total for n in self.nodes)
        busy_slots = sum(n.prefill_slots_total - n.prefill_slots_free for n in self.nodes)
        self.metrics.slot_utilization_pct.append(
            (busy_slots / total_slots * 100.0) if total_slots > 0 else 0.0
        )

        # L3A object count
        if self._l3a_shared:
            self.metrics.l3a_object_count.append(len(self._shared_l3a.objects))
        else:
            self.metrics.l3a_object_count.append(
                sum(len(s.objects) for s in self._worker_l3a_stores if s)
            )

        # Cold evictions this epoch (delta since last epoch)
        self.metrics.cold_evictions_per_epoch.append(0)  # placeholder, incremented below

        # Run L1 eviction checks per GPU node
        for node_id, node in enumerate(self.nodes):
            eviction = self._node_eviction[node_id]
            if eviction.needs_l1_eviction():
                l1 = node.l1_store
                target = int(l1.capacity_bytes * self.config.eviction_hbm_threshold * 0.9)
                to_free = l1.used_bytes - target
                if to_free > 0:
                    evicted = eviction.evict_l1_to_l2(to_free)
                    self.metrics.l1_to_l2_evictions += len(evicted)

        # Run L2 -> L3A hibernation checks per worker (L2 is shared within worker)
        checked_workers = set()
        for node_id, node in enumerate(self.nodes):
            if node.worker_id in checked_workers:
                continue
            checked_workers.add(node.worker_id)
            eviction = self._node_eviction[node_id]
            expired_l2 = eviction.find_ttl_expired_l2(self.sim_clock_us)
            for key in expired_l2:
                eviction.hibernate_l2_to_l3a(key)
                self.metrics.l2_to_l3a_evictions += 1

        # L3a TTL checks
        if self._l3a_shared:
            expired_l3a = self._node_eviction[0].find_ttl_expired_l3a(self.sim_clock_us)
            for key in expired_l3a:
                obj = self._shared_l3a.get(key)
                if obj and obj.session_id != "__shared__":
                    sess = self.sessions.get(obj.session_id)
                    if sess and self.sim_clock_us <= sess.end_us:
                        continue
                self._shared_l3a.remove(key)
                self.metrics.session_cold_evictions += 1
                if self.metrics.cold_evictions_per_epoch:
                    self.metrics.cold_evictions_per_epoch[-1] += 1
        else:
            # Local L3A: check once per worker
            checked_workers_l3a = set()
            for node_id, node in enumerate(self.nodes):
                if node.worker_id in checked_workers_l3a:
                    continue
                checked_workers_l3a.add(node.worker_id)
                eviction = self._node_eviction[node_id]
                expired_l3a = eviction.find_ttl_expired_l3a(self.sim_clock_us)
                for key in expired_l3a:
                    obj = node.l3a_store.get(key)
                    if obj and obj.session_id != "__shared__":
                        sess = self.sessions.get(obj.session_id)
                        if sess and self.sim_clock_us <= sess.end_us:
                            continue
                    node.l3a_store.remove(key)
                    self.metrics.session_cold_evictions += 1
                    if self.metrics.cold_evictions_per_epoch:
                        self.metrics.cold_evictions_per_epoch[-1] += 1

    # ─── Helpers ───

    def _drain_pending_prefills(self) -> None:
        """Backward-compat: drain node[0]'s pending queue."""
        node = self.nodes[0]
        collecting = self.sim_clock_us >= int(self.config.warmup_s * 1_000_000)
        self._push_drain_node(node, collecting)

    def _place_kv_object(
        self,
        cache_key: str,
        session_id: str,
        profile,
        size: int,
        total_tokens: int,
        collecting: bool,
        shared_prefix_id: str | None = None,
        node_id: int = 0,
    ) -> Tier | None:
        """
        Try to place a KV object into the target node's L1; if it doesn't fit,
        fall through to that node's L2, then to shared L3A.
        """
        if shared_prefix_id is None and profile.shared_system_prefix_tokens > 0:
            sp_id = profile.name
        else:
            sp_id = shared_prefix_id

        node = self._get_node(node_id)
        eviction = self._get_node_eviction(node_id)

        # Try node's L1
        l1 = node.l1_store
        if not l1.can_fit(size):
            evicted = eviction.evict_l1_to_l2(
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
                    payload={"cache_key": cache_key, "target": "l2_hibernate", "node_id": node_id},
                ))
            return Tier.L1

        # L1 too small — try node's L2
        l2 = node.l2_store
        if not l2.can_fit(size):
            expired = eviction.find_ttl_expired_l2(self.sim_clock_us)
            for ek in expired:
                eviction.hibernate_l2_to_l3a(ek)
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
                    payload={"cache_key": cache_key, "target": "l3a_hibernate", "node_id": node_id},
                ))
            return Tier.L2

        # L2 too small — try L3A (shared or local)
        l3a = self._get_l3a_for_node(node_id)
        if not l3a.can_fit(size):
            evicted = eviction.cleanup_l3a(
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

        return None

    def _transition(self, request_id: str, next_state: RequestState) -> None:
        current = self.request_states.get(request_id)
        if current is not None:
            validate_transition(current, next_state)
        self.request_states[request_id] = next_state

    def _get_profile(self, name: str):
        return self._profile_map.get(name, self.config.profiles[0])
