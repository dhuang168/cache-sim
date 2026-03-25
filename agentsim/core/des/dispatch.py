from __future__ import annotations
import hashlib
from collections import deque
from typing import Optional

from agentsim.core.des.node import PrefillNode


class PushDispatcher:
    """Cache-affinity-aware push dispatcher."""

    def __init__(self, nodes: list[PrefillNode]):
        self.nodes = nodes

    def dispatch(
        self,
        session_id: str,
        request_id: str,
        payload: dict,
        current_us: int,
    ) -> PrefillNode:
        """Pick the best node for this request."""
        # Find nodes with cache affinity
        affinity_nodes = [
            n for n in self.nodes
            if n.has_session_cached(session_id)
        ]

        # Prefer affinity nodes that aren't overloaded
        if affinity_nodes:
            non_overloaded = [n for n in affinity_nodes if n.queue_pressure() < 0.9]
            if non_overloaded:
                return min(
                    non_overloaded,
                    key=lambda n: (n.projected_free_time_us(current_us), len(n.pending_prefills)),
                )

        # No affinity or all overloaded: pick node with earliest free time
        return min(
            self.nodes,
            key=lambda n: (n.projected_free_time_us(current_us), len(n.pending_prefills)),
        )


class SmartPushDispatcher:
    """Predictive push dispatcher with full tier visibility.
    Scores each node by expected_total = expected_wait + expected_prefill.
    Considers L1, L2, AND L3A affinity (not just L1/L2)."""

    def __init__(self, nodes: list[PrefillNode], l3a_store=None):
        self.nodes = nodes
        self.l3a_store = l3a_store  # global L3A store (or None for local)

    # Cost estimates (relative, in microseconds) for scoring
    L1_HIT_COST = 100_000       # 0.1s — L1 is fast
    L2_HIT_COST = 2_000_000     # 2s — L2 transfer + partial compute
    L3A_HIT_COST = 15_000_000   # 15s — L3A transfer + partial compute
    COLD_MISS_COST = 60_000_000 # 60s — full recompute for coding workloads

    def _cache_tier_for_session(self, session_id: str, node: PrefillNode) -> str:
        """Check which tier has this session's KV on this node's worker."""
        for obj in node.l1_store.objects.values():
            if obj.session_id == session_id:
                return "L1"
        for obj in node.l2_store.objects.values():
            if obj.session_id == session_id:
                return "L2"
        # Check L3A — local or global
        if node.l3a_store:
            for obj in node.l3a_store.objects.values():
                if obj.session_id == session_id:
                    return "L3A_local"
        if self.l3a_store:
            for obj in self.l3a_store.objects.values():
                if obj.session_id == session_id:
                    return "L3A_global"
        return "miss"

    def _expected_prefill_cost(self, tier: str) -> int:
        if tier == "L1":
            return self.L1_HIT_COST
        elif tier == "L2":
            return self.L2_HIT_COST
        elif tier in ("L3A_local", "L3A_global"):
            return self.L3A_HIT_COST
        return self.COLD_MISS_COST

    def dispatch(
        self,
        session_id: str,
        request_id: str,
        payload: dict,
        current_us: int,
    ) -> PrefillNode:
        """Pick node with lowest expected total time (wait + prefill)."""
        best_node = None
        best_score = float('inf')

        for node in self.nodes:
            wait_us = node.projected_free_time_us(current_us) - current_us
            wait_us = max(0, wait_us) + len(node.pending_prefills) * 5_000_000  # ~5s per queued item

            tier = self._cache_tier_for_session(session_id, node)
            prefill_cost = self._expected_prefill_cost(tier)

            total = wait_us + prefill_cost
            if total < best_score:
                best_score = total
                best_node = node

        return best_node or self.nodes[0]


class PullDispatcher:
    """Global-queue pull dispatcher with affinity scoring."""

    def __init__(self, nodes: list[PrefillNode]):
        self.nodes = nodes
        self.global_queue: deque[tuple[str, str, dict, int]] = deque()
        # (session_id, request_id, payload, arrival_us)

    def enqueue(
        self,
        session_id: str,
        request_id: str,
        payload: dict,
        arrival_us: int,
    ) -> None:
        self.global_queue.append((session_id, request_id, payload, arrival_us))

    # Max jobs to scan per pull — caps O(queue_size) to O(K).
    # Production systems (llm-d, Mooncake) score top candidates, not entire queue.
    MAX_SCAN = 64

    def pull(self, node: PrefillNode, current_us: int) -> Optional[tuple[str, str, dict]]:
        """Pull best job for this node from the global queue.
        Scans up to MAX_SCAN oldest jobs for affinity; takes oldest if none match."""
        if not self.global_queue:
            return None

        best_idx = 0  # default: oldest job (front of queue)
        scan_limit = min(len(self.global_queue), self.MAX_SCAN)

        # Scan first K jobs — if any has affinity, take it (highest age wins ties)
        for i in range(scan_limit):
            session_id = self.global_queue[i][0]
            if node.has_session_cached(session_id):
                best_idx = i
                break  # first affinity match among oldest jobs — good enough

        session_id, request_id, payload, _ = self.global_queue[best_idx]
        del self.global_queue[best_idx]
        return (session_id, request_id, payload)


class PrefixHashDispatcher:
    """OpenAI-style prefix-hash routing.

    Routes requests to a target node based on a hash of the first N tokens
    of the prefix (profile name + shared prefix as proxy). If the target
    node is overloaded (queue pressure > threshold), overflows to the
    least-loaded node — modeling the ~15 req/min overflow behavior.

    This creates hotspots: popular system prompts concentrate on a few nodes
    while others sit idle. The overflow mechanism degrades cache hit rates
    because overflow nodes don't have the cached KV.
    """

    def __init__(
        self,
        nodes: list[PrefillNode],
        prefix_hash_tokens: int = 256,
        overflow_threshold: float = 0.9,
    ):
        self.nodes = nodes
        self.prefix_hash_tokens = prefix_hash_tokens
        self.overflow_threshold = overflow_threshold
        self.overflow_count = 0
        self.total_dispatches = 0

    def _compute_prefix_hash(self, profile_name: str, shared_prefix_tokens: int) -> int:
        """Hash the first N tokens of the prefix.

        In production, this would hash actual token IDs. We approximate by
        hashing (profile_name, min(shared_prefix_tokens, prefix_hash_tokens))
        since all sessions of the same profile share the same system prompt.
        """
        prefix_key = f"{profile_name}:{min(shared_prefix_tokens, self.prefix_hash_tokens)}"
        h = hashlib.md5(prefix_key.encode()).hexdigest()
        return int(h, 16)

    def dispatch(
        self,
        session_id: str,
        request_id: str,
        payload: dict,
        current_us: int,
    ) -> PrefillNode:
        """Route to target node by prefix hash. Overflow if overloaded."""
        self.total_dispatches += 1

        profile_name = payload.get("profile", "")
        shared_prefix = payload.get("shared_prefix_tokens", 0)
        prefix_hash = self._compute_prefix_hash(profile_name, shared_prefix)
        target_idx = prefix_hash % len(self.nodes)
        target_node = self.nodes[target_idx]

        # Check if target is overloaded
        if target_node.queue_pressure() < self.overflow_threshold:
            return target_node

        # Overflow: route to least-loaded node (cold miss likely)
        self.overflow_count += 1
        return min(
            self.nodes,
            key=lambda n: (n.queue_pressure(), n.projected_free_time_us(current_us)),
        )
