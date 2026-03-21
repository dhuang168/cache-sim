from __future__ import annotations
from collections import deque
from typing import Optional

from sim.node import PrefillNode


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

    def pull(self, node: PrefillNode, current_us: int) -> Optional[tuple[str, str, dict]]:
        """Pull best job for this node from the global queue."""
        if not self.global_queue:
            return None

        best_idx = -1
        best_score = float('-inf')
        AFFINITY_BONUS = 1_000_000_000  # 1000s in us — huge bonus

        for i, (session_id, request_id, payload, arrival_us) in enumerate(self.global_queue):
            age_us = current_us - arrival_us
            affinity = AFFINITY_BONUS if node.has_session_cached(session_id) else 0
            score = affinity + age_us
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            return None

        session_id, request_id, payload, _ = self.global_queue[best_idx]
        del self.global_queue[best_idx]
        return (session_id, request_id, payload)
