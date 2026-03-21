from __future__ import annotations
import bisect
from collections import deque

from sim.cache import TierStore, Tier


class PrefillNode:
    """Per-node state: local L1/L2, prefill slots, and pending queue."""

    def __init__(
        self,
        node_id: int,
        l1_store: TierStore,
        l2_store: TierStore,
        prefill_slots: int,
        prefill_queue_max: int,
        l3a_store: TierStore | None = None,
    ):
        self.node_id = node_id
        self.l1_store = l1_store
        self.l2_store = l2_store
        self.l3a_store = l3a_store  # None when L3A is shared (global)
        self.prefill_slots_total = prefill_slots
        self.prefill_slots_free = prefill_slots
        self.prefill_queue_max = prefill_queue_max
        self.pending_prefills: deque[tuple] = deque()
        self.active_completions: list[int] = []  # sorted completion times

    def has_session_cached(self, session_id: str) -> bool:
        """Check if any KV object for this session is in this node's L1 or L2."""
        for obj in self.l1_store.objects.values():
            if obj.session_id == session_id:
                return True
        for obj in self.l2_store.objects.values():
            if obj.session_id == session_id:
                return True
        return False

    def projected_free_time_us(self, current_us: int) -> int:
        """Earliest time a prefill slot becomes available."""
        if self.prefill_slots_free > 0:
            return current_us
        if not self.active_completions:
            return current_us
        return self.active_completions[0]

    def queue_pressure(self) -> float:
        """Fraction of queue capacity used."""
        if self.prefill_queue_max <= 0:
            return 1.0
        return len(self.pending_prefills) / self.prefill_queue_max

    def add_completion(self, completion_us: int) -> None:
        bisect.insort(self.active_completions, completion_us)

    def remove_completion(self, completion_us: int) -> None:
        try:
            self.active_completions.remove(completion_us)
        except ValueError:
            pass
