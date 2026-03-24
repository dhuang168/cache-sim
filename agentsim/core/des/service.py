from __future__ import annotations
from collections import deque
from typing import Optional

from agentsim.core.des.config import ServiceConfig
from agentsim.core.des.events import Event, EventType


class ServiceModel:
    """
    Models GPU serving as two distinct resource pools with a queue between them.

        ARRIVAL -> [PrefillQueue] -> PrefillPool -> [DecodeQueue] -> DecodePool -> COMPLETE

    Cache savings reduce effective input length to the prefill pool only.
    """

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.prefill_slots_free = config.n_prefill_slots
        self.decode_slots_free = config.n_decode_slots
        self.prefill_queue: deque[Event] = deque()
        self.decode_queue: deque[Event] = deque()
        self._prefill_blocked: dict[str, int] = {}  # request_id -> blocked_since_us

    @property
    def prefill_queue_full(self) -> bool:
        return len(self.prefill_queue) >= self.config.prefill_queue_max

    @property
    def decode_queue_full(self) -> bool:
        return len(self.decode_queue) >= self.config.decode_queue_max

    def try_admit_prefill(self, event: Event) -> bool:
        """Returns True if a prefill slot was available and consumed."""
        if self.prefill_slots_free > 0:
            self.prefill_slots_free -= 1
            return True
        if not self.prefill_queue_full:
            self.prefill_queue.append(event)
        # If queue is also full, event is dropped (backpressure)
        return False

    def complete_prefill(self, event: Event, current_time_us: int) -> Optional[Event]:
        """
        Frees a prefill slot, returns a DECODE_START event if decode slot available.
        If decode queue is full, blocks (holds prefill slot via _prefill_blocked).
        """
        if self.decode_slots_free > 0:
            self.decode_slots_free -= 1
            return Event(
                time_us=current_time_us,
                seq=0,  # will be reassigned by engine
                event_type=EventType.DECODE_START,
                session_id=event.session_id,
                request_id=event.request_id,
                payload=event.payload,
            )
        elif not self.decode_queue_full:
            # Queue for decode, but hold the prefill slot (backpressure)
            self._prefill_blocked[event.request_id or ""] = current_time_us
            self.decode_queue.append(event)
            return None
        else:
            # Both decode slots and decode queue full — hold prefill slot
            self._prefill_blocked[event.request_id or ""] = current_time_us
            self.decode_queue.append(event)
            return None

    def complete_decode(self, event: Event, current_time_us: int) -> Optional[Event]:
        """Frees decode slot; attempts to drain decode queue."""
        self.decode_slots_free += 1
        return self._drain_decode_queue(current_time_us)

    def _drain_prefill_queue(self) -> None:
        """Try to start queued prefills if slots available."""
        while self.prefill_queue and self.prefill_slots_free > 0:
            self.prefill_slots_free -= 1
            self.prefill_queue.popleft()
            # The engine will handle scheduling the actual prefill

    def _drain_decode_queue(self, current_time_us: int) -> Optional[Event]:
        """Try to start a queued decode if slots available."""
        if not self.decode_queue or self.decode_slots_free <= 0:
            return None

        queued_event = self.decode_queue.popleft()
        self.decode_slots_free -= 1

        # Unblock the prefill slot that was held
        req_id = queued_event.request_id or ""
        if req_id in self._prefill_blocked:
            del self._prefill_blocked[req_id]
            self.prefill_slots_free += 1
            self._drain_prefill_queue()

        return Event(
            time_us=current_time_us,
            seq=0,
            event_type=EventType.DECODE_START,
            session_id=queued_event.session_id,
            request_id=queued_event.request_id,
            payload=queued_event.payload,
        )

    def get_blocked_us(self, current_time_us: int) -> int:
        """Sum of blocked time across all currently blocked prefill slots."""
        total = 0
        for blocked_since in self._prefill_blocked.values():
            total += current_time_us - blocked_since
        return total

    @property
    def active_decode_sequences(self) -> int:
        return self.config.n_decode_slots - self.decode_slots_free
