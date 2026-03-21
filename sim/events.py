from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class EventType(Enum):
    REQUEST_ARRIVAL = auto()
    PREFILL_START = auto()
    PREFILL_COMPLETE = auto()
    DECODE_START = auto()
    DECODE_COMPLETE = auto()
    TTL_FIRE = auto()
    TIER_EVICTION = auto()
    SESSION_RESUME = auto()
    EPOCH_REPORT = auto()
    NODE_PULL_CHECK = auto()


class RequestState(Enum):
    QUEUED = auto()
    CACHE_LOOKUP = auto()
    HIT_L1 = auto()
    HIT_L2 = auto()
    HIT_L3A = auto()
    MISS = auto()
    PREFILLING = auto()
    DECODE_QUEUED = auto()
    DECODING = auto()
    KV_WRITE = auto()
    COMPLETE = auto()


# Valid FSM transitions
_VALID_TRANSITIONS: dict[RequestState, set[RequestState]] = {
    RequestState.QUEUED: {RequestState.CACHE_LOOKUP},
    RequestState.CACHE_LOOKUP: {
        RequestState.HIT_L1,
        RequestState.HIT_L2,
        RequestState.HIT_L3A,
        RequestState.MISS,
    },
    RequestState.HIT_L1: {RequestState.PREFILLING},
    RequestState.HIT_L2: {RequestState.PREFILLING},
    RequestState.HIT_L3A: {RequestState.PREFILLING},
    RequestState.MISS: {RequestState.PREFILLING},
    RequestState.PREFILLING: {RequestState.DECODE_QUEUED},
    RequestState.DECODE_QUEUED: {RequestState.DECODING},
    RequestState.DECODING: {RequestState.KV_WRITE},
    RequestState.KV_WRITE: {RequestState.COMPLETE},
}


class SimError(Exception):
    pass


def validate_transition(current: RequestState, next_state: RequestState) -> None:
    valid = _VALID_TRANSITIONS.get(current, set())
    if next_state not in valid:
        raise SimError(
            f"Illegal FSM transition: {current.name} -> {next_state.name}"
        )


@dataclass(order=True)
class Event:
    time_us: int
    seq: int
    event_type: EventType = field(compare=False)
    session_id: str = field(compare=False)
    request_id: Optional[str] = field(default=None, compare=False)
    payload: Optional[dict] = field(default=None, compare=False)
