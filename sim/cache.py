from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from sim.config import ModelConfig


class Tier(Enum):
    L1 = 1
    L2 = 2
    L3A = 3


class BlockLayout(Enum):
    L1_SMALL = "l1_small"
    L2_LARGE = "l2_large"
    L3A_ALIGNED = "l3a_aligned"


TIER_TO_LAYOUT = {
    Tier.L1: BlockLayout.L1_SMALL,
    Tier.L2: BlockLayout.L2_LARGE,
    Tier.L3A: BlockLayout.L3A_ALIGNED,
}


@dataclass
class CacheObject:
    session_id: str
    shared_prefix_id: Optional[str]
    token_range: tuple[int, int]  # (start_token, end_token) inclusive
    model_id: str
    tier: Tier
    size_bytes: int
    block_count: int
    created_at_us: int
    last_accessed_at_us: int
    ref_count: int
    block_layout: BlockLayout
    is_hibernated: bool = False

    @property
    def prefix_length(self) -> int:
        return self.token_range[1] - self.token_range[0] + 1


def kv_size_bytes(
    token_count: int,
    model: ModelConfig,
    precision_override: Optional[int] = None,
) -> int:
    """
    Compute exact KV cache size for a token sequence.
    Formula: 2 * n_layers * n_kv_heads * head_dim * bytes_per_element * token_count
    """
    bpe = precision_override or model.bytes_per_element
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * bpe * token_count


def allocated_blocks(size_bytes: int, block_size_bytes: int) -> int:
    """Ceiling division — always round up to next full block."""
    return math.ceil(size_bytes / block_size_bytes)


def block_waste_ratio(size_bytes: int, block_size_bytes: int) -> float:
    """Internal fragmentation as a fraction of allocated space."""
    n_blocks = allocated_blocks(size_bytes, block_size_bytes)
    allocated = n_blocks * block_size_bytes
    return (allocated - size_bytes) / allocated


class TierStore:
    """Manages CacheObjects within a single storage tier."""

    def __init__(self, name: str, capacity_bytes: int, block_size_bytes: int):
        self.name = name
        self.capacity_bytes = capacity_bytes
        self.block_size_bytes = block_size_bytes
        self.objects: dict[str, CacheObject] = {}  # cache_key -> CacheObject
        self.used_bytes: int = 0

    def allocated_bytes_for(self, obj: CacheObject) -> int:
        return obj.block_count * self.block_size_bytes

    @property
    def occupancy_pct(self) -> float:
        if self.capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes

    def can_fit(self, size_bytes: int) -> bool:
        n_blocks = allocated_blocks(size_bytes, self.block_size_bytes)
        return self.used_bytes + n_blocks * self.block_size_bytes <= self.capacity_bytes

    def insert(self, key: str, obj: CacheObject) -> None:
        alloc = obj.block_count * self.block_size_bytes
        self.objects[key] = obj
        self.used_bytes += alloc

    def remove(self, key: str) -> Optional[CacheObject]:
        obj = self.objects.pop(key, None)
        if obj is not None:
            self.used_bytes -= obj.block_count * self.block_size_bytes
        return obj

    def get(self, key: str) -> Optional[CacheObject]:
        return self.objects.get(key)

    def fragmentation_bytes(self) -> int:
        total = 0
        for obj in self.objects.values():
            total += obj.block_count * self.block_size_bytes - obj.size_bytes
        return total


class PrefixTrie:
    """
    Radix trie mapping token-sequence hash prefixes to CacheObject keys.

    Uses a dict-backed trie for per-session prefix tracking (write-heavy).
    """

    def __init__(self):
        self._data: dict[int, list[tuple[list[int], str]]] = {}
        self._entries: list[tuple[list[int], str, int]] = []  # (hashes, key, last_access)

    def lookup(self, token_hashes: list[int]) -> tuple[Optional[str], int]:
        """Returns (cache_key_or_None, depth_k) — longest prefix match."""
        best_key = None
        best_depth = 0
        for hashes, key, _ in self._entries:
            match_len = 0
            for i in range(min(len(hashes), len(token_hashes))):
                if hashes[i] == token_hashes[i]:
                    match_len += 1
                else:
                    break
            if match_len > best_depth:
                best_depth = match_len
                best_key = key
        return best_key, best_depth

    def insert(self, token_hashes: list[int], cache_key: str, access_time: int = 0) -> None:
        # Update existing entry if same key
        for i, (h, k, _) in enumerate(self._entries):
            if k == cache_key:
                self._entries[i] = (token_hashes, cache_key, access_time)
                return
        self._entries.append((token_hashes, cache_key, access_time))

    def evict_leaves(self, n: int) -> list[str]:
        """Evict n entries with smallest prefix overlap (leaf-like)."""
        if n <= 0 or not self._entries:
            return []
        # Sort by last access time (oldest first)
        self._entries.sort(key=lambda e: e[2])
        evicted = []
        for _ in range(min(n, len(self._entries))):
            _, key, _ = self._entries.pop(0)
            evicted.append(key)
        return evicted

    def remove_key(self, cache_key: str) -> None:
        self._entries = [(h, k, t) for h, k, t in self._entries if k != cache_key]
