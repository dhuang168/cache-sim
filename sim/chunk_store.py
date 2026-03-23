"""
Chunk-level KV cache storage with hash-based deduplication.

LMCache-style: KV is stored at fixed-size chunk granularity (default 256 tokens).
Identical chunks across sessions share storage via ref counting.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

from sim.cache import Tier, BlockLayout, TIER_TO_LAYOUT, kv_size_bytes, allocated_blocks
from sim.config import ModelConfig


@dataclass
class ChunkObject:
    """A single KV cache chunk (e.g., 256 tokens)."""
    chunk_hash: str
    tier: Tier
    size_bytes: int
    block_count: int
    created_at_us: int
    last_accessed_at_us: int
    ref_count: int
    block_layout: BlockLayout
    is_hibernated: bool = False
    chunk_index: int = 0
    profile_name: str = ""


class ChunkTierStore:
    """Manages ChunkObjects within a single storage tier with hash-based dedup."""

    def __init__(self, name: str, capacity_bytes: int, block_size_bytes: int):
        self.name = name
        self.capacity_bytes = capacity_bytes
        self.block_size_bytes = block_size_bytes
        self.chunks: dict[str, ChunkObject] = {}
        self.used_bytes: int = 0

    @property
    def objects(self) -> dict[str, ChunkObject]:
        """Alias for compat with epoch reporting that iterates store.objects."""
        return self.chunks

    @property
    def occupancy_pct(self) -> float:
        if self.capacity_bytes == 0:
            return 0.0
        return self.used_bytes / self.capacity_bytes

    def can_fit(self, size_bytes: int) -> bool:
        n_blocks = allocated_blocks(size_bytes, self.block_size_bytes)
        return self.used_bytes + n_blocks * self.block_size_bytes <= self.capacity_bytes

    def get(self, chunk_hash: str) -> Optional[ChunkObject]:
        return self.chunks.get(chunk_hash)

    def insert_or_ref(self, chunk_hash: str, chunk: ChunkObject) -> bool:
        """Insert chunk if novel; increment ref_count if duplicate.
        Returns True if novel insert, False if dedup hit.
        Caller must ensure capacity (call can_fit or evict first for novel inserts)."""
        existing = self.chunks.get(chunk_hash)
        if existing:
            existing.ref_count += 1
            existing.last_accessed_at_us = max(existing.last_accessed_at_us, chunk.last_accessed_at_us)
            return False  # dedup hit
        alloc = chunk.block_count * self.block_size_bytes
        self.chunks[chunk_hash] = chunk
        self.used_bytes += alloc
        return True  # novel

    def remove(self, chunk_hash: str) -> Optional[ChunkObject]:
        """Remove chunk regardless of ref_count. Returns the removed chunk."""
        chunk = self.chunks.pop(chunk_hash, None)
        if chunk is not None:
            self.used_bytes -= chunk.block_count * self.block_size_bytes
        return chunk

    def deref(self, chunk_hash: str) -> bool:
        """Decrement ref_count. Remove if it reaches 0. Returns True if removed."""
        chunk = self.chunks.get(chunk_hash)
        if not chunk:
            return False
        chunk.ref_count -= 1
        if chunk.ref_count <= 0:
            self.used_bytes -= chunk.block_count * self.block_size_bytes
            del self.chunks[chunk_hash]
            return True
        return False

    def evict_lru(self, n_bytes_needed: int) -> list[str]:
        """Evict chunks by LRU, preferring ref_count=1. Returns evicted hashes."""
        candidates = sorted(
            self.chunks.items(),
            key=lambda kv: (kv[1].ref_count > 1, kv[1].last_accessed_at_us),
        )
        evicted = []
        freed = 0
        for chunk_hash, chunk in candidates:
            if freed >= n_bytes_needed:
                break
            alloc = chunk.block_count * self.block_size_bytes
            del self.chunks[chunk_hash]
            self.used_bytes -= alloc
            freed += alloc
            evicted.append(chunk_hash)
        return evicted

    def fragmentation_bytes(self) -> int:
        total = 0
        for chunk in self.chunks.values():
            total += chunk.block_count * self.block_size_bytes - chunk.size_bytes
        return total


def chunk_hash_for(profile_name: str, session_id: str,
                   chunk_index: int, shared_chunks: int) -> str:
    """Compute deterministic chunk hash.
    Shared prefix chunks (index < shared_chunks) hash by profile — deduped across sessions.
    Session-unique chunks hash by session — unique per session."""
    if chunk_index < shared_chunks:
        return f"chunk-{profile_name}-{chunk_index}"
    return f"chunk-{session_id}-{chunk_index}"


class ChunkIndex:
    """Tracks which chunk hashes are cached per session for fast consecutive lookup."""

    def __init__(self):
        # session_id -> list of (chunk_index, chunk_hash) sorted by index
        self._session_chunks: dict[str, dict[int, str]] = {}

    def register_chunks(self, session_id: str, chunk_hashes: list[tuple[int, str]]) -> None:
        """Record that these chunks are cached for this session.
        chunk_hashes: list of (chunk_index, chunk_hash)."""
        if session_id not in self._session_chunks:
            self._session_chunks[session_id] = {}
        for idx, h in chunk_hashes:
            self._session_chunks[session_id][idx] = h

    def lookup_consecutive(
        self, session_id: str, total_chunks: int,
        stores: list[ChunkTierStore],
    ) -> tuple[int, Tier | None]:
        """Count consecutive cached chunks from position 0.
        Checks each store in order (L1, L2, L3A).
        Returns (cached_chunk_count, tier_of_last_cached_chunk)."""
        session_map = self._session_chunks.get(session_id, {})
        cached = 0
        last_tier = None
        for i in range(total_chunks):
            chunk_hash = session_map.get(i)
            if chunk_hash is None:
                break
            # Find which store has this chunk
            found = False
            for store in stores:
                obj = store.get(chunk_hash)
                if obj:
                    last_tier = obj.tier
                    obj.last_accessed_at_us = max(obj.last_accessed_at_us, obj.last_accessed_at_us)
                    found = True
                    break
            if not found:
                break
            cached += 1
        return cached, last_tier

    def remove_session(self, session_id: str) -> None:
        self._session_chunks.pop(session_id, None)
