"""
Unit tests for ChunkTierStore and ChunkIndex.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from sim.chunk_store import ChunkObject, ChunkTierStore, ChunkIndex, chunk_hash_for
from sim.cache import Tier, BlockLayout


def _make_chunk(chunk_hash: str, size_bytes: int = 80_000_000, ref_count: int = 1,
                last_accessed: int = 1000, chunk_index: int = 0) -> ChunkObject:
    """Create a test chunk (~80MB, like 256 tokens at 70B FP16)."""
    block_size = 5120
    from sim.cache import allocated_blocks
    return ChunkObject(
        chunk_hash=chunk_hash,
        tier=Tier.L1,
        size_bytes=size_bytes,
        block_count=allocated_blocks(size_bytes, block_size),
        created_at_us=1000,
        last_accessed_at_us=last_accessed,
        ref_count=ref_count,
        block_layout=BlockLayout.L1_SMALL,
        chunk_index=chunk_index,
    )


def test_novel_insert():
    """Inserting a new chunk returns True and increases used_bytes."""
    store = ChunkTierStore("L1", capacity_bytes=1_000_000_000, block_size_bytes=5120)
    chunk = _make_chunk("chunk-coding-0")
    assert store.insert_or_ref("chunk-coding-0", chunk) is True
    assert store.used_bytes > 0
    assert len(store.chunks) == 1
    assert store.chunks["chunk-coding-0"].ref_count == 1


def test_dedup_insert():
    """Inserting same hash returns False, increments ref_count, no used_bytes change."""
    store = ChunkTierStore("L1", capacity_bytes=1_000_000_000, block_size_bytes=5120)
    chunk1 = _make_chunk("chunk-coding-0")
    store.insert_or_ref("chunk-coding-0", chunk1)
    used_after_first = store.used_bytes

    chunk2 = _make_chunk("chunk-coding-0", last_accessed=2000)
    assert store.insert_or_ref("chunk-coding-0", chunk2) is False
    assert store.used_bytes == used_after_first  # no additional storage
    assert store.chunks["chunk-coding-0"].ref_count == 2
    assert store.chunks["chunk-coding-0"].last_accessed_at_us == 2000


def test_deref_removes_at_zero():
    """Deref to ref_count=0 removes chunk and frees bytes."""
    store = ChunkTierStore("L1", capacity_bytes=1_000_000_000, block_size_bytes=5120)
    chunk = _make_chunk("chunk-coding-0")
    store.insert_or_ref("chunk-coding-0", chunk)
    used = store.used_bytes

    assert store.deref("chunk-coding-0") is True  # removed
    assert store.used_bytes == 0
    assert len(store.chunks) == 0


def test_deref_keeps_above_zero():
    """Deref from ref_count=2 keeps chunk alive."""
    store = ChunkTierStore("L1", capacity_bytes=1_000_000_000, block_size_bytes=5120)
    chunk = _make_chunk("chunk-coding-0")
    store.insert_or_ref("chunk-coding-0", chunk)
    # Add second ref
    chunk2 = _make_chunk("chunk-coding-0")
    store.insert_or_ref("chunk-coding-0", chunk2)

    assert store.deref("chunk-coding-0") is False  # not removed
    assert store.chunks["chunk-coding-0"].ref_count == 1
    assert store.used_bytes > 0


def test_capacity_check():
    """can_fit returns False when store is full."""
    # 200 blocks * 5120 = 1,024,000 bytes capacity
    store = ChunkTierStore("L1", capacity_bytes=1_024_000, block_size_bytes=5120)
    assert store.can_fit(512_000) is True
    # Insert ~512KB (100 blocks)
    chunk = _make_chunk("chunk-0", size_bytes=512_000)
    store.insert_or_ref("chunk-0", chunk)
    # Another 512KB should still fit
    assert store.can_fit(512_000) is True
    # But 600KB won't (would need 118 blocks = 604,160, total > 1,024,000)
    assert store.can_fit(600_000) is False


def test_evict_lru_prefers_single_ref():
    """LRU eviction evicts ref_count=1 before ref_count>1."""
    store = ChunkTierStore("L1", capacity_bytes=1_000_000_000, block_size_bytes=5120)
    # Insert shared chunk (ref_count=2, recent access)
    shared = _make_chunk("shared", last_accessed=5000, ref_count=1)
    store.insert_or_ref("shared", shared)
    store.insert_or_ref("shared", _make_chunk("shared", last_accessed=5000))  # ref=2

    # Insert private chunk (ref_count=1, older access)
    private = _make_chunk("private", last_accessed=3000)
    store.insert_or_ref("private", private)

    # Evict enough for one chunk
    evicted = store.evict_lru(1)
    assert "private" in evicted  # single-ref evicted first
    assert "shared" not in evicted


def test_occupancy_pct():
    """Correct occupancy calculation."""
    store = ChunkTierStore("L1", capacity_bytes=1_000_000, block_size_bytes=5120)
    assert store.occupancy_pct == 0.0
    chunk = _make_chunk("chunk-0", size_bytes=5120)  # exactly 1 block
    store.insert_or_ref("chunk-0", chunk)
    assert store.occupancy_pct == pytest.approx(5120 / 1_000_000)


# ─── ChunkIndex tests ───

def test_chunk_index_consecutive_lookup():
    """ChunkIndex returns correct consecutive count."""
    store = ChunkTierStore("L1", capacity_bytes=1_000_000_000, block_size_bytes=5120)
    for i in range(5):
        chunk = _make_chunk(f"chunk-{i}", chunk_index=i)
        store.insert_or_ref(f"chunk-{i}", chunk)

    idx = ChunkIndex()
    idx.register_chunks("s1", [(i, f"chunk-{i}") for i in range(5)])

    cached, tier = idx.lookup_consecutive("s1", 5, [store])
    assert cached == 5
    assert tier == Tier.L1


def test_chunk_index_gap_stops_consecutive():
    """Missing chunk in middle stops consecutive count."""
    store = ChunkTierStore("L1", capacity_bytes=1_000_000_000, block_size_bytes=5120)
    for i in [0, 1, 3, 4]:  # gap at index 2
        chunk = _make_chunk(f"chunk-{i}", chunk_index=i)
        store.insert_or_ref(f"chunk-{i}", chunk)

    idx = ChunkIndex()
    idx.register_chunks("s1", [(i, f"chunk-{i}") for i in [0, 1, 3, 4]])

    cached, tier = idx.lookup_consecutive("s1", 5, [store])
    assert cached == 2  # stops at gap


def test_chunk_hash_for():
    """Shared chunks use profile hash, session chunks use session hash."""
    h_shared = chunk_hash_for("coding", "s1", chunk_index=0, shared_chunks=78)
    assert h_shared == "chunk-coding-0"

    h_session = chunk_hash_for("coding", "s1", chunk_index=78, shared_chunks=78)
    assert h_session == "chunk-s1-78"

    # Same shared chunk for different sessions
    h_shared2 = chunk_hash_for("coding", "s2", chunk_index=0, shared_chunks=78)
    assert h_shared == h_shared2  # dedup!
