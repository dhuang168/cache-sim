# Progress: Simulator Performance Fix — has_session_cached O(1)

**Date:** 2026-03-24
**Branch:** feature/des-core-swap

## Root Cause

`PrefillNode.has_session_cached()` scanned ALL objects in L1 + L2 stores per call. Called 1.3M times during pull dispatch (each pull scores every job in the global queue × every node's object scan). Cost: O(queue_size × objects_per_store) per pull.

## Fix

Added `_session_refcount: dict[str, int]` to `TierStore`. Incremented on insert, decremented on remove. `has_session(sid)` is now O(1) dict lookup.

This is guaranteed equivalent — TierStore tracks the exact same object presence as the original scan, just indexed.

## Verification

- **Exact match**: old engine and new engine produce identical hit rates (all 4 tiers match to 4 decimal places)
- **Speedup**: 3.5× on 2-min sim, **27× on 5-min sim** (2,668s → 99s)
- **All 211 tests pass**
