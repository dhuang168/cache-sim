# Progress: Phase 0.5 — Validation Contract

**Date:** 2026-03-23
**Branch:** feature/des-core-swap
**Status:** Complete — all 4 gate items checked

## What was done

1. **Golden reference established** (`results/golden/phase05_reference.json`):
   - Config: heavy_coding, peak=15, 5-min sim, seed=42, pull dispatch
   - 8 configs: 1/2/4 workers × global/local L3A
   - Each run ~8-15s → total baseline capture ~60s
   - Key numbers: 99.93% hit rate, recompute=26.3%, 9849 completed per config

2. **Tolerance-band tests** (`tests/test_phase05_baseline.py`):
   - 12 tests across 3 scenarios, all pass in 67s
   - Scenario A (1W): hit rate ±2pt, TTFT p50 ±20%, completed count exact match
   - Scenario B (4W global vs local): both ≥99% at controlled load, golden match
   - Scenario C (scaling): 1W global≈local, all ≥99%, more workers→lower utilization

3. **Design decision**: Used peak=15 instead of peak=100 (original report).
   - peak=100 creates severe overload (5M+ requests, hours to run) — not feasible for CI
   - peak=15 creates meaningful load (80% slot util, 10K requests) in 15s per run
   - Trade-off: global-vs-local hit rate divergence doesn't appear at controlled load (both 99.9%). The divergence requires overload + session migration. Tests verify tier distribution and TTFT differences instead.

## Gate checklist
- [x] All three benchmark scenarios defined in code as test fixtures
- [x] Tolerance bands defined in golden file and used by tests
- [x] Reference numbers from cache-sim committed as golden files
- [x] Authoritative vs exploratory metric list agreed (hit rate, TTFT, queue wait = authoritative)

## What was deferred
- High-load (peak=100) validation — too slow for CI, kept in manual heavy_coding_analysis.py
- Phase 1: Port cache-sim DES core into agentsim/core/des/

## Next step
Phase 1 — Core DES Layer: port cache-sim modules into agentsim/core/des/ with byte-based accounting.
