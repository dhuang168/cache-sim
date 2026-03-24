# Progress: Phase 0 — Architecture Contracts

**Date:** 2026-03-23
**Branch:** feature/des-core-swap
**Status:** Complete — all 6 gate items checked

## What was done

1. **Pre-migration checkpoint** (Steps 1-4 from ONBOARDING.md):
   - All 105 tests pass, sanity_plots.py and heavy_coding_analysis.py run clean
   - Tagged `v0.1-prototype` (commit `4ccef69`)
   - Baseline captured: `results/v0.1-prototype-baseline/key_metrics.json` (5M requests, 99.85% hit, TTFT p50=140.7s)
   - Reusable script: `scripts/capture_baseline_metrics.py`
   - Migration branch `feature/des-core-swap` created, rollback tested

2. **Package skeleton** (Steps 5-9):
   - Copied all 8 source files from `files/` to correct `agentsim/` locations
   - **Fixed**: `oracle.py` imported from `integration.chips.profiles` (Layer 3) — violated "Core DES has no upward dependencies" contract. Changed to import `ConfidenceLabel` from `core.contracts` (Layer 1). Removed orphaned `CacheOracleBase` duplicate (use the one from contracts).
   - Import linter: 3 contracts pass, 0 broken
   - CI workflow: `.github/workflows/ci.yml` with pytest + lint-imports

3. **Phase 0 contract review**:
   - Read every class and docstring in `contracts.py`
   - Verified all 8 required interfaces present per ONBOARDING.md checklist
   - Wrote `tests/test_phase0_contracts.py` — 43 tests covering all 7 guiding principles
   - 148 total tests pass (105 existing + 43 Phase 0)

## Gate checklist
- [x] contracts.py reviewed — all 8 required interfaces present
- [x] No ambiguity about Layer 1 vs Layer 2 ownership
- [x] SimPy constraints documented (SweepEstimator PROHIBITED section)
- [x] core/des/README.md exists and points to contracts.py
- [x] lint-imports active (3 contracts kept)
- [x] test_phase0_contracts.py written and passing (43 tests)

## Key observations

- `SavingsEvent.classify()`: L2 transfers slower than recompute → MISS_RECOMPUTE (no L2_BREAK_EVEN). Matches cache-sim behavior where L2 hits are always worthwhile.
- `CacheObject` is `frozen=True` — correct for contracts, but Phase 1 engine needs mutable internal representations (e.g., `last_access_us` updates). Will need mutable internal version with frozen contract objects at the API boundary.
- Import boundary enforcement caught a real violation during setup — validates the lint-imports approach.

## What was deferred
- Phase 0.5: Define golden benchmark scenarios + tolerance bands
- Phase 1: Port cache-sim DES core into agentsim/core/des/

## Next step
Phase 0.5 — Validation Contract: define 3 benchmark scenarios (single-worker, global-vs-local, node scaling) with tolerance bands, commit golden reference numbers.
