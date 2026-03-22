# Phase 0: Test Hardening Report

Date: 2026-03-22

## Summary

Added 8 config consistency regression tests covering every bug found during development that didn't have a test. Full suite: **35 tests in 4.66s**.

## New Tests (`tests/test_config_consistency.py`)

| Test | Bug It Covers | Technique |
|------|--------------|-----------|
| `test_oracle_covers_workload_range` | Oracle clamped at 32k tokens | Verify oracle(max_context) > oracle(max_context/2) for each profile |
| `test_oracle_monotonically_increasing` | Oracle clamping | Check oracle(n+1) ≥ oracle(n) across full range |
| `test_l2_occupancy_never_exceeds_100` | L2 eviction skipped can_fit() | Run stressed sim, assert max L2 occ ≤ 100% |
| `test_all_profiles_generate_sessions` | Diurnal rate = 0 at midnight | Verify each profile creates ≥1 session |
| `test_session_proportions_match_profile_mix` | profile_mix didn't scale rate | Verify proportions within ±15% |
| `test_queue_depth_nonzero_under_load` | Queue depth read wrong variable | Stressed sim → queue depth > 0 |
| `test_ttl_migrations_separate_from_pressure` | Conflated eviction counters | Verify both metrics exist and tracked separately |
| `test_no_global_l3a_penalty_single_worker` | Remote penalty at 1 worker | 1-worker global L3A → n_workers=1 → no penalty |

## Test Suite Summary

| File | Tests | Focus |
|------|-------|-------|
| `test_invariants.py` | 4 | Boundary-condition physics invariants |
| `test_kv_size.py` | 4 | KV size math, block allocation |
| `test_oracle.py` | 5 | Prefill/decode oracle correctness |
| `test_multinode.py` | 14 | Multi-node dispatch, topology, L3A isolation |
| `test_config_consistency.py` | 8 | Bug regression, config validation |
| **Total** | **35** | **4.66s** |

## Bug Coverage

All 12 bugs from CLAUDE.md debug history now have regression tests:
- 4 covered by invariant tests (original)
- 6 covered by multinode tests (added during development)
- 8 covered by config consistency tests (this phase) — some bugs have multiple covering tests
