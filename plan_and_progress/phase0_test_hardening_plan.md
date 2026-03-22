# Phase 0: Test Hardening and Speed — Plan

Date: 2026-03-22

## Goal
Cover all known bugs with regression tests. Improve test speed to <10s.

## Bug Audit: CLAUDE.md Debug History vs Test Coverage

| Bug | Has Test? | Action |
|-----|-----------|--------|
| L2 occupancy not collapsing with TTL=0 | ✅ `test_zero_ttl_collapses_l2` | — |
| Infinite L1 still showed evictions | ✅ `test_infinite_l1_no_evictions` | — |
| Sharing factor not 1.0 with zero shared prefix | ✅ `test_no_shared_prefix_sharing_factor_one` | — |
| BREAK_EVEN events with infinite bandwidth | ✅ `test_zero_bandwidth_penalty_prefers_restore` | — |
| Queue depth showed flat zero | ❌ No direct test | Add: verify queue_depth > 0 under load |
| L1 eviction conflated TTL vs pressure | ❌ No direct test | Add: verify TTL migrations and pressure evictions are separate |
| Diurnal rate = 0 at midnight | ❌ No direct test | Add: verify all profiles generate sessions at sim_start_time=10AM |
| Batch dominated despite profile_mix | ❌ No direct test | Add: verify session proportions match profile_mix ±10% |
| L2 saturation exceeded 100% | ❌ No direct test | Add: verify L2 occupancy ≤ 100% |
| Local L3A searched all workers' SSDs | ✅ `test_local_l3a_worker_isolation` | — |
| Oracle clamped at 32k tokens | ❌ No direct test | Add: verify oracle(65k) > oracle(32k) |
| Global L3A penalty applied at 1 worker | ❌ No direct test | Add: verify no remote penalty when n_workers=1 |

## New Tests to Add

### Config consistency tests (`tests/test_config_consistency.py`)
1. `test_oracle_covers_workload_range` — oracle table max tokens ≥ max expected context
2. `test_l2_occupancy_never_exceeds_100` — run sim, assert all L2 occupancy ≤ 100%
3. `test_all_profiles_generate_sessions` — verify each profile with mix > 0 creates sessions
4. `test_session_proportions_match_profile_mix` — proportions within ±15% of target
5. `test_queue_depth_nonzero_under_load` — stressed config → queue depth > 0
6. `test_ttl_migrations_separate_from_pressure` — verify both metrics tracked independently
7. `test_no_global_l3a_penalty_single_worker` — 1 worker global L3A → no remote penalty overhead
8. `test_oracle_monotonically_increasing` — oracle(n+1) >= oracle(n) for all n in range

### Speed optimization
- Current: 27 tests in ~5s (already fast after diurnal fix)
- Target: add ~8 more tests, stay <10s total
- Use minimal sim durations (5-10s) for consistency tests
- Use unit-level tests (no sim run) where possible (oracle, config validation)

## Implementation
1. Create `tests/test_config_consistency.py` with 8 new tests
2. Verify all 35 tests pass in <10s
3. Create `docs/phase0_test_report.md` (new report, don't modify existing)
4. Update CLAUDE.md test section
5. Commit + push
