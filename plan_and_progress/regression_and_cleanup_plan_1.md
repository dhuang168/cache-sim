# Regression Tests, Report Cleanup, Debug Skill — Plan

Date: 2026-03-21

## (a) Missing tests + dev methodology update

### What was missing when local/global L3A was introduced
1. No test that local L3A only searches own worker's SSD
2. No test that session migration + local L3A = higher miss rate than global
3. No multi-worker test with enough duration for L1/L2 to saturate
4. `test_global_vs_local_l3a_hit_rate` used short sim where L1 absorbed everything — never exercised L3A

### Tests to add
- `test_local_l3a_worker_isolation`: local L3A on worker 0 is NOT visible from worker 1
- `test_global_l3a_cross_worker_access`: global L3A hit from any worker
- `test_session_migration_global_advantage`: with forced migration (small L1/L2 + multiple workers), global > local hit rate

### Dev methodology update
Add: "For every new feature, write tests that use extreme conditions to exercise the feature as the sole differentiator in the shortest possible time"

## (b) Report cleanup
- Remove outdated SSD sweep that showed TIE (was based on buggy cross-worker search)
- Rewrite "When does global L3A matter" with correct findings
- Remove the old triage conclusion that said "affinity keeps sessions on worker"
- Keep the methodology sections (they're still valid)

## (c) Debug skill
Distill the observe→question→investigate→fix cycle into a reusable skill document
