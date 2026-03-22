# Documentation Refresh — Plan

Date: 2026-03-22

## Scope
Update all 4 docs to reflect current codebase state (27 tests, worker topology, extended oracle, heavy_coding config, CLI args, etc.)

## Phase 1: README.md
- Update test count (14 multinode tests, 27 total)
- Add heavy_coding.json and legacy_v1.json to configs
- Add heavy_coding_analysis.py to scripts
- Note oracle extends to 262k tokens
- Add --config/--outdir to quick start

## Phase 2: CLAUDE.md
- Update multinode test count to 14
- Add config consistency rule to debug history

## Phase 3: docs/user_manual.md (major rewrite)
- Add sim_start_time_s field
- Clarify n_gpus_per_worker (worker topology explanation)
- Clarify L3A capacity semantics (per-worker, global = ×N)
- Explain profile_mix as rate multiplier
- Update benchmark table to 262k
- Add heavy coding recipes
- Add heavy_coding_analysis.py usage
- Add --config/--outdir CLI args

## Phase 4: docs/example_report.md
- Update to 27 tests
- Add 5 missing multinode test descriptions
- Update performance notes

## Phase 5: Crosscheck, commit, push
