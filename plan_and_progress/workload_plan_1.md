# Realistic Coding Workload — Plan #1

Date: 2026-03-21

## Motivation

Research (see `research_coding_workload_tokens.md`) shows the current "coding" profile drastically underestimates real workloads:
- `shared_system_prefix_tokens=2048` vs real ~20-30k
- `input_len_mean_tokens=800` vs real ~8-15k
- `prefix_stability=0.85→0.4` vs real ~0.95→0.80

This means KV objects are much smaller than reality, which is why cache tier pressure and global vs local L3A differences don't manifest.

## Phase 1: Update workload profiles

### Update existing "coding" profile

| Parameter | Current | Proposed | Rationale |
|---|---|---|---|
| `shared_system_prefix_tokens` | 2,048 | 20,000 | System prompt + tool defs |
| `input_len_mean_tokens` | 800 | 8,000 | User message + tool output per turn |
| `input_len_sigma_tokens` | 400 | 4,000 | Proportional spread |
| `context_growth_min_tokens` | 500 | 2,000 | Prior turn adds response + tool calls |
| `context_growth_max_tokens` | 2,000 | 10,000 | Large tool outputs |
| `prefix_stability_initial` | 0.85 | 0.95 | 95% of context is stable prefix |
| `prefix_stability_final` | 0.4 | 0.80 | Even late turns, 80%+ is cached |

### Add new "agentic_coding" profile

| Parameter | Value | Rationale |
|---|---|---|
| `name` | `"agentic_coding"` | Heavy agent sessions (Claude Code agent, Cursor agent) |
| `arrival_rate_peak` | 20.0 | Fewer but heavier sessions |
| `diurnal_peak_trough_ratio` | 4.0 | Same diurnal pattern |
| `iat_mean_s` | 30.0 | ~30s between agent turns |
| `iat_dist` | `"exponential"` | Standard |
| `input_len_mean_tokens` | 15,000 | Large tool outputs each turn |
| `input_len_sigma_tokens` | 7,500 | High variance |
| `output_len_pareto_alpha` | 1.5 | Long outputs (code generation) |
| `output_len_pareto_xmin` | 200 | Min output |
| `context_growth_min_tokens` | 5,000 | Multi-file edits per turn |
| `context_growth_max_tokens` | 20,000 | Large search/read results |
| `prefix_stability_initial` | 0.95 | Nearly all context is prefix |
| `prefix_stability_final` | 0.85 | High even at end of session |
| `session_duration_mean_s` | 1,800.0 | 30-min agent sessions |
| `session_duration_dist` | `"lognormal"` | Standard |
| `shared_system_prefix_tokens` | 30,000 | Full system prompt + tools + CLAUDE.md |

### Update profile_mix

| Profile | Current | Proposed | Rationale |
|---|---|---|---|
| chat | 0.40 | 0.30 | Reduced share |
| coding | 0.20 | 0.20 | Same share, but heavier per-request |
| batch | 0.25 | 0.20 | Reduced share |
| agent | 0.15 | 0.15 | Same |
| agentic_coding | — | 0.15 | New profile |

### Config file changes

- Update `configs/default.json` with new coding profile values and add agentic_coding
- Keep old values available as `configs/legacy_v1.json` for backward compat

## Phase 2: Re-run analysis with realistic workloads

With 60k-token KV objects (~20GB at 70B FP16), cache tiers will be stressed:
- L1 (80GB) can hold ~4 objects → heavy eviction pressure
- L2 (4TB) still large, but TTL-driven flow to L3A increases
- L3A capacity matters more → global vs local L3A gap should appear

Re-generate:
- `node_scaling.png` — expect visible global vs local L3A differences
- `ttl_sensitivity.png` — TTL impact should be more pronounced
- `sustaining_qps.png` — sustaining QPS should decrease (heavier requests)
- Stressed config may need adjustment for the larger KV objects

## Phase 3: Update tests

- Verify existing 22 tests still pass with updated default config
- Add test for agentic_coding profile

## Phase 4: Documentation + crosscheck + commit

- Update README, CLAUDE.md, user_manual, example_report
- Crosscheck plan vs deliverables
- Commit and push
