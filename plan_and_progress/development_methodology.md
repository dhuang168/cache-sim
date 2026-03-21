# Development Methodology

Distilled from feedback across the multi-node cache simulator development. This methodology applies to all future development on this project.

## Core Principle: Observe → Question → Investigate → Improve

Never accept surface-level results. When something looks unexpected:
1. **Observe**: Notice the anomaly (e.g., "only batch generates traffic", "global L3A has lower hit rate")
2. **Question**: Ask why — form a clear hypothesis
3. **Investigate**: Add metrics or run diagnostics to confirm/refute
4. **Improve**: Fix the root cause, validate the fix with data

### Examples from this project
| Observation | Hypothesis | Metric Added | Outcome |
|---|---|---|---|
| Only batch profile generates KV objects | Diurnal rate = 0 at sim start (midnight) | Session count by profile | Fixed: `sim_start_time_s` offset to 10 AM |
| Batch dominates despite 20% mix weight | `profile_mix` doesn't scale arrival rate | Session proportions vs config | Fixed: scale rate by mix weight |
| Global L3A has lower hit rate than local | 50ms penalty → slower prefills → more evictions | `prefill_duration_us`, `slot_utilization_pct`, `cold_evictions_per_epoch` | Confirmed: 94% vs 97% hit rate |

## Standard Development Workflow

### Phase 0: Think First
- Answer questions and reason through the problem before writing code
- Research when empirical data is needed — save findings to `plan_and_progress/research_*.md`
- Propose plan and save to `plan_and_progress/<feature>_plan_<N>.md`

### Phase 1–N: Implementation
Each phase includes:
- Code changes + documentation updates (README, CLAUDE.md, user_manual, examples)
- Smoke test before long runs — profile quick runs to estimate total time
- Traffic mix verification before any simulation analysis
- Performance benchmarks for new features

### Validation Phase
- Validate hypotheses with metrics — don't guess
- Crosscheck plan vs deliverables — line-by-line gap analysis

### Final Phase: Ship
- Regenerate affected plots
- Save progress report to `plan_and_progress/`
- Commit ALL artifacts (code, tests, plots, configs, docs)
- Push to GitHub

## Quality Checklist

- [ ] All tests pass
- [ ] Traffic mix verified (all profiles active, proportions match config)
- [ ] Unexpected results investigated (hypothesis → metrics → validation)
- [ ] Docs updated
- [ ] Plots regenerated
- [ ] Plan crosschecked — no gaps
- [ ] Everything committed and pushed
