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
| Only batch profile generates KV objects | Diurnal rate = 0 at sim start | Session count by profile | Fixed: `sim_start_time_s` offset to 10 AM |
| Batch dominates despite 20% mix weight | `profile_mix` doesn't scale arrival rate | Session proportions vs config | Fixed: scale rate by mix weight |
| Global = local L3A at all durations | Affinity keeps sessions on worker | Dispatch stats (affinity/non-affinity) | Wrong: 97% non-affinity dispatch found |
| Global = local despite 97% migration | ??? | Code trace of `_find_cache_object_with_node` | **Bug: local L3A searched ALL workers' SSDs** |
| L2 saturation 154% | L2 can't exceed capacity | Code trace of `evict_l1_to_l2()` | Bug: no `can_fit()` check before L2 insert |

## Standard Development Workflow

### Phase 0: Think First
- Answer questions and reason through the problem before writing code
- Research when empirical data is needed — save findings to `plan_and_progress/research_*.md`
- Propose plan and save to `plan_and_progress/<feature>_plan_<N>.md`

### Phase 1–N: Implementation
Each phase includes:
- **Code changes** + **documentation updates** (README, CLAUDE.md, user_manual, examples)
- **Feature tests with extreme conditions** (see Test Design below)
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

## Test Design: Extreme Conditions

For every new feature, write tests that:
1. **Use the right metrics** to detect the feature's impact
2. **Use extreme conditions** that force the feature to be the sole differentiator
3. **Be fast** — shortest possible sim that exercises the feature
4. **Include isolation tests** — directly verify boundaries (e.g., object on worker 0 NOT visible from worker 1)
5. **Include end-to-end tests** — full sim with expected metric difference

Example: The local L3A cross-worker bug was caught because:
- `test_local_l3a_worker_isolation` directly verifies the search boundary
- `test_session_migration_global_advantage` uses tiny L1/L2 + 2 workers to force migration in 10s
- `test_global_l3a_cross_worker_access` verifies global L3A is visible from any node

## Debug Skill: Systematic Simulation Debugging

When results are unexpected or two configurations are unexpectedly identical:

1. **Question the result** — does it match physical intuition?
2. **Add instrumentation** — dispatch stats, per-tier counts, per-worker breakdowns, occupancy over time
3. **Trace the data path** — follow a request through the code, check what stores are searched
4. **Write isolation tests** — before fixing, write a test that exposes the bug
5. **Fix and validate** — re-run, verify expected gap appears, check magnitudes
6. **Add regression tests** — extreme-condition tests that would catch reintroduction

Key lesson: the L3A bug persisted because tests used short sims where L1 absorbed everything. L3A was never exercised, so the cross-worker search bug was invisible.

## Quality Checklist

- [ ] All tests pass
- [ ] New features have extreme-condition regression tests
- [ ] Traffic mix verified (all profiles active, proportions match config)
- [ ] Unexpected results investigated (hypothesis → metrics → validation)
- [ ] Docs updated
- [ ] Plots regenerated
- [ ] Plan crosschecked — no gaps
- [ ] Everything committed and pushed
