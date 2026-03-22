# Development Methodology

## Workflow

Every non-trivial task follows this sequence:

1. **Research**: When empirical data is needed, research first. Save findings to `plan_and_progress/research_*.md` before proposing designs.
2. **Plan**: Answer questions and reason through the problem in text. Save plan to `plan_and_progress/<feature>_plan_<N>.md`. Get alignment before coding.
3. **Implement**: Code + tests + documentation in each phase. Tests must use extreme conditions that force the feature to be the sole differentiator.
4. **Verify**: Traffic/workload mix validation. Config consistency check (when changing parameter A, verify all dependent parameters B still valid). Crosscheck plan vs deliverables line-by-line.
5. **Report**: Create NEW report in `docs/` (never overwrite existing reports). Cross-check every claim against actual data.
6. **Ship**: Commit ALL artifacts (code, tests, plots, configs, docs, reports, plans). Push.

## Core Practices

### Observe → Question → Investigate → Improve
When results are unexpected or two configurations that should differ are identical:
1. Question the result (does it match physical intuition?)
2. Add instrumentation (metrics to expose the hidden mechanism)
3. Trace the data path (follow a request through the code)
4. Write an isolation test (expose the bug before fixing)
5. Fix and validate with data
6. Add regression test (extreme conditions that catch reintroduction)

### Validate Hypotheses With Metrics
Never state a hypothesis as fact. Add metrics → measure → conclude. Example: "50ms latency causes feedback loop" → add prefill_duration, slot_utilization, cold_evictions_per_epoch → measure correlation → confirm or refute.

### Config Consistency
When changing any parameter, walk the dependency map:
- Changing workload sizes → check oracle/model covers the range
- Changing arrival rates → check system not in severe overload (>50% drops = meaningless metrics)
- Changing topology → check capacity semantics (per-node vs global)
- Changing sim duration → check steady state reached

### Test Design: Extreme Conditions
For every new feature:
- What metric proves it works?
- What extreme condition makes it the ONLY differentiator?
- What isolation test proves the boundary?
- What end-to-end test proves the value?

### Experiment Design
- Controlled load (below capacity to avoid drops)
- One variable at a time
- Multiple durations (short sims may be misleading)
- Seed sensitivity at capacity edges (run 3+ seeds)
- Always report: hit rate AND completed requests AND drop rate

### Reports
- Lead with key finding, not methodology
- Never overwrite existing reports — create new versioned ones
- Cross-check every claim against actual metric values
- Note caveats (sim duration, seed sensitivity, load regime)

### Documentation
Update with every implementation phase:
- Architecture doc (how it works end-to-end)
- User manual (config reference, recipes, caveats)
- Debug history (every bug and its fix)
- Comparison docs (alignment with production systems)

### Batch Operations
Combine multiple tests/commands into single invocations to minimize friction. Smoke test before long runs — profile a quick run to estimate total time.
