# Simulation Development Guide

A methodology for building, validating, and analyzing discrete-event simulators. Distilled from building the PCS multi-node cache simulator.

## 1. Research Before Building

**Never design from assumptions — validate with data first.**

Before implementing a feature, research the real-world system:
- What are the actual parameter values? (e.g., "60K tokens per coding prompt" — validated via Claude Code, Cursor, Copilot data)
- What are the sharing patterns? (e.g., "70-99% of system prompt is shared across users")
- What does the production system actually do? (e.g., "vLLM uses hash-based block matching, not trie")

Save research to `plan_and_progress/research_*.md` before proposing designs. Include sources.

## 2. Plan Before Coding

Every non-trivial task follows:

1. **Phase 0**: Answer questions, reason through the problem, save plan
2. **Phase 1-N**: Implementation + tests + documentation per phase
3. **Crosscheck**: Plan vs deliverables gap analysis
4. **Ship**: Commit all artifacts (code, tests, plots, reports, plans)

Save plans to `plan_and_progress/<feature>_plan_<N>.md`. Never skip Phase 0.

## 3. Test Design: Extreme Conditions

For every new feature, write tests that **force the feature to be the sole differentiator**:

| Principle | Example |
|-----------|---------|
| **Right metric** | Test L3A isolation via `_find_cache_object_with_node`, not just hit rate |
| **Extreme condition** | Tiny L1/L2 (1MB) forces all objects to L3A — exercises L3A logic in 10s |
| **Isolation test** | Place object on worker 0, verify NOT visible from worker 1 |
| **End-to-end test** | Full sim with 2 workers → global > local hit rate |
| **Fast** | Use shortest sim duration that exercises the feature |

The local L3A cross-worker search bug persisted across multiple dev cycles because tests used short sims where L1 absorbed everything — L3A was never exercised.

## 4. Observe → Question → Investigate → Improve

When results look wrong, unexpected, or two configurations that should differ are identical:

### Step 1: Question the result
- Does this match physical intuition?
- Are two things identical that should differ?
- Are magnitudes reasonable?

### Step 2: Add instrumentation
Don't guess — add metrics to expose the hidden mechanism:
- Dispatch stats (affinity vs non-affinity)
- Per-tier object counts and occupancy
- Per-worker breakdowns
- TTFT decomposed by cache hit type

### Step 3: Trace the data path
Follow a request through the code:
- What stores are searched?
- What penalties are applied?
- What's included in the metric vs excluded?

### Step 4: Isolation test
Write a test that directly exposes the bug before fixing it.

### Step 5: Fix and validate
Re-run, verify the expected gap appears, check magnitudes make sense.

### Step 6: Regression test
Add extreme-condition tests that would catch reintroduction.

### Examples from this project

| Observation | Investigation | Root Cause |
|------------|---------------|------------|
| Only batch generates traffic | Session count by profile | Diurnal rate = 0 at midnight |
| Global = local L3A at all durations | Dispatch stats → 97% non-affinity | Local L3A searched ALL workers' SSDs |
| Cache hits slower than cold misses | Oracle values at 65K tokens → 17s (clamped) | Oracle table stopped at 32K tokens |
| Queue wait lower for local despite more misses | Checked drop rates → 88% dropped | Contention model treated global as 1 SSD |
| Report claims contradicted plots | Compared metric values vs plot data | Claims were wrong, plots were correct |

## 5. Config Consistency

When changing any parameter, walk the dependency map:

| When you change... | Also check... |
|---|---|
| Workload token counts | Oracle table covers the range (no clamping) |
| Arrival rates | System not in severe overload (>50% drops = meaningless metrics) |
| Profile mix | Traffic mix verification (proportions match config) |
| Number of workers | L3A capacity semantics, arrival vs throughput |
| Tier capacities | Objects can physically fit (KV size vs tier capacity) |
| Sim duration | Long enough for steady state |
| Diurnal parameters | sim_start_time_s in active period |

**Rule**: If any metric looks suspiciously flat, identical, or inverted — investigate before trusting.

## 6. Experiment Design

### Controlled load
Set arrival rate below system capacity to avoid drops. If >50% dropped, queue metrics have survivorship bias.

### Compare one variable at a time
Hold everything constant except the variable under test. Use the same seed.

### Multiple durations
Short sims (1 min) show transient behavior. Steady state requires 5-20 min for coding workloads (L1/L2 must saturate for migration effects).

### Seed sensitivity
At capacity edges, results are seed-dependent. Run with 3+ seeds to distinguish real effects from stochastic artifacts.

### Metrics to always report
- Hit rate AND completed requests (hit rate alone is misleading if 88% dropped)
- Queue wait AND TTFT (queue wait excludes compute; TTFT includes both)
- Drop rate (the hidden metric — if high, other metrics are unreliable)
- Slot utilization (relates arrival rate to throughput capacity)

## 7. Report Writing

### Structure
1. Lead with the key finding (not the methodology)
2. Show data tables with all relevant metrics
3. Explain WHY the result is what it is
4. Note caveats (sim duration, seed sensitivity, load regime)
5. Include reproduction instructions

### Preserve history
Never overwrite existing reports. Create new versioned reports for each phase (`docs/phase1_*.md`, `docs/experiment_a_*.md`).

### Cross-check claims against data
Before publishing, verify every claim in the report matches the actual numbers. Claims about "queue depth diverges" must be supported by the data showing divergence.

## 8. Documentation Lifecycle

Update docs as part of every implementation phase, not as an afterthought:
- **Architecture doc**: How the system works end-to-end (request lifecycle, assumptions)
- **User manual**: Config reference, recipes, caveats
- **CLAUDE.md**: Debug history (every bug and its fix), test structure, module descriptions
- **vLLM comparison**: Alignment gaps, what's implemented vs what's missing

## Reference: File Organization

```
plan_and_progress/
├── research_*.md          # Research findings (saved before design)
├── *_plan_*.md            # Implementation plans (Phase 0)
├── *_progress_*.md        # Progress reports (after milestones)
├── development_methodology.md  # This guide (project copy)
└── session_summary.md     # Session-level summary

docs/
├── architecture.md        # How the simulation works
├── user_manual.md         # Config reference and recipes
├── vllm_comparison.md     # Alignment with production systems
├── heavy_coding_report.md # Primary analysis report
├── phase*_report.md       # Per-phase sanity reports
└── experiment_*_report.md # Experiment-specific reports
```
