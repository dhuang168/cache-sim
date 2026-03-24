---
name: Reflection cycle — cross-check results, investigate inconsistencies
description: After answering research questions, systematically cross-reference all findings to identify inconsistencies and unexplained trends before moving on
type: feedback
---

After completing a set of research questions or experiments, always run a **reflection cycle** before starting new work:

1. **Cross-reference** all results against each other and prior findings. Look for:
   - Numbers that contradict each other across experiments
   - Trends that can't be explained by the model
   - Results that are suspiciously clean (e.g., 0.0% difference when a difference is expected)
   - Gaps between observer/engine counts that suggest missing instrumentation

2. **Categorize** each inconsistency:
   - **Triage** (potential simulator bug) — investigate immediately
   - **Design insight** (real behavior, surprising) — document the explanation
   - **Artifact** (test setup issue) — fix the test, re-run

3. **Plan investigations** (max 7 at a time) with:
   - Hypothesis (falsifiable)
   - Experiment design (what to run, what to measure)
   - Script name (consistent naming: `scripts/investigation_i[N].py`)

4. **Run investigations**, update the report, then **reflect again** on the new results.

5. **Repeat** until no unexplained inconsistencies remain.

**Why:** The Q1-Q11 analysis found 6 inconsistencies that would have been silently accepted without cross-referencing. Issue 3 (observer 6:1 count gap) may be an emission bug. Issue 6 (chunk mode too pessimistic) affects all LMCache/SGLang conclusions. These would have been shipped as findings without the reflection step.

**How to apply:** After every batch of experiments:
- Write `plan_and_progress/plan-next-investigations.md` with issues + hypotheses
- Run investigation scripts
- Update the report with corrected findings
- Reflect again until clean
