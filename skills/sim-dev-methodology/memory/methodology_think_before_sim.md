---
name: Think before running — check config and hypotheses before launching sims
description: Before running any simulation to investigate an inconsistency, exhaust free/cheap checks first (read code, diff configs, reason about the math)
type: feedback
---

**Rule: Before launching a simulation to answer a question, exhaust all free checks first.**

1. **Read the code** — check configs, defaults, parameter overrides. Many "inconsistencies" are just different parameters.
2. **Diff the configs** — compare the exact parameters between the two runs. A single default difference (e.g., dispatch_algorithm="push" vs "pull") can explain everything.
3. **Reason about the math** — estimate whether the hypothesis is plausible given the parameter values, before burning compute to verify.
4. **Only then run a sim** — and start with the smallest sim that can distinguish the hypotheses (2 min, not 20 min).

**Why:** The I2 investigation wasted 3+ hours of compute running 20-min sims to find a divergence that didn't exist in the new engine. The root cause (push vs pull dispatch) was discoverable in 30 seconds by `grep dispatch_algorithm scripts/heavy_coding_analysis.py`. The sim was unnecessary — the answer was in the code.

**How to apply:** When an experiment produces unexpected results:
1. First: `grep` for config differences between the two runs (free, instant)
2. Second: check defaults in config.py for any unset parameters (free, instant)
3. Third: estimate whether the hypothesis makes sense given the numbers (free, 2 min thinking)
4. Fourth: run a 2-min sim to verify (cheap, seconds)
5. Last resort: run the full-duration sim (expensive, minutes-hours)

Steps 1-3 would have caught the I2 root cause without any simulation.
