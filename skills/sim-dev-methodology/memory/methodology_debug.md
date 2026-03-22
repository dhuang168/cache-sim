---
name: Debug cycle — observe question investigate improve
description: Systematic debugging for unexpected simulation results
type: feedback
---

When results are unexpected: (1) Question — does it match intuition? (2) Instrument — add metrics to expose the mechanism (3) Trace — follow data through the code (4) Isolate — write test that exposes the bug before fixing (5) Fix + validate with data (6) Regression test with extreme conditions.

**Why:** Surface-level results hide bugs. The L3A cross-worker search bug persisted across multiple cycles because tests used short sims where the feature was never exercised.

**How to apply:** Never accept "it works" at face value. If two configs that should differ show identical results — investigate. Add instrumentation before guessing.
