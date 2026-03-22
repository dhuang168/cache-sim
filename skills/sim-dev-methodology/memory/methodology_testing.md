---
name: Test design — extreme conditions
description: Write tests that force the feature under test to be the sole differentiator
type: feedback
---

For every new feature: (1) What metric proves it works? (2) What extreme condition makes it the ONLY differentiator? (3) What isolation test proves the boundary? (4) What end-to-end test proves the value? Use the shortest sim duration that exercises the feature.

**Why:** The L3A isolation bug was invisible in short sims because L1 absorbed everything. Extreme conditions (tiny L1, multiple workers) force the feature to be exercised.

**How to apply:** When adding a feature like "local vs global cache", don't just run a normal sim. Make L1/L2 tiny (1MB) so all objects must go through the cache tier under test. Add an isolation test that directly places an object and verifies visibility boundaries.
