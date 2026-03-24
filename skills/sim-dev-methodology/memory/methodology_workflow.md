---
name: Development workflow
description: Standard task sequence — research, plan, implement+test, verify, report, ship
type: feedback
---

Every non-trivial task: (1) Research and save findings (2) Plan and save to plan_and_progress/ (3) Implement with extreme-condition tests + docs (4) Verify config consistency + crosscheck plan (5) Create NEW report (never overwrite) (6) **Reflect**: cross-reference results, identify inconsistencies, investigate until clean (7) Commit ALL artifacts + push.

**How to apply:** Follow this sequence for every feature. Skip nothing. Phase 0 (research+plan) prevents wasted work. Crosscheck catches gaps. **Reflection (step 6) catches data inconsistencies that would otherwise ship as conclusions.** Reports preserve state for future reference.
