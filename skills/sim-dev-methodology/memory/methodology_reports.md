---
name: Report writing — preserve history, cross-check claims
description: Create versioned reports, never overwrite, verify every claim against data
type: feedback
---

Reports:
- **Lead with finding**, not methodology
- **Never overwrite** existing reports — create new versioned files (phase1_report.md, experiment_a_report.md)
- **Cross-check every claim** against actual metric values before publishing
- **Note caveats**: sim duration, seed sensitivity, load regime, steady-state status
- **Include reproduction instructions**

**Why:** Report claimed "queue depth diverges between global and local" — but the actual data showed they were identical. The claim was wrong, the plots were correct. Always verify claims against data.

**How to apply:** After writing a report, re-read each claim and find the supporting number in the data. If the number doesn't match the claim, fix the claim, not the data.
