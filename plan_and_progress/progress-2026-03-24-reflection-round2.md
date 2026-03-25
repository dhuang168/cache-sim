# Reflection Round 2: I2 Divergence Root Cause Found

**Date:** 2026-03-24

## Issue: I2 shows no global-local divergence at 20 min, but original report showed 68.6% local

### Root Cause: Dispatch Algorithm Difference

The original `heavy_coding_analysis.py` uses **push dispatch** (the default — `dispatch_algorithm` is never overridden). Our I2 investigation script uses **pull dispatch**.

**Evidence:**
- `scripts/heavy_coding_analysis.py`: no `dispatch_algorithm` setting → default = `"push"`
- `scripts/investigation_i2.py`: explicitly sets `dispatch_algorithm = "pull"`
- `sim/config.py`: `dispatch_algorithm: str = "push"` (default)

### Why Pull Eliminates the Divergence

| Metric (2 min, 4W local) | Push | Pull |
|--------------------------|------|------|
| Affinity rate | 11.0% | 24.3% |
| L1 hit | 93.1% | 81.4% |
| Miss rate | 0.14% | 0.14% |

Pull achieves **2.2× higher affinity** because nodes self-select work matching their cached sessions. Push assigns at arrival without knowing node cache state. Under overload at 20 min:
- Push: sessions get committed to random nodes → migration → local L3A misses → 68.6%
- Pull: sessions pulled by nodes that have them cached → less migration → 99.8%

### Resolution

This is **not a bug** — it's a real algorithmic difference. The original finding (68.6% local with push) is valid for push dispatch. Pull dispatch genuinely eliminates the global-local gap because better affinity prevents the session migration cascade.

### Updated Understanding

| Dispatch | Global L3A needed? | Why |
|----------|-------------------|-----|
| Push | Yes (critical at 4+ workers) | Sessions migrate → local misses → cascade |
| Pull | No (local sufficient) | Self-selection prevents migration cascade |

This is itself a valuable finding: **the choice of dispatch algorithm determines whether global L3A is needed**, not just the hardware topology.

### Action Items
- [x] Root cause identified (dispatch algorithm difference)
- [x] Verified with 2-min push vs pull comparison
- [ ] Update research_questions_report.md Q1 answer with this finding
- [ ] Consider re-running original report scenarios with push dispatch to reproduce 68.6%
