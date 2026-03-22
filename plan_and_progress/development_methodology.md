# Development Methodology

**Full guide**: See `docs/simulation_development_guide.md` for the complete methodology with examples.

This file is a quick reference. The guide contains the detailed practices distilled from this project.

## Quick Reference

1. **Research** → `plan_and_progress/research_*.md`
2. **Plan** → `plan_and_progress/*_plan_*.md`
3. **Implement** → code + tests (extreme conditions) + docs
4. **Verify** → traffic mix, config consistency, crosscheck plan vs deliverables
5. **Report** → new `docs/*_report.md` (never overwrite existing)
6. **Ship** → commit ALL artifacts + push

## Core Practices

- **Observe → Question → Investigate → Improve** (never accept surface results)
- **Extreme-condition tests** (force the feature to be the sole differentiator)
- **Config consistency** (when changing A, check what depends on A)
- **Validate hypotheses with metrics** (don't guess — instrument and measure)
- **Preserve reports** (new versioned files, not overwrites)
- **Research before building** (save findings before proposing designs)

## Reference Documents

| Document | Purpose |
|----------|---------|
| `docs/simulation_development_guide.md` | Full methodology with examples |
| `docs/architecture.md` | How the simulation works end-to-end |
| `docs/vllm_comparison.md` | Alignment gaps with vLLM |
| `plan_and_progress/session_summary.md` | What was built and key findings |
| `plan_and_progress/vllm_alignment_master_plan.md` | Phase 0-5 status |
