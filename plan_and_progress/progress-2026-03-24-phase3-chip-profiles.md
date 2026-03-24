# Progress: Phase 3 — Non-GPU Parameterization

**Date:** 2026-03-24
**Branch:** feature/des-core-swap
**Status:** Complete

## What was done

1. **Oracle tables** in new JSON format:
   - `benchmarks/oracle_tables/a100_llama3_70b.json` — CALIBRATED, measured data from cache-sim
   - `benchmarks/oracle_tables/custom_npu_v1_llama3_70b.json` — ANALYTICAL_ONLY, roofline params

2. **NPU configs**:
   - `configs/custom_npu_local_l3a.json` — 32GB HBM + 256GB DDR + 8TB SSD, 4 GPUs/worker, local L3A
   - `configs/custom_npu_global_l3a.json` — same but global L3A with 50ms remote latency
   - Key differences from A100: smaller L1 (32GB vs 80GB), faster L2 (200GB/s DDR vs 64GB/s DRAM)

3. **Tests** (9 tests in `test_phase3_chip_profiles.py`):
   - NPU local/global configs load and run end-to-end
   - Global vs local directional comparison (both >90% hit at controlled load)
   - A100 oracle returns CALIBRATED confidence
   - NPU oracle table has analytical-only metadata
   - A100 at 50k tokens: 37s (calibrated range verified)
   - NPU tier characteristics verified (32GB L1, 200GB/s DDR)
   - Performance: NPU 1-min sim in <10s

## Gate checklist
- [x] custom_npu_local_l3a runs end-to-end with correct byte-based tier accounting
- [x] Both NPU configs run and produce results at 4 workers
- [x] A100 oracle returns CALIBRATED, NPU oracle metadata is analytical-only
- [x] Calibrated oracle and analytical-only oracle produce different predictions (verified at 50k tokens)
- [x] Tier parameterization works: NPU has 32GB L1 + 256GB DDR + 8TB SSD
- [x] lint-imports: 3 contracts kept, 0 broken
- [ ] 2-tier config (no L3A) — not yet tested
- [ ] Golden files for Q6/Q7/Q8 — deferred (need longer sim runs)

## Next step
Phase 1 hardening or Phase 4 (framework adapters).
