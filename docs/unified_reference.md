# AgentSim Unified Reference — All Findings Cross-Referenced

This document consolidates findings from all reports, research questions, and investigations into a single reference. Every claim is traceable to a specific script and golden file.

---

## 1. Cache Hit Rates by Configuration

| Config | Dispatch | L3A Mode | Hit Rate | Source |
|--------|----------|----------|----------|--------|
| 1W 8GPU, peak=100, 5min | pull | global | 99.8% | `scripts/research_q1.py` |
| 4W 32GPU, peak=100, 5min | pull | global | 99.8% | `scripts/research_q1.py` |
| 4W 32GPU, peak=100, 5min | pull | local | 99.8% | `scripts/research_q1.py` |
| 4W 32GPU, peak=100, 20min | push | global | 99.8% | `scripts/heavy_coding_analysis.py` |
| 4W 32GPU, peak=100, 20min | push | local | 68.6% | `scripts/heavy_coding_analysis.py` |
| 4W 32GPU, peak=15, 5min | pull | global | 99.9% | `results/golden/phase05_reference.json` |
| 8W 64GPU, peak=100, 5min | pull | global | 99.8% | Scenario D baseline |
| NPU 4W, peak=15, 5min | pull | local | 99.98% | `scripts/research_q8.py` |
| NPU 4W, peak=15, 5min | pull | global | 99.98% | `scripts/research_q8.py` |

**Key insight**: The 68.6% local outlier is ONLY with push dispatch at 20 min. Pull dispatch maintains 99.8% at all scales and durations tested.

---

## 2. SavingsEvent Distribution (1W, peak=100, 5min)

| Event | Count | Fraction | Script |
|-------|-------|----------|--------|
| CACHE_HIT_L1 | 17,477 | 26.8% | `scripts/research_q2.py` |
| CACHE_HIT_L2_WORTHWHILE | 9 | 0.0% | |
| CACHE_HIT_L3A_WORTHWHILE | 38,972 | 59.7% | |
| CACHE_HIT_L3A_BREAK_EVEN | 8,666 | 13.3% | |
| COLD_MISS | 129 | 0.2% | |

Golden file: `results/phase1-baseline/q2_savings_breakdown.json`

---

## 3. Disaggregated P/D Impact

| Cluster | Legacy TTFT p50 | Disagg TTFT p50 | Improvement | Source |
|---------|----------------|-----------------|-------------|--------|
| 4W global | 11.0s | 9.8s | −11% | `docs/phase5_disagg_lmcache_study.md` |
| 8W global | 10.5s | 9.0s | −14% | |
| 12W global | 10.4s | 8.9s | −14% | |
| Default (4GPU) | 6.7s | 6.3s | −6.5% | `docs/phase4_disaggregated_pd_report.md` |
| Stressed (4GPU) | 8.9s | 8.1s | −8.9% | |

KV transfer overhead: mean 68-113ms (<2% of TTFT at 50 GB/s RDMA).

---

## 4. Chunk Dedup vs Object Mode

| Mode | Hit Rate | Recompute | TTFT p50 | Source |
|------|----------|-----------|----------|--------|
| Object (per-session) | 99.7% | 23.8% | 11.0s | Phase 5 study |
| Chunk LRU (LMCache) | 70.3% | 73.0% | 47.5s | Phase 5 study |
| Chunk tail-first (vLLM) | 76.8% | 54.4% | 20.6s | Phase 5 Section 7 |
| CacheBlend (estimated) | ~25.6% | 78.2% | — | `scripts/investigation_i6.py` |

Golden files: `results/disagg_lmcache_study.json`

---

## 5. Dispatch Algorithm Comparison

| Algorithm | Affinity Rate | L1 Hit | TTFT p50 | Source |
|-----------|-------------|--------|----------|--------|
| Push (4W local) | 11.0% | 93.1% | — | `scripts/investigation_i5.py` |
| Pull (4W local) | 24.3% | 81.4% | — | `scripts/investigation_i5.py` |
| Push (4W global, 20min) | — | — | ~25s | `docs/experiment_c_push_pull_report.md` |
| Pull (4W global, 20min) | 46% | — | ~25s | `docs/experiment_c_push_pull_report.md` |

**Pull achieves 2× higher affinity than push**, which eliminates the need for global L3A at controlled load.

---

## 6. Worker Topology Impact

| GPUs/Worker | Workers | Affinity | L1 Hit | TTFT p50 | Source |
|------------|---------|----------|--------|----------|--------|
| 4 | 8 | 10.3% | 11.8% | 7.5s | `scripts/investigation_i5.py` |
| 8 | 4 | 58.4% | 61.8% | 4.5s | `scripts/investigation_i5.py` |

8 GPUs/worker provides 5.7× better affinity due to larger shared L2 pool per worker.

---

## 7. NPU vs A100

| Chip | Confidence | L1 | L2 Hit | TTFT p50 | Source |
|------|-----------|-----|--------|----------|--------|
| A100 (80GB HBM) | CALIBRATED | 80GB | 0.0% | 22.3s | `scripts/research_q6.py` |
| NPU (32GB HBM) | ANALYTICAL-ONLY | 32GB | 74.2% | 78.5s | `scripts/research_q6.py`, `research_q7.py` |

NPU 3.5× slower TTFT. But DDR (L2) is highly utilized on NPU — 74.2% hit rate, 100% saturation. More DDR capacity would improve TTFT.

---

## 8. Protocol Observability

| Protocol | Detection Method | Accuracy vs Ground Truth | Source |
|----------|-----------------|-------------------------|--------|
| Anthropic | Explicit cache tokens | Ground truth | `scripts/research_q4.py` |
| OpenAI (50s baseline) | TTFT < 35% of baseline | 32.3% (67.7% gap) | `scripts/research_q5.py` |
| OpenAI (84s auto-calibrated) | TTFT < 35% of p95 cold | 46.2% (53.8% gap) | `scripts/investigation_i4.py` |

---

## 9. TTL and Eviction

| Metric | Value | Source |
|--------|-------|--------|
| TTL expiry misses | 0 (0.0%) | `scripts/research_q3.py` |
| Cold misses | 21 (0.2%) | |
| Eviction misses | 39 (0.4%) | |
| L2 hit (A100, TTL) | 0.0% | `scripts/investigation_i1.py` |
| L2 hit (A100, LRU) | 0.0% | `scripts/investigation_i1.py` |
| L2 hit (NPU) | 74.2% | `scripts/research_q7.py` |

TTL vs LRU makes no difference on A100 — L2 is unused regardless of eviction policy.

---

## 10. Performance Benchmarks

| Optimization | Before | After | Speedup | Source |
|-------------|--------|-------|---------|--------|
| Chunk store eviction | O(n log n) sort | OrderedDict LRU | 300× | `docs/phase5_disagg_lmcache_study.md` |
| has_session_cached | O(objects) scan | O(1) refcount | 4× | `scripts/investigation_perf_fix.py` |
| pull() queue scan | O(queue_size) | O(64) capped | 3× | Commit `27a1fd9` |
| evict_l1_to_l2 sort | O(n log n) sorted | heapq.nsmallest | 2× | Commit `27a1fd9` |
| **Combined** | **2,668s** (5min sim) | **7.6s** | **349×** | |

---

## 11. L3A Saturation Timeline (4W, peak=100, pull dispatch)

| Duration | Global L3A Sat | Local L3A Sat | Hit Delta | Source |
|----------|---------------|--------------|-----------|--------|
| 2 min | 98.4% | 79.7% | 0.0 pt | `scripts/investigation_i2.py` |
| 5 min | 99.3% | 91.7% | 0.0 pt | |
| 10 min | 99.6% | 95.7% | 0.0 pt | |
| 15 min | 99.7% | 97.0% | 0.0 pt | |
| 20 min | 99.8% | 97.7% | 0.0 pt | |

No divergence with pull dispatch at any duration. L3A saturates by 5 min globally but sessions don't migrate enough to cause local misses when dispatch has good affinity.

---

## Report Index

| Report | Focus | Location |
|--------|-------|----------|
| Heavy Coding Report | Original: global vs local L3A, 20min push dispatch | `docs/heavy_coding_report.md` |
| Phase 0 Test Report | Invariant tests, boundary conditions | `docs/phase0_test_report.md` |
| Phase 1 Block Allocation | Token block sizes, fragmentation | `docs/phase1_block_allocation_report.md` |
| Phase 2 Block Sharing | Cross-session prefix sharing | `docs/phase2_block_sharing_report.md` |
| Phase 3 Reactive Eviction | LRU vs TTL eviction policies | `docs/phase3_reactive_eviction_report.md` |
| Experiment A: Block Size | 16/256/4096 token blocks | `docs/experiment_a_block_size_report.md` |
| Experiment B: Sharing + L3A | Three-tier sharing + global L3A | `docs/experiment_b_sharing_l3a_report.md` |
| Experiment C: Push vs Pull | Dispatch algorithm comparison | `docs/experiment_c_push_pull_report.md` |
| Phase 2 Full Analysis | Multi-node + sharing + LRU combined | `docs/phase2_full_analysis_report.md` |
| Phase 4: Disaggregated P/D | Prefill-decode separation, 14% TTFT reduction | `docs/phase4_disaggregated_pd_report.md` |
| Phase 5: Disagg + LMCache | 18-config comparison, chunk fragility | `docs/phase5_disagg_lmcache_study.md` |
| Research Questions Q1-Q11 | 11 answered questions, 4 phases | `docs/research_questions_report.md` |
| Investigation Round 1 | 7 inconsistencies resolved, perf root cause | `docs/investigation_report_round1.md` |
| vLLM Comparison | Block hashing, prefix caching alignment | `docs/vllm_comparison.md` |
| Journey & Lessons | Development methodology, 7 key lessons | `docs/journey_and_lessons.md` |

---

## Reproducibility Scripts

All scripts produce deterministic output (seed=42) and save results to JSON golden files.

```bash
# Research questions
.venv/bin/python scripts/research_q1.py   # Q1: node scaling
.venv/bin/python scripts/research_q2.py   # Q2: savings breakdown
.venv/bin/python scripts/research_q3.py   # Q3: TTL expiry
.venv/bin/python scripts/research_q4.py   # Q4: Anthropic taxonomy
.venv/bin/python scripts/research_q5.py   # Q5: observability gap
.venv/bin/python scripts/research_q6.py   # Q6: NPU vs A100
.venv/bin/python scripts/research_q7.py   # Q7: DDR break-even
.venv/bin/python scripts/research_q8.py   # Q8: NPU global L3A
.venv/bin/python scripts/research_q9.py   # Q9: LMCache offload
.venv/bin/python scripts/research_q10.py  # Q10: Sarathi vs Orca
.venv/bin/python scripts/research_q11.py  # Q11: SGLang vs vLLM

# Investigations
.venv/bin/python scripts/investigation_i1.py  # I1: L2 TTL vs LRU
.venv/bin/python scripts/investigation_i2.py  # I2: saturation timeline
.venv/bin/python scripts/investigation_i3.py  # I3: emission coverage
.venv/bin/python scripts/investigation_i4.py  # I4: OpenAI baseline
.venv/bin/python scripts/investigation_i5.py  # I5: dispatch affinity
.venv/bin/python scripts/investigation_i6.py  # I6: CacheBlend estimate
.venv/bin/python scripts/investigation_i7.py  # I7: disagg + stressed

# Comparison studies
.venv/bin/python scripts/disagg_lmcache_study.py  # 18-config study
.venv/bin/python scripts/capture_baseline_metrics.py  # baseline capture
```
