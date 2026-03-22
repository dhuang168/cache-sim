# Session Summary: Multi-Node Cache Simulator Development

Date: 2026-03-21 to 2026-03-22

## What Was Built

Starting from a single-node three-tier cache simulator, we built a production-realistic multi-node simulation platform for evaluating KV cache strategies for LLM inference.

### Architecture Changes
- **Multi-node prefill dispatch** with push, pull, and smart push algorithms
- **Worker topology**: 8 GPUs/worker sharing DRAM (L2) and SSD (L3A), configurable
- **Global vs local L3A**: pooled SSD (N × 8TB) vs per-worker SSD (8TB each)
- **Per-SSD bandwidth contention**: concurrent reads share SSD bandwidth, distributed across N workers' SSDs
- **First-access warming**: L3A reads populate local L1, no repeated SSD access

### Cache Features (vLLM Alignment Phases 0-3)
- **Phase 0**: 8 config consistency regression tests (35→ tests)
- **Phase 1**: Configurable token block sizes (16/256/4096) with boundary rounding
- **Phase 2**: Three-tier cross-session sharing (framework/workspace/session) with ref counting
- **Phase 3**: LRU reactive eviction policy (alternative to TTL with access refresh)

### Workload
- 5 profiles: chat, coding, batch, agent, agentic_coding
- Coding profiles research-validated: 40-80K tokens, 80-95% cacheable
- Profile mix scaled by arrival rate weights
- Diurnal model with configurable start time

### Oracle
- Extended to 262K tokens with O(n²) attention scaling
- Correct cold miss costs: 37s (50K tokens) to 120s (100K tokens)

## Key Findings

| Finding | Evidence |
|---------|----------|
| **Global L3A essential at 4+ workers** | 99.8% hit, 60% more throughput vs local (78% hit) at 20 min |
| **2 workers: local wins** | Per-SSD contention on 2 SSDs too high for global |
| **Session migration is the mechanism** | 97% non-affinity dispatch after L1/L2 saturate (~5 min) |
| **Pull dispatch preferred** | +0.5% throughput, -8% queue wait, zero drops at 5 min |
| **Block granularity negligible for coding** | 16 vs 4096 tokens: <5% boundary loss on 50K contexts |
| **Sharing saves 127TB but increases L1 evictions** | Pinned shared objects consume L1 capacity |
| **TTL ≈ LRU at controlled load** | Access-refresh in TTL mode converges with LRU behavior |
| **Capacity cliff at peak=4-5** | Seed-dependent collapse at system edge |
| **50ms L3A remote latency is negligible** | <0.3% overhead on 18.9s mean prefill |

## Bugs Found and Fixed

| Bug | Root Cause | How Found |
|-----|-----------|-----------|
| Only batch generated traffic | Diurnal rate = 0 at midnight | Traffic mix verification |
| Profile mix didn't scale rates | profile_mix not applied to arrival rate | Observed proportions vs config |
| L2 saturation >100% | evict_l1_to_l2 skipped can_fit() | Observed impossible metric |
| Local L3A searched all workers | _find_cache_object searched globally | Questioned identical global/local results |
| Oracle clamped at 32K tokens | Table too short for v2 workloads | Cache hits appeared slower than cold misses |
| Global L3A contention modeled as 1 SSD | Should be N SSDs distributed | Questioned 88% drop rate |
| Report claims contradicted plots | Assumed divergence that data showed didn't exist | Cross-checked claims vs actual metrics |

## Test Suite

72 tests in 6 seconds:
- 4 invariant tests
- 4 KV size tests
- 5 oracle tests
- 14 multinode tests
- 8 config consistency tests
- 21 block allocation tests
- 10 sharing tests
- 6 eviction policy tests

## Artifacts

### Configs
- `configs/default.json` — v2 realistic hardware (80GB L1, 1TB L2, 8TB L3A)
- `configs/heavy_coding.json` — 90% coding workload
- `configs/legacy_v1.json` — original v1 profiles

### Reports (preserved, not overwritten)
- `docs/heavy_coding_report.md` — 20-min heavy coding analysis
- `docs/phase0_test_report.md` — test hardening
- `docs/phase1_block_allocation_report.md` — block size analysis
- `docs/phase2_block_sharing_report.md` — sharing + contention
- `docs/phase2_full_analysis_report.md` — comprehensive 6-dimension analysis
- `docs/phase3_reactive_eviction_report.md` — LRU vs TTL
- `docs/experiment_a_block_size_report.md` — block size sweep
- `docs/experiment_b_sharing_l3a_report.md` — sharing × L3A mode
- `docs/experiment_c_push_pull_report.md` — dispatch comparison
- `docs/architecture.md` — end-to-end simulation flow
- `docs/vllm_comparison.md` — alignment with vLLM

### Research
- `plan_and_progress/research_coding_workload_tokens.md`
- `plan_and_progress/research_vllm_prefix_caching.md`
- `plan_and_progress/research_cross_session_sharing.md`
