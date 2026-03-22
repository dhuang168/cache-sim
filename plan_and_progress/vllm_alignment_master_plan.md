# vLLM Alignment — Master Plan

Date: 2026-03-22

## Phases

### Phase 0: Test Hardening and Speed
Enhance unit tests to cover all bugs found during development. Improve test speed so the full suite runs in <10s.

Goals:
- Add tests for every bug in CLAUDE.md debug history that doesn't have a regression test
- Add config consistency validation tests (oracle range covers workload, arrival rate vs throughput, etc.)
- Optimize slow tests (reduce sim duration where possible)
- Target: full suite <10s, all known bugs covered

### Phase 1: Block-Level Allocation (High Priority)
Replace whole-context `CacheObject` with block chains. Support configurable block sizes:
- **vLLM style**: 16-token blocks (GPU-optimized, fine granularity)
- **OpenAI style**: 256-token prefix blocks (coarser, API-level caching)
- **Anthropic style**: message-boundary breakpoints (static cache at message boundaries)
- **Non-GPU approach**: 4K page-aligned blocks (OS page size, for CPU/SSD tiers)

This enables:
- Partial prefix sharing (first N blocks match, rest diverge)
- Block-level eviction (evict tail blocks, keep shared prefix)
- More accurate fragmentation modeling

### Phase 2: Cross-Session Block Sharing (High Priority)
Replace per-session KV objects with shared physical blocks using reference counting.
- Identical token blocks across sessions share one physical copy
- Ref counting per block (not per object)
- Implicit deduplication of system prompts
- Memory savings proportional to shared prefix length × concurrent sessions

### Phase 3: Reactive Eviction (Medium Priority)
Replace TTL-driven tier migration with reactive LRU eviction.
- Evict only when space is needed (not proactively by timer)
- LRU among ref_cnt==0 blocks
- Blocks linger after request completes (future requests can reclaim)
- Tie-break: preserve shared prefix roots

### Phase 4: Preemption Model (Medium Priority)
Allow scheduler to preempt running requests to free KV blocks.
- Preempted requests' blocks freed and request requeued
- Recompute mode (default): discard KV, recompute on reschedule
- Swap mode: offload KV to CPU/SSD before freeing GPU blocks
- Priority-based scheduling

### Phase 5: Batched Prefill (Low Priority)
Model GPU utilization from batching multiple prefills together.
- Chunked prefill: split long prefills into chunks interleaved with decode
- Batch throughput model replaces single-sequence oracle
- GPU utilization metric

## No Change
- **Global L3A**: Keep as-is. Forward-looking feature not in vLLM.

## Development Methodology
Each phase follows:
1. Save plan to `plan_and_progress/`
2. Research if needed, save findings
3. Implementation with extreme-condition tests
4. Traffic mix + config consistency verification
5. Create NEW sanity report in `docs/` (e.g., `docs/phase1_sanity_report.md`) — never overwrite existing reports
6. Documentation updates (architecture, user manual — existing reports preserved as historical record)
7. Crosscheck plan vs deliverables
8. Commit all artifacts + push
