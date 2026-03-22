# vLLM Alignment — Master Plan

Date: 2026-03-22

## Phases

### Phase 0: Test Hardening and Speed ✅ COMPLETE
- 8 config consistency regression tests covering all 12 known bugs
- 35 tests in 4.66s

### Phase 1: Block-Level Allocation ✅ COMPLETE
- Configurable `block_size_tokens`: 0=legacy, 16=vLLM, 256=OpenAI, 4096=page
- Block boundary rounding: only full blocks count as cached
- Finding: negligible impact on coding workloads (5% loss at 4K blocks vs 0.025% at 16-token)
- Anthropic message-style deferred to future phase
- 21 new tests, 56 total

### Phase 2: Cross-Session Block Sharing ✅ COMPLETE
- Three-tier sharing model: framework (56K), workspace (15K), session (unique)
- Token counts and group sizes fully configurable
- Reference counting on shared objects
- Cross-worker duplication tracking (probabilistic, not assumed)
- Global L3A bandwidth contention model (concurrent reads share bandwidth)
- 10 new tests, 66 total in 5.97s

### Phase 3: Reactive Eviction ✅ COMPLETE
- `eviction_policy="lru"` config option (default "ttl" for backward compat)
- LRU mode: no TTL_FIRE events, objects stay until pressure eviction
- Objects linger after session ends — future sessions reclaim them
- Combines with Phase 2 sharing: hot shared prefixes never evicted under LRU
- 6 new tests, 72 total in 6.09s

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
