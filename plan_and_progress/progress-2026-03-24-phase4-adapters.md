# Progress: Phase 4 — Framework Adapters

**Date:** 2026-03-24
**Branch:** feature/des-core-swap
**Status:** Complete

## What was done

Three framework config adapters that translate real-world framework configs into AgentSim SimConfig for what-if analysis:

1. **VLLMConfigAdapter** (`integration/adapters/vllm.py`):
   - Maps block_size → block_size_bytes, max_num_seqs → n_prefill_slots
   - prefix_caching → LRU eviction, gpu_memory_utilization → L1 capacity scaling
   - Chunked prefill flag in run_id (metadata only)

2. **LMCacheConfigAdapter** (`integration/adapters/lmcache.py`):
   - Maps local_cpu → L2 tier, local_disk → L3A, remote_url → global L3A
   - chunk_size → block_size_tokens, implicit chunk dedup + demand-pull
   - Produces tier spec documentation

3. **SGLangConfigAdapter** (`integration/adapters/sglang.py`):
   - RadixAttention → chunk dedup + tail_first eviction (preserves shared prefixes)
   - mem_fraction_static → L1 capacity scaling
   - max_running_requests → slot allocation

All three:
- Accept a framework config dict + ModelConfig → produce SimConfig
- Include `mapping_documentation()` with explicit "schema mapping only" caveat
- Run end-to-end through the DES engine (verified by tests)

## Tests
15 tests covering:
- Config conversion correctness (fields mapped correctly)
- Parameter scaling (GPU memory, TP size, memory fraction)
- End-to-end engine execution (adapted configs produce simulation results)
- Mapping documentation exists with caveat

## Gate checklist
- [x] VLLMConfigAdapter converts config without error
- [x] LMCacheConfigAdapter produces correct tier specs (chunk dedup + demand-pull)
- [x] SGLangConfigAdapter maps RadixAttention → tail_first eviction
- [x] All adapted configs run through DES engine
- [x] Schema mapping documented with "does not run framework" caveat
- [x] lint-imports: 3 contracts kept, 0 broken
- [ ] Sub-agent spawning — deferred (engine addition, not adapter)
- [ ] Golden files for Q9/Q10/Q11 — need longer comparison sims

## Next step
Phase 1 hardening (CacheKey threading, confidence labels, MetricsCollector as ObserverBase).
