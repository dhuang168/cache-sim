# Progress: Disaggregated Continuous Batching (Phase 1)

**Date:** 2026-03-22
**Status:** Complete

## What was done

Implemented disaggregated prefill-decode separation mode for the cache simulator:

1. **FSM changes** (`sim/events.py`): Added `KV_TRANSFERRING` state and `KV_TRANSFER_COMPLETE` event type
2. **Config** (`sim/config.py`): 6 new `ServiceConfig` fields with backward-compatible defaults
3. **Decode node** (`sim/node.py`): Lightweight `DecodeNode` class (slots + queue, no cache)
4. **Oracle** (`sim/oracle.py`): `kv_transfer_time_us()` for RDMA transfer modeling
5. **Metrics** (`sim/metrics.py`): KV transfer time/size, per-decode-node active seqs, decode queue wait
6. **Engine** (`sim/engine.py`): KV transfer handler, modified prefill_complete/decode_start/decode_complete for disaggregated mode, epoch reporting
7. **Config** (`configs/disaggregated.json`): 3:1 prefill:decode reference config
8. **Tests** (`tests/test_disaggregated.py`): 10 new tests, all passing
9. **Docs** (`CLAUDE.md`): Updated with disaggregated mode architecture and config reference

## Results

82/82 tests pass (72 existing + 10 new). Backward compatibility confirmed.

60s comparison (default config, 4 GPUs):
- **Colocated 4-GPU**: TTFT p50=3513ms, p95=8017ms
- **Disaggregated 3P:1D**: TTFT p50=3080ms, p95=7415ms (12% lower p50)
- KV transfer overhead: mean=67.7ms, p95=384.7ms (for ~19GB KV objects over 50 GB/s RDMA)
- Transfer is ~2% of mean prefill time — not the bottleneck

## What was deferred

- `decode_batch_fill_factor` is defined but not yet used in the decode oracle (currently, the batch degradation model uses raw active_sequences)
- Distributed KV cache pool (Mooncake-style cache-aware routing)
- Phase 2: LMCache features (chunk-based dedup, demand-pull tier promotion, LRU eviction)

## Next step

Phase 2: LMCache-compatible features (configurable eviction policy, chunk-level dedup, demand-pull tier migration)
