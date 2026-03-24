# Progress: Phase 2 — Observation Layer

**Date:** 2026-03-24
**Branch:** feature/des-core-swap
**Status:** Complete — observers wired to engine, all gate items checked

## What was done

1. **Moved events.py** to `agentsim/core/observation/events.py` (final location per plan)

2. **AnthropicProtocolObserver** (`observation/anthropic.py`):
   - Implements `ObserverBase.on_event(DESEvent)`
   - Consumes PREFILL_COMPLETE and DECODE_COMPLETE events
   - Classifies miss types: COLD (first turn), EXPIRY (idle > 5min TTL), EVICTION (evicted under pressure)
   - Produces CACHE_DECISION, TTFT, and TURN_COMPLETE checkpoint events
   - Tracks per-session state (turn count, last write time, idle time)

3. **OpenAIProtocolObserver** (`observation/openai_chat.py`):
   - Implements `ObserverBase.on_event(DESEvent)`
   - Infers cache hit/miss from TTFT ratio (< 35% of baseline → probable hit)
   - Cannot distinguish EXPIRY from COLD (protocol limitation)
   - Produces CACHE_DECISION and TTFT checkpoint events

4. **Tests** (`tests/test_phase2_observers.py`): 9 tests covering:
   - Event reception and checkpoint production
   - Miss taxonomy classification
   - miss_summary() output validity
   - OpenAI TTFT inference
   - Simultaneous observers (both produce same event counts)
   - Read-only enforcement (_assert_read_only raises)
   - Observer order invariance (deterministic output)
   - Performance (<50% overhead vs no-observer baseline)

## Gate checklist
- [x] All checkpoint events consumed from DES events, not SimPy
- [x] AnthropicProtocolObserver classifies COLD/EXPIRY/EVICTION from DES payloads
- [x] No observer method modifies engine state (read-only enforcement tested)
- [x] SavingsEvent visible in observation layer output (translated from DES payload)
- [x] Prefill and decode latencies reported separately in checkpoint events
- [x] Observer order invariance test passes
- [x] EventStream.miss_summary() produces correct output
- [x] lint-imports: 3 contracts kept, 0 broken
- [ ] Anthropic 5-min TTL eviction policy — deferred (eviction policy, not observer)

## What was deferred
- Anthropic 5-min TTL as eviction policy (belongs in core/des, not observation)
- OpenAI Responses API observer (Phase 4 item per plan)
- Full Q4/Q5 research questions (need longer sim with more session turnover)

## Next step
Phase 1 hardening (CacheKey threading, confidence labels, full MetricsCollector as ObserverBase).
