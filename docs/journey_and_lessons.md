# AgentSim Development Journey and Lessons Learned

## Timeline

| Date | Milestone | Key Decision |
|------|-----------|-------------|
| Session 1 | cache-sim prototype | Multi-tier L1/L2/L3A DES with 5 workload profiles |
| Session 2 | Multi-node dispatch | Push/pull dispatchers, worker topology, global vs local L3A |
| Session 3 | Research + planning | Disaggregated P/D research, LMCache research |
| Session 4 | Phase 1: Disaggregated P/D | 3:1 prefill:decode, KV transfer model, 14% TTFT reduction |
| Session 4 | Phase 2: LMCache chunk dedup | 256-token chunks, demand-pull promotion, tail-first eviction |
| Session 4 | Phase 5 study | 18-config comparison, chunk mode fragility discovery |
| Session 5 | vLLM tail-first eviction | Research into production eviction strategies |
| Session 5 | Pre-migration checkpoint | v0.1-prototype tag, baseline capture, import-linter |
| Session 5 | AgentSim Phases 0-4 | Contracts, DES port, observers, NPU configs, framework adapters |
| Session 5 | Research Q1-Q11 | 11 questions answered, 7 investigations, 349× perf fix |

---

## Key Findings

### 1. Dispatch Algorithm > L3A Topology

The choice between push and pull dispatch has **more impact** than global vs local L3A. Pull dispatch achieves 99.8% hit rate on local L3A (eliminating the need for global L3A) because nodes self-select work that matches their cache. Push dispatch only achieves 68.6% local at 4 workers because sessions get committed to random nodes.

**Lesson**: Before building complex infrastructure (global L3A), try smarter scheduling first.

### 2. Object-Mode Caching Beats Chunk Dedup for Long Contexts

LMCache-style 256-token chunk dedup achieves 45-55% hit rate vs 99.9% for per-session objects on heavy coding workloads. The consecutive-chunk lookup breaks when any single chunk is evicted from a 200-400 chunk chain.

vLLM's tail-first eviction improves chunk mode (76.8% vs 70.3% plain LRU) by preserving shared prefix chunks. But object mode remains 3× better because one object covers the entire prefix — no chain to break.

**Lesson**: More granular caching is not always better. For long-context workloads, monolithic objects are more resilient than fine-grained chunks.

### 3. L2 (DDR/DRAM) Value Depends on L1/L2 Capacity Ratio

On A100 (80GB L1): L2 hit rate = 0.0% — objects transit through L2 without being re-accessed. On custom NPU (32GB L1, 256GB DDR): L2 hit rate = 74.2% — most objects reside in DDR and ARE re-accessed.

**Lesson**: A tier's usefulness isn't inherent — it depends on the capacity ratio relative to the working set. Don't assume all tiers are equally valuable.

### 4. 5-Minute TTL Expiry is Not a Problem for Coding Workloads

Zero TTL expiry misses in all simulations. Coding sessions have inter-turn intervals well below 5 minutes (agentic_coding IAT = 30s). Eviction pressure (0.4%) is a larger concern.

**Lesson**: Measure before optimizing. The 5-minute TTL seemed like a potential issue but data shows it's irrelevant for this workload.

### 5. TTFT-Based Cache Inference Has a Fundamental 50%+ Gap

OpenAI protocol (TTFT-inferred) misclassifies 54-68% of cache decisions compared to Anthropic protocol (explicit cache tokens). Even with auto-calibrated baselines, the TTFT distribution for hits and misses overlaps too much for reliable inference.

**Lesson**: Protocol observability matters. Explicit cache signals (Anthropic) are dramatically more accurate than inferred signals (OpenAI TTFT ratio).

---

## Development Methodology Lessons

### Lesson 1: Profile Before Scale

**Mistake**: Launched 20-min sims with a new chunk store that had O(n log n) eviction per insertion. Took 40+ minutes for one config.

**Fix**: A 30-second cProfile run would have shown `sorted()` consuming 50% of runtime. Added methodology rule: profile every new feature at small scale before running long sims. Gate: 2-min sim at target scale must complete in <30s.

**Impact**: Caught the same issue twice (chunk store, then engine session scan).

### Lesson 2: Think Before Running

**Mistake**: Ran 3+ hours of 20-min sims to investigate why global-local L3A divergence didn't appear. The answer was a 30-second grep: the original report used push dispatch, our investigation used pull.

**Fix**: Before launching any sim to investigate an inconsistency, exhaust free checks first: grep configs, diff parameters, reason about the math. Most "inconsistencies" are config differences, not bugs.

### Lesson 3: Reflection Cycle After Experiments

**Mistake**: Would have shipped Q1-Q11 findings without cross-referencing them. Cross-referencing found 6 inconsistencies including a possible emission bug and a model pessimism issue.

**Fix**: After every batch of experiments, systematically cross-reference ALL results. Categorize each inconsistency (bug/insight/artifact). Plan max 7 investigations. Run until clean.

### Lesson 4: Write Scripts to Files Before Running

**Mistake**: Inline bash scripts with long Python code — ephemeral, unreviewable, not reproducible.

**Fix**: Always write experiment scripts as named files (`scripts/research_q1.py`, `scripts/investigation_i1.py`), then run them. User can review and approve in one shot. Scripts committed alongside results.

### Lesson 5: Save Results to Reports Immediately

**Mistake**: Tail-first eviction comparison results were only in terminal output until the user asked.

**Fix**: After any experiment, immediately write findings into a report file. Don't move to the next task until the report reflects all completed experiments.

### Lesson 6: Commit and Push After Every Phase

**Mistake**: Left work uncommitted between phases.

**Fix**: Always commit AND push to main after completing each implementation phase. Clean checkpoints at every boundary.

### Lesson 7: Performance Optimization Must Not Change Simulation Results

The has_session_cached O(1) fix initially produced a 1.3% L1/L3A hit rate delta because the session index didn't track L2 placements. Switching to TierStore._session_refcount (automatically maintained on insert/remove) achieved exact match.

**Lesson**: Performance optimizations must be verified to produce identical results before being accepted. "Close enough" is not acceptable for a simulator.

---

## Architecture Evolution

```
cache-sim v0.1 (prototype)
  └─ 11 modules in sim/
     └─ 105 tests, 10 reports

AgentSim migration (feature/des-core-swap)
  ├─ Phase 0:   contracts.py (43 tests)
  ├─ Phase 0.5: golden baselines (17 tests)
  ├─ Phase 1:   DES port to agentsim/core/des/ (13 tests)
  ├─ Phase 2:   Observation layer (9 tests)
  ├─ Phase 3:   NPU chip profiles (9 tests)
  ├─ Phase 4:   Framework adapters (15 tests)
  ├─ Hardening:  CacheKey, confidence labels
  ├─ Research:   Q1-Q11 (11 scripts, golden files)
  └─ Investigations: I1-I7 (7 scripts, all resolved)

Total: 211 tests, 18 reports, 349× perf improvement
```

---

## What Worked Well

1. **Interface-first port** — defining contracts before porting code prevented "two worldviews" problem
2. **Tolerance-band testing** — golden baselines caught config differences immediately
3. **Import-linter enforcement** — caught a real boundary violation (oracle importing from integration)
4. **Parallel old+new engines** — could verify identical output at every step
5. **v0.1-prototype tag** — permanent rollback point gave confidence to make changes

## What We'd Do Differently

1. **Set dispatch_algorithm explicitly in ALL scripts** — the push/pull default caused the biggest investigation waste
2. **Profile every new feature before any sim >30s** — would have saved hours on chunk store and engine perf
3. **Use smaller sims for investigations** — 2-min sims answer 90% of questions at 1% of the cost
4. **Cross-reference results immediately** — don't wait until all experiments are done to look for inconsistencies
