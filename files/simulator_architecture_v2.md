# Agentic LLM Simulator — Revised Architecture

## Core Concept in One Sentence

The SimPy event loop drives agentic sessions. On each turn it calls
into a Framework Adapter that runs the real scheduling algorithm
in-process. The scheduler decides batching/KV cache. A pluggable
Hardware Backend answers "how long does this batch take?" — either
analytically (no GPU needed) or via a real device.

---

## High-Level Flow

```
                    ┌─────────────────────────────────┐
                    │       SimPy Event Loop           │
                    │  (session lifecycle only)        │
                    │                                  │
                    │  t=0:   Session A arrives        │
                    │  t=0:   Session B arrives        │
                    │  t=0:   Fire Turn A.1            │
                    │  t=δ1:  Fire Turn B.1            │
                    │  t=T1:  Turn A.1 complete        │
                    │         → sample think_time      │
                    │  t=T1+θ: Fire Turn A.2           │
                    │  ...                             │
                    └──────────────┬──────────────────┘
                                   │  fire_turn(request)
                                   │  ← returns latency (ms)
                                   ▼
                    ┌─────────────────────────────────┐
                    │      Framework Adapter           │
                    │  (thin in-process wrapper)       │
                    │                                  │
                    │  submit(request)                 │
                    │    → scheduler.step()            │
                    │    → kv_cache.alloc()            │
                    │    → hw_backend.execute(batch)   │
                    │    → kv_cache.update()           │
                    │    ← return (ttft_ms, total_ms,  │
                    │              cache_hit, tokens)  │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
   ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐
   │  VLLMAdapter    │  │  SGLangAdapter  │  │  NativeAdapter     │
   │                 │  │                 │  │  (your own         │
   │  Imports vLLM   │  │  Imports        │  │   scheduler for    │
   │  scheduler +    │  │  SGLang         │  │   non-GPU target)  │
   │  block manager  │  │  scheduler +    │  │                    │
   │  as a library   │  │  RadixAttn      │  │                    │
   │  (no HTTP)      │  │  (no HTTP)      │  │                    │
   └────────┬────────┘  └────────┬────────┘  └────────┬───────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │  execute(batch_plan)
                                 ▼
                    ┌─────────────────────────────────┐
                    │      Hardware Backend            │
                    │  (pluggable, swap freely)        │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼──────────────────┐
              ▼                ▼                  ▼
   ┌──────────────────┐ ┌───────────────┐ ┌────────────────────┐
   │ AnalyticalNPU    │ │ AnalyticalGPU │ │ RealGPU            │
   │                  │ │               │ │                    │
   │ HBM bw model     │ │ A100/H100     │ │ Calls real         │
   │ DDR bw model     │ │ roofline      │ │ torch forward()    │
   │ 4K page size     │ │ model         │ │ on actual device   │
   │ No GPU needed    │ │               │ │                    │
   │                  │ │               │ │                    │
   │ predict_lat()    │ │ predict_lat() │ │ measure_lat()      │
   │  = pure math     │ │  = pure math  │ │  = wall clock      │
   └──────────────────┘ └───────────────┘ └────────────────────┘
```

---

## SimPy Session Lifecycle (Request-Level, No Token-by-Token)

```python
class AgenticSession:
    """
    One Claude Code-style agentic session.
    Each turn fires as a single atomic request.
    Turn N+1 is unlocked only after turn N completes
    + a sampled agent think time.
    """

    def run(self, env, adapter, metrics):
        for turn_idx, turn in enumerate(self.turns):

            # 1. Build the request for this turn
            #    Input = system_prompt + file_contents + history
            #    These grow each turn → key KV cache signal
            request = Request(
                session_id   = self.session_id,
                turn_id      = turn_idx,
                input_tokens = turn.input_tokens,   # grows each turn
                output_tokens= turn.expected_output_tokens,
                prefix_len   = turn.cached_prefix_tokens,  # what's in KV
            )

            # 2. Fire the turn — blocks until full response
            t_start = env.now
            result  = yield env.process(adapter.submit(request))
            t_end   = env.now

            # 3. Record metrics
            metrics.record(
                session_id = self.session_id,
                turn_id    = turn_idx,
                ttft_ms    = result.ttft_ms,
                total_ms   = t_end - t_start,
                cache_hit  = result.cache_hit,
                tokens_out = result.tokens_generated,
            )

            # 4. Sample agent think time → wait → fire next turn
            #    Distribution: LogNormal(mean=2s, sigma=1s)
            #    Models: agent parsing response, deciding next tool call
            if turn_idx < len(self.turns) - 1:
                think_time = self.think_time_dist.sample()
                yield env.timeout(think_time)

            # 5. If this turn spawned sub-agents, run them concurrently
            for sub in self.sub_agents_spawned_at[turn_idx]:
                env.process(AgenticSession(sub).run(env, adapter, metrics))
```

---

## Framework Adapter — What "Algorithmic Part Only" Means

```python
class VLLMAdapter:
    """
    Imports vLLM's scheduler + block manager as a Python library.
    Does NOT start an HTTP server.
    Does NOT do token-by-token decoding.

    vLLM components used in-process:
      - Scheduler         → decides which requests run in this iteration
      - BlockSpaceManager → allocates / frees KV cache pages
      - (optionally) LMCacheConnector → prefix cache lookup

    vLLM components NOT used:
      - AsyncLLMEngine    (the HTTP serving layer)
      - ModelRunner       (actual GPU forward pass)
      - Tokenizer         (we work in token counts already)

    The Hardware Backend replaces ModelRunner entirely.
    """

    def __init__(self, hw_backend, kv_config):
        from vllm.core.scheduler import Scheduler
        from vllm.core.block_manager import BlockSpaceManager

        self.scheduler     = Scheduler(config=kv_config)
        self.block_manager = BlockSpaceManager(
            block_size     = kv_config.block_size,   # your 4096
            num_gpu_blocks = kv_config.hbm_blocks,
            num_cpu_blocks = kv_config.ddr_blocks,
        )
        self.hw_backend = hw_backend

    def submit(self, request: Request) -> SimPy.Process:
        """
        1. Add request to scheduler queue
        2. Run scheduler.step() → get batch plan
        3. Allocate KV cache pages for batch
        4. Call hw_backend.execute(batch) → get latency
        5. Free pages for completed requests
        6. Return result with latency + cache stats
        """
        self.scheduler.add_request(request)

        # Scheduler decides who runs together (real vLLM logic)
        batch_plan = self.scheduler.schedule()

        # KV cache allocation (real vLLM block manager logic)
        for req in batch_plan.requests:
            self.block_manager.allocate(req)

        # Hardware backend computes latency analytically or measures it
        latency = self.hw_backend.execute(batch_plan)

        # Free completed requests
        for req in batch_plan.completed:
            self.block_manager.free(req)

        return Result(
            ttft_ms       = latency.ttft,
            total_ms      = latency.total,
            cache_hit     = batch_plan.prefix_cache_hit,
            tokens_generated = request.output_tokens,
        )
```

---

## Hardware Backend — The Analytical Non-GPU Model

```
For a given batch_plan the backend answers one question:
"How long does this batch take on this hardware?"

Inputs:
  - num_prefill_tokens   (new tokens to process)
  - num_cached_tokens    (KV already in HBM — skip prefill)
  - num_decode_steps     (output tokens to generate)
  - batch_size           (concurrent requests)
  - memory_state         (HBM used %, DDR used %)

Latency model (roofline, no simulation loop):

  prefill_lat = prefill_tokens × bytes_per_token / HBM_bandwidth
                + prefill_tokens² × flops_per_token / TOPS
                (take max → memory or compute bound)

  kv_load_lat = 0                    if hit in HBM
              = pages × page_bytes / HBM_bandwidth   if hit in DDR
              = pages × page_bytes / DDR_bandwidth   if cold miss

  decode_lat  = output_tokens × (
                    bytes_per_token × batch_size / HBM_bandwidth
                )   ← decode is almost always memory-bound

  total_lat   = prefill_lat + kv_load_lat + decode_lat

No loops. One function call per turn. Fast enough to run 100K
sessions in seconds.
```

---

## What You Do NOT Simulate (And Why That's Fine)

```
NOT simulated                    Why it's fine
──────────────────────────────────────────────────────
Individual token steps           Token latency = total / output_tokens
                                 You care about TTFT and E2E, not each step

vLLM's internal CUDA kernels     hw_backend.execute() analytically
                                 models the same roofline behavior

HTTP/gRPC overhead               You're calling in-process; no network layer

Tokenization                     Workload generator produces token counts
                                 directly from trace distributions

Speculative decoding             Can add as a latency multiplier if needed
```

---

## Event Types in SimPy (Complete List)

```
Event                    Trigger                   Action
────────────────────────────────────────────────────────────────
SESSION_ARRIVE           Poisson / trace replay    Start session process
TURN_START               Previous turn complete    Submit request to adapter
                         + think_time sampled
TURN_COMPLETE            adapter.submit() returns  Record metrics, sample
                                                   think_time, schedule
                                                   TURN_START or SESSION_END
SUBAGENT_SPAWN           Turn result triggers it   Start new session process
                                                   concurrently
SESSION_END              Last turn complete        Write session summary
```

These are the **only** events you need. Everything inside a turn
(prefill, decode, KV cache) is handled by the adapter + backend
synchronously — no SimPy events needed inside a turn.

---

## Build Order

```
Week 1:   HardwareModel
          AnalyticalNPUBackend (HBM + DDR roofline)
          Unit tests: predict_latency matches known benchmarks

Week 2:   AgenticSessionGenerator
          Turn + think_time distributions calibrated from
          Claude Code trace data (Azure LLM dataset as proxy)

Week 3:   BaseAdapter interface + NativeAdapter
          (your own scheduler, no framework dependency yet)
          End-to-end SimPy run producing metrics CSV

Week 4:   VLLMAdapter
          Import vLLM scheduler + BlockSpaceManager in-process
          Validate: same sessions, same configs → same metrics
          as NativeAdapter (or understand the delta)

Week 5:   SGLangAdapter
          Same pattern, swap in SGLang's RadixAttention scheduler

Week 6:   LMCache layer
          Add prefix cache lookup into VLLMAdapter.submit()
          Compare: vLLM baseline vs. vLLM+LMCache

Week 7+:  RealGPUBackend + validation
          Run same sessions against real hardware
          Cross-validate analytical predictions
```
