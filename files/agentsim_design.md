# AgentSim — Final Locked Design
# Three-Phase Agentic LLM Inference Simulator

---

## The Three Goals — Final Spec

| | Goal 1 | Goal 2 | Goal 3 |
|---|---|---|---|
| **Purpose** | KV cache policy efficiency at scale | Detect cache miss patterns in streaming event stream | De-risk non-NVIDIA hardware, validate framework interop |
| **Granularity** | Request-level | Protocol checkpoint events | Request-level + real framework code paths |
| **Speed** | Must be fast — billions of tokens, tens of workers | Single-session deep analysis OK | Correctness over speed |
| **Framework** | Algorithmic in-process | Protocol-aware adapter (Anthropic + OpenAI) | Real vLLM / SGLang adapters |
| **Hardware** | Analytical roofline | Analytical roofline | Parameterized, anchored to real chip specs |
| **Build phase** | Phase 1 (Weeks 1–4) | Phase 2 (Weeks 5–7) | Phase 3 (Weeks 8–10) |

---

## Protocol Event Schema — The Heart of Goal 2

### Anthropic Messages API — SSE Event Sequence

```
CONNECTION_OPEN
    │
    ▼
message_start                        ← CHECKPOINT: CACHE_DECISION
    usage:
      input_tokens: N
      cache_read_input_tokens: R      ← R > 0 = cache HIT (prefix reused)
      cache_creation_input_tokens: W  ← W > 0 = cache WRITE (new entry)
    │
    │  gap = pure prefill time
    ▼
content_block_start (type: text)
    │
    ▼
content_block_delta                  ← CHECKPOINT: TTFT
    delta.text: "..."                   (first content token)
    │
    ▼
content_block_delta × N              ← CHECKPOINT: STREAMING_THROUGHPUT
    (inter-delta timing = decode rate)  measure inter-event gaps
    │
    ▼
content_block_stop
    │
    ▼
message_delta                        ← CHECKPOINT: TURN_COMPLETE
    usage.output_tokens: M
    │
    ▼
message_stop
    │
    ▼
CONNECTION_CLOSE

Cache miss detection:
  COLD_MISS:    cache_read == 0 AND cache_creation > 0  (first time seeing prefix)
  WARM_HIT:     cache_read > 0  AND cache_creation == 0 (full prefix reuse)
  PARTIAL_HIT:  cache_read > 0  AND cache_creation > 0  (prefix grew)
  EXPIRY_MISS:  cache_read == 0 AND cache_creation > 0  (was hit before, now expired)
                → detected by comparing against previous turn's write
```

### OpenAI Chat Completions API — SSE Event Sequence

```
CONNECTION_OPEN
    │
    ▼
data: {"id":..., "choices":[{"delta":{"role":"assistant"}, "index":0}]}
    │                                ← NO cache signal at this point
    │  gap = prefill time (inferred from TTFT latency spike vs. expected)
    ▼
data: {"choices":[{"delta":{"content":"<first token>"}}]}  ← CHECKPOINT: TTFT
    │
    ▼
data: {"choices":[{"delta":{"content":"..."}}]} × N        ← STREAMING_THROUGHPUT
    │
    ▼
data: {"choices":[{"delta":{}, "finish_reason":"stop"}]}   ← TURN_COMPLETE
    │
    ▼
data: [DONE]
    │
    ▼
CONNECTION_CLOSE

Cache miss detection (inferred, not explicit):
  PROBABLE_MISS:  TTFT > expected_prefill_latency_for_hit × threshold
  PROBABLE_HIT:   TTFT ≈ near-zero (prefix served from cache)
  → Calibrate threshold per model per hardware from known-hit baselines
```

### OpenAI Responses API — SSE Event Sequence

```
CONNECTION_OPEN
    │
    ▼
response.created                     ← session open
    │
    ▼
response.output_item.added           ← content block starting
    │
    ▼
response.content_part.added
    │
    ▼
response.output_text.delta           ← CHECKPOINT: TTFT (first delta)
    delta: "<first token>"
    │
    ▼
response.output_text.delta × N       ← STREAMING_THROUGHPUT
    │
    ▼
response.output_text.done            ← CHECKPOINT: TURN_COMPLETE
    │
    ▼
response.output_item.done
    │
    ▼
response.done
    │
    ▼
CONNECTION_CLOSE

Cache: Same inference approach as Chat Completions.
```

---

## Unified Checkpoint Event Schema (All Protocols → One Model)

```python
@dataclass
class CheckpointEvent:
    # Identity
    session_id:    str
    turn_id:       int
    protocol:      Literal["anthropic", "openai_chat", "openai_responses"]

    # Event type
    kind: Literal[
        "CACHE_DECISION",        # Anthropic only — explicit signal
        "TTFT",                  # All protocols — first content token
        "BATCH_BOUNDARY",        # Internal — new request joined decode batch
        "EVICTION_TRIGGER",      # Internal — KV cache pressure event
        "STREAMING_STALL",       # All — inter-delta gap exceeds threshold
        "TURN_COMPLETE",         # All — last token / finish_reason
        "EXPIRY_BOUNDARY",       # Anthropic — 5-min TTL crossed mid-session
    ]

    # Timing
    wall_time_ms:   float        # real or simulated wall clock
    sim_time_ms:    float        # SimPy simulation time

    # Cache state at this checkpoint
    cache_read_tokens:     int   # tokens served from cache  (Anthropic: explicit)
    cache_written_tokens:  int   # tokens written to cache   (Anthropic: explicit)
    cache_miss_type: Optional[Literal[
        "COLD",        # never seen this prefix
        "EXPIRY",      # seen before, TTL expired
        "EVICTION",    # seen before, evicted under memory pressure
        "NONE",        # hit — no miss
    ]]

    # Latency intervals
    prefill_latency_ms:   Optional[float]  # connection_open → TTFT
    decode_throughput_tps: Optional[float] # tokens/sec during decode phase

    # Protocol-specific raw payload (for debugging)
    raw_event: Optional[dict]
```

---

## Ephemeral Cache Expiry (5-Minute TTL) — Simulation Model

```
Anthropic's ephemeral cache has a 5-minute TTL.
In an agentic coding session, turns can be spaced > 5 minutes apart
(user reviewing code, running tests, etc.)

Simulation model:

  cache_entry_expires_at = turn_N_message_start_time + 300_000ms

  When turn N+1 fires:
    if sim_time >= cache_entry_expires_at:
        → EXPIRY_MISS
        → cache_creation_tokens = full_prefix_len  (re-write full prefix)
        → cache_read_tokens = 0
    else:
        → WARM_HIT or PARTIAL_HIT depending on prefix growth

  Think time distribution calibration:
    LogNormal(mean=45s, sigma=60s) for typical Claude Code turns
    → ~15% of turns will experience expiry under this distribution
    → This is the key tail event Goal 2 is designed to surface
```

---

## Cross-Request Prefix Reuse Detection

```
Within a session (sequential turns):
  Trivial — same session_id, prefix grows monotonically.
  Reuse = previous turn's output absorbed into next turn's input.

Across sessions (different users hitting same system prompt):
  Shared system prompt prefix = same first N tokens.
  Detection signal: cache_read_tokens > 0 on FIRST turn of a session.
  If first turn shows cache_read > 0 → cross-session prefix reuse confirmed.
  This is the LMCache / vLLM prefix caching payoff measurement.

Across prefill/decode disaggregated nodes (Goal 3):
  Prefix computed on prefill node, KV transferred to decode node.
  Miss appears as TTFT spike even when cache_read > 0
  → transfer latency adds to prefill, not a true miss but looks like one.
  Goal 3 hardware model must distinguish: cache_hit_with_transfer vs cold_miss.
```

---

## Hardware Model — Parameterized + Anchored to Real Chips

```python
# Real chip targets (Phase 3 anchors)
CHIP_PROFILES = {
    "nvidia_h100_sxm": ChipProfile(
        hbm_bandwidth_gbps   = 3350,
        hbm_capacity_gb      = 80,
        ddr_bandwidth_gbps   = None,   # no DDR tier
        compute_tflops_bf16  = 989,
        page_size_tokens     = 16,     # vLLM default
        interconnect_gbps    = 900,    # NVLink 4
    ),
    "amd_mi300x": ChipProfile(
        hbm_bandwidth_gbps   = 5300,
        hbm_capacity_gb      = 192,
        ddr_bandwidth_gbps   = None,
        compute_tflops_bf16  = 1307,
        page_size_tokens     = 16,
        interconnect_gbps    = 896,    # Infinity Fabric
    ),
    "intel_gaudi3": ChipProfile(
        hbm_bandwidth_gbps   = 3700,
        hbm_capacity_gb      = 128,
        ddr_bandwidth_gbps   = None,
        compute_tflops_bf16  = 1835,
        page_size_tokens     = 16,
        interconnect_gbps    = 600,
    ),
    "custom_npu_hbm_ddr": ChipProfile(
        hbm_bandwidth_gbps   = 900,    # parameterized
        hbm_capacity_gb      = 32,
        ddr_bandwidth_gbps   = 200,    # DDR tier present
        ddr_capacity_gb      = 256,
        compute_tflops_bf16  = 200,
        page_size_tokens     = 4096,   # your 4K requirement
        interconnect_gbps    = 400,
    ),
}
```

---

## Complete Repository Structure

```
agentsim/
├── core/
│   ├── hardware_model.py     ← ChipProfile, roofline latency prediction
│   ├── session_model.py      ← AgenticSession, Turn, think-time distributions
│   └── events.py             ← CheckpointEvent, all event kinds, unified schema
│
├── protocol/
│   ├── base.py               ← ProtocolAdapter ABC
│   ├── anthropic_adapter.py  ← SSE → CheckpointEvent, cache signal extraction
│   ├── openai_chat_adapter.py← SSE → CheckpointEvent, inferred cache detection
│   └── openai_responses_adapter.py
│
├── sim/
│   ├── request_sim.py        ← Goal 1: fast request-level SimPy simulator
│   ├── checkpoint_sim.py     ← Goal 2: protocol checkpoint event simulator
│   └── workers.py            ← parallel worker pool for Goal 1 scale
│
├── framework/
│   ├── base.py               ← FrameworkAdapter ABC
│   ├── native_adapter.py     ← Goal 1/2: your own scheduler, no framework dep
│   ├── vllm_adapter.py       ← Goal 3: real vLLM scheduler in-process
│   └── sglang_adapter.py     ← Goal 3: real SGLang scheduler in-process
│
├── chips/
│   ├── profiles.py           ← CHIP_PROFILES dict (all real chip specs)
│   └── custom_npu.py         ← your parameterized non-GPU model
│
└── metrics/
    ├── collector.py          ← MetricsCollector, per-session, per-turn
    └── report.py             ← summary DataFrames, CDF plots, miss breakdown
```
