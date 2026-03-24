"""
⚠️  DEMOTED — for sweep tool use only, not authoritative simulation

Under the final plan, this module is replaced by core/des/workload.py
which ports cache-sim's validated NHPP workload generator with 5
research-calibrated profiles (coding, agentic_coding, chat, batch, agent).

What's wrong with this module vs final plan:
  1. Uses simple Poisson arrivals (expovariate) — Core DES requires NHPP
     with diurnal pattern (peak at 9 AM) to model realistic sustained load.
  2. TokenProfileDistribution is not calibrated to cache-sim profiles:
     - system_prompt_tokens=8000 vs validated coding=20k, agentic_coding=30k
     - max_turns=25 vs coding=1hr sessions, agentic_coding=30-min sessions
  3. AgenticSessionGenerator positioned in core/ — final plan demotes
     this to sweep/ only.

What IS preserved:
  - AgenticSession, Turn dataclasses — concept is correct, used in sweep
  - ThinkTimeDistribution — LogNormal think time is directionally right
  - SweepSessionGenerator renamed to clarify its sweep-only role

For authoritative session modeling, use core/des/workload.py (Phase 1).
MIT License.
"""

# --- ORIGINAL FILE BELOW — USE ONLY IN sweep/ CONTEXT ---

"""
agentsim/core/session_model.py  [ORIGINAL — SWEEP ONLY]

Agentic session model for Claude Code-style workloads.
Generates sessions with multi-turn structure, think-time distributions,
and realistic token profiles calibrated to coding assistant behavior.
MIT License.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Optional, List
import random
import math


# ---------------------------------------------------------------------------
# Turn — one request/response pair within a session
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    turn_id:             int
    input_tokens:        int    # total input incl. system prompt + history
    output_tokens:       int    # expected output (sampled)
    cached_prefix_len:   int    # tokens that should be in KV cache

    # These tell the cache simulation what the prefix looks like
    system_prompt_tokens: int   # stable across all turns of session
    tool_def_tokens:      int   # stable — tool list injected each turn
    file_context_tokens:  int   # grows each turn as files are read
    history_tokens:       int   # grows each turn

    @property
    def new_tokens(self) -> int:
        """Tokens that are NOT cached — must be prefilled fresh."""
        return max(0, self.input_tokens - self.cached_prefix_len)

    @property
    def cache_pressure_bytes(self) -> int:
        """Rough estimate of KV bytes this turn will occupy."""
        return self.input_tokens * 2 * 128 * 2  # assumes GQA kv_heads=8, head_dim=128, bf16


# ---------------------------------------------------------------------------
# Agentic Session
# ---------------------------------------------------------------------------

@dataclass
class AgenticSession:
    session_id:      str
    turns:           List[Turn]
    sub_sessions:    List[AgenticSession] = field(default_factory=list)

    # Which turn spawns each sub-session (for sub-agent modeling)
    sub_session_spawn_turn: dict[int, List[str]] = field(default_factory=dict)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def num_turns(self) -> int:
        return len(self.turns)


# ---------------------------------------------------------------------------
# Think-time distribution
# Represents the gap between receiving a response and sending the next turn.
# For an agentic session, this models the agent parsing the tool response,
# deciding the next action, and formulating the next request.
# ---------------------------------------------------------------------------

class ThinkTimeDistribution:
    """
    LogNormal distribution for inter-turn think time.

    Calibrated from Claude Code-style workloads:
    - Most turns: agent quickly decides next tool call (2–5s)
    - Some turns: agent reads a long file result (10–30s)
    - Rare turns: user is reviewing output (60–300s)
    - Very rare: user steps away (>300s → likely expiry event)

    Parameters tuned so ~15% of turns exceed 5-minute TTL.
    """

    def __init__(
        self,
        mean_s:  float = 45.0,   # mean think time in seconds
        sigma:   float = 1.4,    # log-space std dev
        min_s:   float = 0.5,    # floor — agent needs at least this long
        max_s:   float = 3600.0, # ceiling — session likely dead beyond this
        rng:     Optional[random.Random] = None,
    ):
        self.mean_s = mean_s
        self.sigma  = sigma
        self.min_s  = min_s
        self.max_s  = max_s
        self.rng    = rng or random.Random()

        # Compute lognormal mu from desired mean
        # E[X] = exp(mu + sigma²/2)
        self.mu = math.log(mean_s) - (sigma ** 2) / 2

    def sample(self) -> float:
        """Sample a think time in seconds."""
        val = self.rng.lognormvariate(self.mu, self.sigma)
        return max(self.min_s, min(self.max_s, val))

    def p_expiry(self, ttl_s: float = 300.0) -> float:
        """Fraction of turns expected to exceed the cache TTL."""
        # CDF of lognormal: Φ((ln(x) - mu) / sigma)
        from math import erf, sqrt
        z = (math.log(ttl_s) - self.mu) / self.sigma
        return 0.5 * (1 - erf(z / sqrt(2)))


# ---------------------------------------------------------------------------
# Token profile distributions
# Calibrated to coding assistant workloads.
# ---------------------------------------------------------------------------

class TokenProfileDistribution:
    """
    Generates realistic token counts for each turn of a Claude Code session.

    Key characteristics of coding assistant workloads:
    - System prompt: large and STABLE (50K–100K tokens) — high cache value
    - Tool definitions: STABLE, injected every turn
    - File contents: GROW each turn as agent reads more files
    - Output tokens: SHORT relative to input (edits, not essays)
    - High input:output ratio (5:1 to 20:1)
    """

    def __init__(
        self,
        system_prompt_tokens:  int   = 8000,   # CLAUDE.md + tool defs
        tool_def_tokens:       int   = 2000,   # tool list
        initial_file_tokens:   int   = 5000,   # files read on first turn
        file_growth_per_turn:  float = 0.3,    # 30% chance of reading new file
        new_file_tokens_mean:  int   = 3000,   # new file size
        output_tokens_mean:    int   = 800,    # typical response length
        output_tokens_sigma:   float = 0.8,    # log-space std dev
        max_turns:             int   = 25,
        rng:                   Optional[random.Random] = None,
    ):
        self.system_prompt_tokens  = system_prompt_tokens
        self.tool_def_tokens       = tool_def_tokens
        self.initial_file_tokens   = initial_file_tokens
        self.file_growth_per_turn  = file_growth_per_turn
        self.new_file_tokens_mean  = new_file_tokens_mean
        self.output_tokens_mean    = output_tokens_mean
        self.output_tokens_sigma   = output_tokens_sigma
        self.max_turns             = max_turns
        self.rng                   = rng or random.Random()

    def generate_turns(self, session_id: str) -> List[Turn]:
        """Generate a realistic sequence of turns for one session."""
        turns = []

        # Stable prefix: system prompt + tool defs (cached from turn 1)
        stable_prefix = self.system_prompt_tokens + self.tool_def_tokens

        # Growing context: accumulated file contents + history
        file_tokens    = self.initial_file_tokens
        history_tokens = 0

        num_turns = self.rng.randint(3, self.max_turns)

        for i in range(num_turns):
            # Possibly read a new file this turn
            if i > 0 and self.rng.random() < self.file_growth_per_turn:
                new_file = int(
                    self.rng.lognormvariate(
                        math.log(self.new_file_tokens_mean) - 0.3,
                        0.6
                    )
                )
                file_tokens += new_file

            # User query tokens (short — the actual instruction)
            user_query_tokens = self.rng.randint(20, 200)

            total_input = (
                stable_prefix
                + file_tokens
                + history_tokens
                + user_query_tokens
            )

            # Cached prefix = stable part + file contents already seen
            # On turn 0: nothing cached yet (cold start)
            # On turn 1+: stable prefix + previous file contents are cached
            if i == 0:
                cached = 0
            else:
                cached = stable_prefix + (file_tokens - self._last_new_file_tokens)

            cached = min(cached, total_input)

            # Output tokens (short for coding edits)
            output = max(50, int(
                self.rng.lognormvariate(
                    math.log(self.output_tokens_mean) - (self.output_tokens_sigma**2)/2,
                    self.output_tokens_sigma
                )
            ))

            turns.append(Turn(
                turn_id               = i,
                input_tokens          = total_input,
                output_tokens         = output,
                cached_prefix_len     = cached,
                system_prompt_tokens  = self.system_prompt_tokens,
                tool_def_tokens       = self.tool_def_tokens,
                file_context_tokens   = file_tokens,
                history_tokens        = history_tokens,
            ))

            # History grows by output tokens of this turn
            history_tokens += output
            self._last_new_file_tokens = 0  # reset for next turn

        return turns

    # internal state for turn generation
    _last_new_file_tokens: int = 0


# ---------------------------------------------------------------------------
# Session Generator — produces sessions for the SimPy event loop
# ---------------------------------------------------------------------------

class AgenticSessionGenerator:
    """
    Generates agentic sessions for simulation.

    Two modes:
    - synthetic: Poisson arrivals, LogNormal think times, parametric tokens
    - trace:     Replay from recorded session trace file (CSV/JSONL)
    """

    CACHE_TTL_S = 300.0   # Anthropic ephemeral cache TTL

    def __init__(
        self,
        mode:              str   = "synthetic",   # "synthetic" | "trace"
        arrival_rate_qps:  float = 1.0,           # sessions per second
        token_profile:     Optional[TokenProfileDistribution] = None,
        think_time_dist:   Optional[ThinkTimeDistribution]    = None,
        include_subagents: bool  = True,
        subagent_prob:     float = 0.3,           # prob a turn spawns subagent
        seed:              int   = 42,
        trace_file:        Optional[str] = None,
    ):
        self.mode              = mode
        self.arrival_rate_qps  = arrival_rate_qps
        self.include_subagents = include_subagents
        self.subagent_prob     = subagent_prob
        self.trace_file        = trace_file

        self.rng = random.Random(seed)
        self.token_profile = token_profile or TokenProfileDistribution(
            rng=self.rng
        )
        self.think_time_dist = think_time_dist or ThinkTimeDistribution(
            rng=self.rng
        )
        self._session_counter = 0

    def next_arrival_gap_s(self) -> float:
        """Inter-session arrival gap (exponential = Poisson arrivals)."""
        return self.rng.expovariate(self.arrival_rate_qps)

    def generate_session(self) -> AgenticSession:
        """Generate one complete agentic session."""
        self._session_counter += 1
        session_id = f"s{self._session_counter:06d}"

        turns = self.token_profile.generate_turns(session_id)

        # Sub-sessions (sub-agents spawned by tool calls)
        sub_sessions = []
        spawn_map: dict[int, List[str]] = {}

        if self.include_subagents:
            for t in turns:
                if self.rng.random() < self.subagent_prob:
                    sub = self._generate_subagent_session(session_id, t.turn_id)
                    sub_sessions.append(sub)
                    spawn_map.setdefault(t.turn_id, []).append(sub.session_id)

        return AgenticSession(
            session_id              = session_id,
            turns                   = turns,
            sub_sessions            = sub_sessions,
            sub_session_spawn_turn  = spawn_map,
        )

    def sample_think_time_s(self) -> float:
        """Sample inter-turn think time in seconds."""
        return self.think_time_dist.sample()

    def will_expire(self, think_time_s: float) -> bool:
        """Would this think time cross the Anthropic cache TTL?"""
        return think_time_s >= self.CACHE_TTL_S

    # ------------------------------------------------------------------

    def _generate_subagent_session(
        self, parent_id: str, spawn_turn: int
    ) -> AgenticSession:
        """Sub-agents are shorter, shallower sessions."""
        sub_id = f"{parent_id}_sub{spawn_turn}"
        sub_profile = TokenProfileDistribution(
            system_prompt_tokens = 4000,   # lighter system prompt
            max_turns            = 5,      # sub-agents are short
            rng                  = self.rng,
        )
        turns = sub_profile.generate_turns(sub_id)
        return AgenticSession(session_id=sub_id, turns=turns)
