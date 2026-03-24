"""
⚠️  SUPERSEDED — DO NOT USE FOR NEW CODE

This file has been split into two replacements under the final plan:

  integration/chips/profiles.py
    → ChipProfile, ModelConfig, TierSpec, ConfidenceLabel, CHIP_PROFILES
    → Byte-based tier specs (capacity_bytes, block_size_bytes)
    → page_size_tokens removed — use TierSpec.block_size_bytes instead

  core/des/oracle.py
    → CacheOracleBase, PiecewiseOracle, RooflineOracle, OracleFactory
    → Piecewise-linear interpolation from measured tables (primary)
    → Roofline math retained as ANALYTICAL_ONLY fallback only
    → Every latency call returns (latency_us, ConfidenceLabel)

Why superseded:
  1. HardwareModel roofline is now ConfidenceLabel.ANALYTICAL_ONLY —
     it cannot be positioned as the primary oracle for calibrated targets.
  2. page_size_tokens violates the byte-based accounting principle —
     the final plan requires all tier specs in bytes.
  3. LatencyResult uses milliseconds — Core DES uses microseconds (int64).
  4. File location (core/) conflicts with final plan layer structure —
     chip specs belong in integration/chips/, oracle in core/des/.

Retained for historical reference only.
agentsim_final_plan.md — Phase 0 architecture contracts.
"""

# --- ORIGINAL FILE BELOW — DO NOT IMPORT IN NEW CODE ---

"""
agentsim/core/hardware_model.py  [ORIGINAL]

Parameterized roofline hardware model.
Anchored to real chip specs but fully configurable.
No GPU, no CUDA, no torch dependency — pure math.
MIT License.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Chip Profile — static hardware spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipProfile:
    name:                   str
    hbm_bandwidth_gbps:     float       # HBM read bandwidth
    hbm_capacity_gb:        float       # total HBM capacity
    compute_tflops_bf16:    float       # peak BF16 TFLOPS
    page_size_tokens:       int         # KV cache page granularity
    interconnect_gbps:      float       # P2P / NVLink / Infinity Fabric

    # Optional DDR tier (non-GPU targets, custom NPUs)
    ddr_bandwidth_gbps:     Optional[float] = None
    ddr_capacity_gb:        Optional[float] = None

    # Optional CXL tier
    cxl_bandwidth_gbps:     Optional[float] = None
    cxl_capacity_gb:        Optional[float] = None

    @property
    def has_ddr(self) -> bool:
        return self.ddr_bandwidth_gbps is not None

    @property
    def total_capacity_gb(self) -> float:
        cap = self.hbm_capacity_gb
        if self.ddr_capacity_gb:
            cap += self.ddr_capacity_gb
        if self.cxl_capacity_gb:
            cap += self.cxl_capacity_gb
        return cap


# ---------------------------------------------------------------------------
# Real chip profiles — anchored to published specs
# ---------------------------------------------------------------------------

CHIP_PROFILES: dict[str, ChipProfile] = {

    "nvidia_h100_sxm": ChipProfile(
        name                 = "NVIDIA H100 SXM5",
        hbm_bandwidth_gbps   = 3350,
        hbm_capacity_gb      = 80,
        compute_tflops_bf16  = 989,
        page_size_tokens     = 16,
        interconnect_gbps    = 900,    # NVLink 4
    ),

    "nvidia_a100_80g": ChipProfile(
        name                 = "NVIDIA A100 80GB SXM",
        hbm_bandwidth_gbps   = 2000,
        hbm_capacity_gb      = 80,
        compute_tflops_bf16  = 312,
        page_size_tokens     = 16,
        interconnect_gbps    = 600,    # NVLink 3
    ),

    "amd_mi300x": ChipProfile(
        name                 = "AMD Instinct MI300X",
        hbm_bandwidth_gbps   = 5300,
        hbm_capacity_gb      = 192,
        compute_tflops_bf16  = 1307,
        page_size_tokens     = 16,
        interconnect_gbps    = 896,    # Infinity Fabric
    ),

    "intel_gaudi3": ChipProfile(
        name                 = "Intel Gaudi 3",
        hbm_bandwidth_gbps   = 3700,
        hbm_capacity_gb      = 128,
        compute_tflops_bf16  = 1835,
        page_size_tokens     = 16,
        interconnect_gbps    = 600,
    ),

    # Parameterized non-GPU target with HBM + DDR hierarchy
    # Tune these values to match your actual NPU spec sheet
    "custom_npu_hbm_ddr": ChipProfile(
        name                 = "Custom NPU (HBM + DDR)",
        hbm_bandwidth_gbps   = 900,
        hbm_capacity_gb      = 32,
        ddr_bandwidth_gbps   = 200,
        ddr_capacity_gb      = 256,
        compute_tflops_bf16  = 200,
        page_size_tokens     = 4096,   # your 4K page requirement
        interconnect_gbps    = 400,
    ),
}


# ---------------------------------------------------------------------------
# Model config — per-model memory cost constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    name:              str
    num_layers:        int
    num_heads:         int
    head_dim:          int
    num_kv_heads:      int
    hidden_dim:        int
    dtype_bytes:       int = 2         # BF16 = 2 bytes

    @property
    def bytes_per_token_kv(self) -> int:
        """KV cache bytes per token per layer (both K and V)."""
        return 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def total_bytes_per_token_kv(self) -> int:
        """KV cache bytes per token across all layers."""
        return self.bytes_per_token_kv * self.num_layers


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "llama3_8b": ModelConfig(
        name="Llama 3 8B",
        num_layers=32, num_heads=32, head_dim=128,
        num_kv_heads=8, hidden_dim=4096,
    ),
    "llama3_70b": ModelConfig(
        name="Llama 3 70B",
        num_layers=80, num_heads=64, head_dim=128,
        num_kv_heads=8, hidden_dim=8192,
    ),
    "llama3_405b": ModelConfig(
        name="Llama 3 405B",
        num_layers=126, num_heads=128, head_dim=128,
        num_kv_heads=8, hidden_dim=16384,
    ),
}


# ---------------------------------------------------------------------------
# Latency result
# ---------------------------------------------------------------------------

@dataclass
class LatencyResult:
    prefill_ms:        float    # time to first token
    decode_ms:         float    # time from first token to last token
    kv_load_ms:        float    # KV cache load from DDR (0 if HBM hit)
    total_ms:          float    # prefill + kv_load + decode
    bottleneck:        str      # "compute" | "hbm_bandwidth" | "ddr_bandwidth"

    # Breakdown flags
    compute_bound:     bool
    memory_bound:      bool


# ---------------------------------------------------------------------------
# Hardware Model — roofline latency prediction
# ---------------------------------------------------------------------------

class HardwareModel:
    """
    Analytical roofline model for LLM inference latency.

    Does not simulate individual tokens.
    Takes a batch description, returns latency in ms.
    No GPU required — pure arithmetic.

    Roofline model:
        latency = max(compute_time, memory_time)
        where:
            compute_time  = FLOPs / (TFLOPS × efficiency)
            memory_time   = bytes / (bandwidth_GBps × efficiency)
    """

    COMPUTE_EFFICIENCY = 0.55    # typical MFU for inference
    MEMORY_EFFICIENCY  = 0.85    # typical bandwidth utilization

    def __init__(self, chip: ChipProfile, model: ModelConfig):
        self.chip  = chip
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_prefill(
        self,
        input_tokens:       int,
        cached_tokens:      int  = 0,
        batch_size:         int  = 1,
        cache_location:     str  = "hbm",  # "hbm" | "ddr" | "cold"
    ) -> LatencyResult:
        """
        Predict latency for prefill phase (→ TTFT).

        cached_tokens: how many tokens already have KV in cache.
        cache_location: where those cached KVs live.
        new_tokens = input_tokens - cached_tokens (must be computed fresh).
        """
        new_tokens = max(0, input_tokens - cached_tokens)

        # --- Compute time for new tokens (attention + FFN) ---
        # Attention FLOPs ≈ 4 × seq_len × new_tokens × hidden_dim (approx)
        attn_flops = 4 * input_tokens * new_tokens * self.model.hidden_dim
        ffn_flops  = 8 * new_tokens * self.model.hidden_dim ** 2 / self.model.hidden_dim
        total_flops = (attn_flops + ffn_flops) * self.model.num_layers * batch_size

        compute_tflops = self.chip.compute_tflops_bf16 * 1e12
        compute_s = total_flops / (compute_tflops * self.COMPUTE_EFFICIENCY)

        # --- Memory time: load model weights for new tokens ---
        # Weight bytes touched per token ≈ 2 × hidden_dim² × num_layers × dtype
        weight_bytes = (
            2 * self.model.hidden_dim ** 2
            * self.model.num_layers
            * self.model.dtype_bytes
            * new_tokens
            / self.model.hidden_dim  # amortized per token
        )
        hbm_bw = self.chip.hbm_bandwidth_gbps * 1e9 * self.MEMORY_EFFICIENCY
        weight_load_s = weight_bytes / hbm_bw

        # --- KV cache load time (for cached prefix) ---
        kv_load_s = self._kv_load_latency(cached_tokens, cache_location)

        # --- Roofline ---
        compute_bound = compute_s >= weight_load_s
        memory_s = weight_load_s + kv_load_s
        prefill_s = max(compute_s, memory_s)

        return LatencyResult(
            prefill_ms    = prefill_s  * 1000,
            decode_ms     = 0.0,
            kv_load_ms    = kv_load_s  * 1000,
            total_ms      = (prefill_s) * 1000,
            bottleneck    = "compute" if compute_bound else "hbm_bandwidth",
            compute_bound = compute_bound,
            memory_bound  = not compute_bound,
        )

    def predict_decode(
        self,
        output_tokens:  int,
        batch_size:     int = 1,
        kv_tokens:      int = 0,   # total KV context at decode time
    ) -> LatencyResult:
        """
        Predict latency for decode phase (first token → last token).

        Decode is almost always memory-bandwidth bound.
        Each decode step loads model weights + full KV context once.
        """
        bytes_per_step = (
            # Model weights (loaded once per step per token)
            2 * self.model.hidden_dim ** 2
            * self.model.num_layers
            * self.model.dtype_bytes
            / self.model.hidden_dim
            * batch_size
            +
            # KV context (grows each step)
            self.model.total_bytes_per_token_kv * kv_tokens
        )

        hbm_bw  = self.chip.hbm_bandwidth_gbps * 1e9 * self.MEMORY_EFFICIENCY
        step_s  = bytes_per_step / hbm_bw
        total_s = step_s * output_tokens

        # Compute bound check (rare for decode)
        flops_per_step  = 2 * self.model.hidden_dim ** 2 * self.model.num_layers * 2
        compute_tflops  = self.chip.compute_tflops_bf16 * 1e12
        compute_step_s  = flops_per_step / (compute_tflops * self.COMPUTE_EFFICIENCY)
        compute_bound   = compute_step_s >= step_s

        return LatencyResult(
            prefill_ms    = 0.0,
            decode_ms     = total_s * 1000,
            kv_load_ms    = 0.0,
            total_ms      = total_s * 1000,
            bottleneck    = "compute" if compute_bound else "hbm_bandwidth",
            compute_bound = compute_bound,
            memory_bound  = not compute_bound,
        )

    def predict_turn(
        self,
        input_tokens:    int,
        output_tokens:   int,
        cached_tokens:   int = 0,
        cache_location:  str = "hbm",
        batch_size:      int = 1,
    ) -> LatencyResult:
        """
        Full turn = prefill + decode.
        Returns combined result with ttft = prefill_ms.
        """
        prefill = self.predict_prefill(
            input_tokens   = input_tokens,
            cached_tokens  = cached_tokens,
            batch_size     = batch_size,
            cache_location = cache_location,
        )
        decode = self.predict_decode(
            output_tokens = output_tokens,
            batch_size    = batch_size,
            kv_tokens     = input_tokens,
        )
        total_ms = prefill.total_ms + decode.total_ms
        return LatencyResult(
            prefill_ms    = prefill.prefill_ms,
            decode_ms     = decode.decode_ms,
            kv_load_ms    = prefill.kv_load_ms,
            total_ms      = total_ms,
            bottleneck    = prefill.bottleneck,
            compute_bound = prefill.compute_bound,
            memory_bound  = prefill.memory_bound,
        )

    def kv_pages_required(self, token_count: int) -> int:
        """How many KV pages does this token count need?"""
        return math.ceil(token_count / self.chip.page_size_tokens)

    def kv_bytes_required(self, token_count: int) -> int:
        """Total KV cache bytes for this token count."""
        return token_count * self.model.total_bytes_per_token_kv

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kv_load_latency(self, cached_tokens: int, location: str) -> float:
        """Latency to load cached KV from the given memory tier."""
        if cached_tokens == 0 or location == "cold":
            return 0.0

        kv_bytes = self.kv_bytes_required(cached_tokens)

        if location == "hbm":
            bw = self.chip.hbm_bandwidth_gbps * 1e9 * self.MEMORY_EFFICIENCY
        elif location == "ddr":
            if not self.chip.has_ddr:
                raise ValueError(f"Chip {self.chip.name} has no DDR tier")
            bw = self.chip.ddr_bandwidth_gbps * 1e9 * self.MEMORY_EFFICIENCY
        elif location == "transfer":
            # Cross-node KV transfer (disaggregated prefill/decode)
            bw = self.chip.interconnect_gbps * 1e9 * self.MEMORY_EFFICIENCY
        else:
            raise ValueError(f"Unknown cache location: {location}")

        return kv_bytes / bw
