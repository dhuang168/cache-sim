"""
integration/chips/profiles.py

Static hardware specifications only.
No latency math here — that belongs in core/des/oracle.py.

Key principle from final plan:
  - All tier capacities are BYTES
  - Tokens are metadata, never the accounting unit
  - page_size_tokens is REMOVED — use TierSpec.block_size_bytes instead

MIT License.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Confidence label — required on every hardware target
# Appears in all report outputs, not just documentation.
# ---------------------------------------------------------------------------

class ConfidenceLabel(str, Enum):
    CALIBRATED       = "calibrated"       # measured oracle tables exist
    SEMI_CALIBRATED  = "semi-calibrated"  # partial tables + interpolation
    ANALYTICAL_ONLY  = "analytical-only"  # roofline math, no measurements


# ---------------------------------------------------------------------------
# TierSpec — one memory tier, fully byte-based
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierSpec:
    """
    Specification for one memory tier (L1/L2/L3A).

    All capacity and transfer quantities are BYTES.
    Tokens are never used as a capacity unit.
    """
    name:               str     # "L1" | "L2" | "L3A"
    medium:             str     # "HBM" | "DRAM" | "DDR" | "SSD" | "NVMe"
    capacity_bytes:     int     # total capacity in bytes
    bandwidth_bps:      int     # peak read bandwidth in bytes/sec
    block_size_bytes:   int     # minimum transfer / page granularity in bytes
    scope:              str     # "per_gpu" | "per_worker" | "global"
    latency_floor_us:   int     # minimum access latency in microseconds
    remote_latency_us:  int = 0 # additional latency for cross-worker access (L3A global)

    @property
    def capacity_gb(self) -> float:
        return self.capacity_bytes / (1024 ** 3)

    @property
    def bandwidth_gbps(self) -> float:
        return self.bandwidth_bps / (1024 ** 3)


# ---------------------------------------------------------------------------
# ChipProfile — static hardware spec, no latency math
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipProfile:
    """
    Static specification for a compute chip and its memory hierarchy.

    Tiers are expressed as TierSpec objects (byte-based).
    No page_size_tokens — that was token-based and is removed.
    Block size is now block_size_bytes on each TierSpec.

    confidence: required for all targets.
    oracle_table: path to benchmark latency table, or None for analytical-only.
    """
    name:               str
    tiers:              tuple[TierSpec, ...]
    compute_tflops_bf16: float          # peak BF16 TFLOPS
    interconnect_bps:   int             # P2P bandwidth (NVLink / IF / PCIe)
    n_gpus_per_worker:  int             # GPUs sharing L2/L3A on same host
    confidence:         ConfidenceLabel
    oracle_table:       Optional[str]   # path to JSON oracle table, or None

    @property
    def l1(self) -> Optional[TierSpec]:
        return next((t for t in self.tiers if t.name == "L1"), None)

    @property
    def l2(self) -> Optional[TierSpec]:
        return next((t for t in self.tiers if t.name == "L2"), None)

    @property
    def l3a(self) -> Optional[TierSpec]:
        return next((t for t in self.tiers if t.name == "L3A"), None)

    @property
    def interconnect_gbps(self) -> float:
        return self.interconnect_bps / (1024 ** 3)


# ---------------------------------------------------------------------------
# ModelConfig — per-model KV cache math
# (unchanged from original — bytes_per_token_kv was already correct)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    name:           str
    num_layers:     int
    num_heads:      int
    head_dim:       int
    num_kv_heads:   int
    hidden_dim:     int
    dtype_bytes:    int = 2     # BF16 = 2 bytes

    @property
    def bytes_per_token_kv(self) -> int:
        """KV cache bytes per token per layer (K + V)."""
        return 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def total_bytes_per_token_kv(self) -> int:
        """KV cache bytes per token across ALL layers."""
        return self.bytes_per_token_kv * self.num_layers

    def kv_bytes_for_tokens(self, token_count: int) -> int:
        """Total KV cache bytes for a given token count."""
        return token_count * self.total_bytes_per_token_kv

    def block_size_bytes_for_tokens(self, page_tokens: int) -> int:
        """
        Convert a token-count page size to bytes.
        Use this when interfacing with systems that specify page size in tokens.
        Example: vLLM block_size=16 → 16 * total_bytes_per_token_kv
        """
        return page_tokens * self.total_bytes_per_token_kv


# ---------------------------------------------------------------------------
# Real chip profiles
# ---------------------------------------------------------------------------

# Standard L1 spec for A100/H100 (80 GB HBM, vLLM default block)
def _gpu_l1(capacity_gb: float, bandwidth_tbps: float) -> TierSpec:
    return TierSpec(
        name             = "L1",
        medium           = "HBM",
        capacity_bytes   = int(capacity_gb * 1024**3),
        bandwidth_bps    = int(bandwidth_tbps * 1024**4),
        block_size_bytes = 5 * 1024,          # 5 KB — cache-sim default
        scope            = "per_gpu",
        latency_floor_us = 1,
    )

# Standard L2 spec for 8-GPU server (1 TB DRAM shared by 8 GPUs)
_GPU_SERVER_L2 = TierSpec(
    name             = "L2",
    medium           = "DRAM",
    capacity_bytes   = 1 * 1024**4,           # 1 TB
    bandwidth_bps    = int(64 * 1024**3),      # 64 GB/s
    block_size_bytes = 32 * 1024**2,           # 32 MB — cache-sim default
    scope            = "per_worker",
    latency_floor_us = 100,
)

# Standard L3A spec (8 TB NVMe SSD per worker)
def _gpu_l3a(remote: bool = False) -> TierSpec:
    return TierSpec(
        name             = "L3A",
        medium           = "SSD",
        capacity_bytes   = 8 * 1024**4,        # 8 TB
        bandwidth_bps    = int(25 * 1024**3),  # 25 GB/s
        block_size_bytes = 256 * 1024**2,      # 256 MB — cache-sim default
        scope            = "global" if remote else "per_worker",
        latency_floor_us = 1_000,
        remote_latency_us = 50_000 if remote else 0,  # 50ms cross-worker
    )


CHIP_PROFILES: dict[str, ChipProfile] = {

    "nvidia_h100_sxm": ChipProfile(
        name               = "NVIDIA H100 SXM5",
        tiers              = (
            _gpu_l1(capacity_gb=80, bandwidth_tbps=3.35),
            _GPU_SERVER_L2,
            _gpu_l3a(),
        ),
        compute_tflops_bf16 = 989,
        interconnect_bps   = int(900 * 1024**3),   # 900 GB/s NVLink 4
        n_gpus_per_worker  = 8,
        confidence         = ConfidenceLabel.SEMI_CALIBRATED,
        oracle_table       = "benchmarks/oracle_tables/h100_llama3_70b.json",
    ),

    "nvidia_a100_80g": ChipProfile(
        name               = "NVIDIA A100 80GB SXM",
        tiers              = (
            _gpu_l1(capacity_gb=80, bandwidth_tbps=2.0),
            TierSpec(
                name             = "L2",
                medium           = "DRAM",
                # 1 TB host DRAM shared across 8 GPUs on the same worker.
                # Bandwidth: 25 GB/s per GPU × 8 GPUs = 200 GB/s aggregate.
                capacity_bytes   = 1 * 1024**4,           # 1 TB shared
                bandwidth_bps    = int(200 * 1024**3),    # 200 GB/s aggregate (25 GB/s × 8)
                block_size_bytes = 32 * 1024**2,          # 32 MB
                scope            = "per_worker",
                latency_floor_us = 100,
            ),
            _gpu_l3a(),
        ),
        compute_tflops_bf16 = 312,
        # Intra-node: NVLink 3 at 600 GB/s (between GPUs on same server)
        #             — implicit in same-worker L2 access, not stored here
        # Cross-node: 200 GB/s total / 8 GPUs = 25 GB/s per GPU
        #             — stored as per-GPU cross-node rate for L3A transfer math
        interconnect_bps   = int(25 * 1024**3),           # 25 GB/s per GPU cross-node
        n_gpus_per_worker  = 8,
        confidence         = ConfidenceLabel.CALIBRATED,
        oracle_table       = "benchmarks/oracle_tables/a100_llama3_70b.json",
    ),

    "amd_mi300x": ChipProfile(
        name               = "AMD Instinct MI300X",
        tiers              = (
            _gpu_l1(capacity_gb=192, bandwidth_tbps=5.3),
            _GPU_SERVER_L2,
            _gpu_l3a(),
        ),
        compute_tflops_bf16 = 1307,
        interconnect_bps   = int(896 * 1024**3),   # 896 GB/s Infinity Fabric
        n_gpus_per_worker  = 8,
        confidence         = ConfidenceLabel.ANALYTICAL_ONLY,
        oracle_table       = None,   # no measured table yet
    ),

    "intel_gaudi3": ChipProfile(
        name               = "Intel Gaudi 3",
        tiers              = (
            _gpu_l1(capacity_gb=128, bandwidth_tbps=3.7),
            _GPU_SERVER_L2,
            _gpu_l3a(),
        ),
        compute_tflops_bf16 = 1835,
        interconnect_bps   = int(600 * 1024**3),
        n_gpus_per_worker  = 8,
        confidence         = ConfidenceLabel.ANALYTICAL_ONLY,
        oracle_table       = None,
    ),

    # Custom NPU with HBM + DDR + SSD three-tier hierarchy (Option A).
    # Mirrors the GPU topology (L1/L2/L3A) but with NPU-specific specs.
    #
    # Note on "4K page size" from original spec:
    #   "4K" refers to 4096 tokens per KV page.
    #   In bytes, for Llama3-70B BF16:
    #   4096 tokens × (80 layers × 8 kv_heads × 128 dim × 2 × 2 bytes) = ~671 MB
    #   That is the block_size_bytes on L1 below.
    #   Override at runtime with model.block_size_bytes_for_tokens(4096).
    #
    # Two named variants:
    #   "custom_npu_local_l3a"  — each worker's SSD is local only
    #   "custom_npu_global_l3a" — SSDs are pooled across all workers
    # This mirrors the GPU global/local L3A toggle for direct comparison.
    "custom_npu_local_l3a": ChipProfile(
        name               = "Custom NPU (HBM + DDR + SSD, local L3A)",
        tiers              = (
            TierSpec(
                name             = "L1",
                medium           = "HBM",
                capacity_bytes   = 64 * 1024**3,         # 64 GB
                bandwidth_bps    = int(1800 * 1024**3),  # 1,800 GB/s
                block_size_bytes = 4096 * 160 * 2,       # 4K-token page placeholder
                scope            = "per_gpu",
                latency_floor_us = 1,
            ),
            TierSpec(
                name             = "L2",
                medium           = "DDR",
                # 256 GB per NPU × 16 NPUs = 4 TB total shared pool.
                # Bandwidth: 64 GB/s per NPU × 16 NPUs = 1,024 GB/s aggregate.
                capacity_bytes   = 4 * 1024**4,           # 4 TB shared across 16 NPUs
                bandwidth_bps    = int(1024 * 1024**3),   # 1,024 GB/s aggregate (64 GB/s × 16)
                block_size_bytes = 32 * 1024**2,          # 32 MB
                scope            = "per_worker",
                latency_floor_us = 200,
            ),
            TierSpec(
                name              = "L3A",
                medium            = "SSD",
                capacity_bytes    = 8 * 1024**4,         # 8 TB
                bandwidth_bps     = int(25 * 1024**3),   # 25 GB/s
                block_size_bytes  = 256 * 1024**2,       # 256 MB
                scope             = "per_worker",        # local — not shared
                latency_floor_us  = 1_000,
                remote_latency_us = 0,
            ),
        ),
        compute_tflops_bf16 = 640,
        # 25 GB/s cross-node interconnect — same as A100 cross-node.
        # Used for cross-worker L3A transfers in global mode.
        interconnect_bps    = int(25 * 1024**3),         # 25 GB/s per NPU
        n_gpus_per_worker   = 16,
        confidence          = ConfidenceLabel.ANALYTICAL_ONLY,
        oracle_table        = None,
    ),

    "custom_npu_global_l3a": ChipProfile(
        name               = "Custom NPU (HBM + DDR + SSD, global L3A)",
        tiers              = (
            TierSpec(
                name             = "L1",
                medium           = "HBM",
                capacity_bytes   = 64 * 1024**3,         # 64 GB
                bandwidth_bps    = int(1800 * 1024**3),  # 1,800 GB/s
                block_size_bytes = 4096 * 160 * 2,
                scope            = "per_gpu",
                latency_floor_us = 1,
            ),
            TierSpec(
                name             = "L2",
                medium           = "DDR",
                # 256 GB per NPU × 16 NPUs = 4 TB total shared pool.
                # Bandwidth: 64 GB/s per NPU × 16 NPUs = 1,024 GB/s aggregate.
                capacity_bytes   = 4 * 1024**4,           # 4 TB shared across 16 NPUs
                bandwidth_bps    = int(1024 * 1024**3),   # 1,024 GB/s aggregate (64 GB/s × 16)
                block_size_bytes = 32 * 1024**2,
                scope            = "per_worker",
                latency_floor_us = 200,
            ),
            TierSpec(
                name              = "L3A",
                medium            = "SSD",
                capacity_bytes    = 8 * 1024**4,         # 8 TB per worker, pooled globally
                bandwidth_bps     = int(25 * 1024**3),   # 25 GB/s
                block_size_bytes  = 256 * 1024**2,
                scope             = "global",            # pooled across all workers
                latency_floor_us  = 1_000,
                remote_latency_us = 50_000,              # 50ms cross-worker
            ),
        ),
        compute_tflops_bf16 = 640,
        interconnect_bps    = int(25 * 1024**3),         # 25 GB/s per NPU cross-node
        n_gpus_per_worker   = 16,
        confidence          = ConfidenceLabel.ANALYTICAL_ONLY,
        oracle_table        = None,
    ),
}


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
