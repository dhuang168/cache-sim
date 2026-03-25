"""
vLLM Config Adapter — maps vLLM serve config to AgentSim SimConfig.

Schema mapping only. Does not run vLLM. Maps scheduling semantics
to DES equivalents for what-if analysis.

Mapping:
  vllm.block_size         → TierSpec.block_size_bytes (L1)
  vllm.max_num_seqs       → ServiceConfig.n_prefill_slots
  vllm.prefix_caching     → CacheConfig.eviction_policy = "lru"
  vllm.enable_chunked_prefill → scheduling hint (metadata only)
  vllm.gpu_memory_utilization → L1 capacity_bytes scaling
"""
from __future__ import annotations
from typing import Optional

from agentsim.core.des.config import (
    SimConfig, TierConfig, ModelConfig, ServiceConfig, CacheConfig,
    WorkloadProfile,
)


class VLLMConfigAdapter:
    """
    Reads a vLLM-style config dict and produces an AgentSim SimConfig.

    This is schema-level mapping — it translates vLLM concepts to
    their DES simulation equivalents. It does not guarantee runtime
    fidelity with actual vLLM behavior.
    """

    # Default vLLM values
    DEFAULT_BLOCK_SIZE = 16  # tokens
    DEFAULT_MAX_NUM_SEQS = 256
    DEFAULT_GPU_MEM_UTIL = 0.90

    def from_vllm_config(
        self,
        vllm_config: dict,
        model: ModelConfig,
        base_config: Optional[SimConfig] = None,
    ) -> SimConfig:
        """
        Convert vLLM config dict to SimConfig.

        Args:
            vllm_config: dict with vLLM serve parameters
            model: ModelConfig for KV size calculation
            base_config: optional base SimConfig to override (uses defaults if None)
        """
        block_size_tokens = vllm_config.get("block_size", self.DEFAULT_BLOCK_SIZE)
        max_num_seqs = vllm_config.get("max_num_seqs", self.DEFAULT_MAX_NUM_SEQS)
        gpu_mem_util = vllm_config.get("gpu_memory_utilization", self.DEFAULT_GPU_MEM_UTIL)
        prefix_caching = vllm_config.get("enable_prefix_caching", False)
        chunked_prefill = vllm_config.get("enable_chunked_prefill", False)
        tp_size = vllm_config.get("tensor_parallel_size", 1)

        # Compute bytes per token for block size conversion
        bytes_per_token = (2 * model.n_layers * model.n_kv_heads * model.head_dim
                          * model.bytes_per_element)
        block_size_bytes = block_size_tokens * bytes_per_token

        # L1 capacity scaled by gpu_memory_utilization
        # Default A100 80GB, scaled by utilization
        gpu_memory_bytes = vllm_config.get("gpu_memory_bytes", 80 * 1024**3)
        l1_capacity = int(gpu_memory_bytes * gpu_mem_util)

        # Build tier config
        tiers = [
            TierConfig(
                name="L1",
                capacity_bytes=l1_capacity,
                bandwidth_bytes_per_s=3_000_000_000_000,  # 3 TB/s HBM
                latency_floor_us=1,
                block_size_bytes=block_size_bytes,
            ),
            TierConfig(
                name="L2",
                capacity_bytes=vllm_config.get("cpu_offload_gb", 0) * 1024**3 or 1024**4,
                bandwidth_bytes_per_s=64_000_000_000,  # 64 GB/s DRAM
                latency_floor_us=10,
                block_size_bytes=33554432,
            ),
            TierConfig(
                name="L3A",
                capacity_bytes=8 * 1024**4,  # 8 TB SSD default
                bandwidth_bytes_per_s=7_000_000_000,
                latency_floor_us=100,
                block_size_bytes=268435456,
            ),
        ]

        service = ServiceConfig(
            n_prefill_slots=max_num_seqs // tp_size,
            n_decode_slots=max_num_seqs,
            prefill_queue_max=max_num_seqs * 2,
            decode_queue_max=max_num_seqs * 4,
        )

        cache = CacheConfig(
            block_size_tokens=block_size_tokens,
            eviction_policy="lru" if prefix_caching else "ttl",
        )

        # Build SimConfig from base or defaults
        if base_config:
            cfg = base_config
            cfg.tiers = tiers
            cfg.model = model
            cfg.service = service
            cfg.cache = cache
        else:
            cfg = SimConfig(
                run_id=f"vllm-{model.model_id}",
                seed=42,
                tiers=tiers,
                model=model,
                profiles=[],  # caller must set profiles
                profile_mix={},
                service=service,
                ttl_l2_s=300.0,
                ttl_l3a_s=3600.0,
                eviction_hbm_threshold=0.9,
                eviction_ram_threshold=0.95,
                sim_duration_s=600.0,
                warmup_s=60.0,
                epoch_report_interval_s=30.0,
                cache=cache,
            )

        # Store vLLM-specific metadata for reporting
        cfg.run_id = f"vllm-{'chunked' if chunked_prefill else 'orca'}-{model.model_id}"

        return cfg

    @staticmethod
    def mapping_documentation() -> dict:
        """Return the schema mapping for documentation."""
        return {
            "vllm.block_size": "TierSpec.block_size_bytes (L1) = block_size × bytes_per_token",
            "vllm.max_num_seqs": "ServiceConfig.n_prefill_slots",
            "vllm.enable_prefix_caching": "CacheConfig.eviction_policy = 'lru'",
            "vllm.enable_chunked_prefill": "metadata only (scheduling hint)",
            "vllm.gpu_memory_utilization": "L1 capacity_bytes = gpu_memory × utilization",
            "vllm.tensor_parallel_size": "n_prefill_slots = max_num_seqs / tp_size",
            "caveat": "Schema mapping only — does not guarantee runtime fidelity with actual vLLM",
        }
