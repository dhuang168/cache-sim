"""
SGLang Config Adapter — maps SGLang config to AgentSim SimConfig.

Key mapping: SGLang RadixAttention → prefix-aware LRU eviction.
SGLang's radix tree evicts leaf nodes first (sequence ends),
preserving shared prefixes — modeled as tail_first chunk eviction.

Schema mapping only. Does not run SGLang.
"""
from __future__ import annotations
from typing import Optional

from agentsim.core.des.config import (
    SimConfig, CacheConfig, ModelConfig,
)


class SGLangConfigAdapter:
    """
    Maps SGLang config to AgentSim SimConfig.

    SGLang concepts:
      RadixAttention     → chunk dedup + tail_first eviction (preserves prefixes)
      chunked_prefill    → scheduling hint (metadata)
      tp_size            → n_prefill_slots scaling
      mem_fraction_static → L1 capacity scaling
    """

    def from_sglang_config(
        self,
        sglang_config: dict,
        model: ModelConfig,
        base_config: Optional[SimConfig] = None,
    ) -> SimConfig:
        if base_config is None:
            raise ValueError("SGLangConfigAdapter requires a base_config")

        cfg = base_config
        bytes_per_token = (2 * model.n_layers * model.n_kv_heads * model.head_dim
                          * model.bytes_per_element)

        # RadixAttention → prefix-aware eviction
        radix_attention = sglang_config.get("radix_attention", True)

        # Memory fraction for KV cache
        mem_fraction = sglang_config.get("mem_fraction_static", 0.80)
        gpu_memory = sglang_config.get("gpu_memory_bytes", 80 * 1024**3)
        cfg.tiers[0].capacity_bytes = int(gpu_memory * mem_fraction)

        # Scheduling
        max_running = sglang_config.get("max_running_requests", 256)
        tp_size = sglang_config.get("tp_size", 1)
        cfg.service.n_prefill_slots = max_running // tp_size
        cfg.service.n_decode_slots = max_running

        # Cache config: RadixAttention = chunk dedup + tail_first
        if radix_attention:
            block_size = sglang_config.get("page_size", 1)  # SGLang uses token-level by default
            cfg.cache = CacheConfig(
                block_size_tokens=max(block_size, 16),  # minimum 16 for simulation feasibility
                eviction_policy="lru",
                deduplication="chunk",
                tier_migration="demand_pull",
                chunk_eviction="tail_first",  # SGLang-style: evict leaves first
            )
        else:
            cfg.cache = CacheConfig(
                eviction_policy="lru",
            )

        cfg.run_id = f"sglang-{'radix' if radix_attention else 'basic'}-{model.model_id}"
        return cfg

    @staticmethod
    def mapping_documentation() -> dict:
        return {
            "sglang.radix_attention": "chunk dedup + tail_first eviction (preserves shared prefixes)",
            "sglang.mem_fraction_static": "L1 capacity_bytes = gpu_memory × fraction",
            "sglang.max_running_requests": "ServiceConfig.n_prefill_slots / n_decode_slots",
            "sglang.tp_size": "n_prefill_slots = max_running / tp_size",
            "sglang.page_size": "CacheConfig.block_size_tokens (minimum 16)",
            "caveat": "Schema mapping only — does not run SGLang. RadixAttention modeled as "
                      "tail_first chunk eviction, not actual radix tree.",
        }
