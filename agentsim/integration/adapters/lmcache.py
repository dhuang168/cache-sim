"""
LMCache Config Adapter — maps LMCache config to AgentSim tier specs.

Maps LMCache's tiered storage (GPU → CPU → Disk → Remote) to
AgentSim's L1/L2/L3A hierarchy with chunk-based deduplication.

Schema mapping only. Does not run LMCache.
"""
from __future__ import annotations
from typing import Optional

from agentsim.core.des.config import (
    SimConfig, TierConfig, CacheConfig, ModelConfig,
)


class LMCacheConfigAdapter:
    """
    Maps LMCache YAML/dict config to AgentSim SimConfig modifications.

    LMCache tiers map to:
      GPU memory      → L1 (managed by engine, not LMCache)
      local_cpu       → L2 (DRAM offload)
      local_disk      → L3A (SSD)
      remote (Redis)  → L3A global mode
    """

    def from_lmcache_config(
        self,
        lmc_config: dict,
        model: ModelConfig,
        base_config: Optional[SimConfig] = None,
    ) -> SimConfig:
        """
        Apply LMCache config to a SimConfig.

        Args:
            lmc_config: dict with LMCache parameters
            model: ModelConfig for KV size calculation
            base_config: SimConfig to modify (required)
        """
        if base_config is None:
            raise ValueError("LMCacheConfigAdapter requires a base_config")

        cfg = base_config
        bytes_per_token = (2 * model.n_layers * model.n_kv_heads * model.head_dim
                          * model.bytes_per_element)

        # Chunk size
        chunk_size_tokens = lmc_config.get("chunk_size", 256)

        # CPU offload tier (L2)
        if lmc_config.get("local_cpu", True):
            cpu_size_gb = lmc_config.get("max_local_cpu_size", 5.0)
            cfg.tiers[1].capacity_bytes = int(cpu_size_gb * 1024**3)

        # Disk tier (L3A)
        if lmc_config.get("local_disk", False):
            disk_path = lmc_config.get("local_disk_path", "/tmp/lmcache")
            # L3A stays at default capacity

        # Remote tier → global L3A mode
        if lmc_config.get("remote_url"):
            cfg.service.l3a_shared = True
            cfg.service.l3a_remote_latency_us = lmc_config.get(
                "remote_latency_us", 50_000
            )

        # Cache config: chunk dedup + LRU + demand-pull
        cfg.cache = CacheConfig(
            block_size_tokens=chunk_size_tokens,
            eviction_policy=lmc_config.get("eviction_policy", "lru"),
            deduplication="chunk",
            tier_migration="demand_pull",
        )

        cfg.run_id = f"lmcache-chunk{chunk_size_tokens}"
        return cfg

    def to_tier_specs(self, lmc_config: dict, model: ModelConfig) -> list[dict]:
        """Return LMCache tier mapping as documentation dict."""
        bytes_per_token = (2 * model.n_layers * model.n_kv_heads * model.head_dim
                          * model.bytes_per_element)
        chunk_bytes = lmc_config.get("chunk_size", 256) * bytes_per_token

        specs = []
        if lmc_config.get("local_cpu", True):
            specs.append({
                "lmcache_tier": "local_cpu",
                "agentsim_tier": "L2",
                "capacity_gb": lmc_config.get("max_local_cpu_size", 5.0),
                "chunk_bytes": chunk_bytes,
            })
        if lmc_config.get("local_disk", False):
            specs.append({
                "lmcache_tier": "local_disk",
                "agentsim_tier": "L3A (local)",
            })
        if lmc_config.get("remote_url"):
            specs.append({
                "lmcache_tier": "remote",
                "agentsim_tier": "L3A (global)",
            })
        return specs

    @staticmethod
    def mapping_documentation() -> dict:
        return {
            "lmcache.chunk_size": "CacheConfig.block_size_tokens",
            "lmcache.local_cpu": "L2 tier active",
            "lmcache.max_local_cpu_size": "L2 capacity_bytes",
            "lmcache.local_disk": "L3A tier active",
            "lmcache.remote_url": "L3A global mode (l3a_shared=True)",
            "lmcache.eviction_policy": "CacheConfig.eviction_policy",
            "implicit": "deduplication='chunk', tier_migration='demand_pull'",
            "caveat": "Schema mapping only — does not run LMCache",
        }
