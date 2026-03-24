"""
Phase 4 — Framework adapter tests.

Verifies:
1. VLLMConfigAdapter converts config without error
2. LMCacheConfigAdapter produces correct tier specs
3. SGLangConfigAdapter maps RadixAttention to tail_first
4. Adapted configs run through the DES engine
5. Schema mapping documentation exists
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from agentsim.core.des.config import SimConfig, ModelConfig
from agentsim.core.des.engine import SimEngine
from agentsim.integration.adapters.vllm import VLLMConfigAdapter
from agentsim.integration.adapters.lmcache import LMCacheConfigAdapter
from agentsim.integration.adapters.sglang import SGLangConfigAdapter

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "heavy_coding.json")

MODEL = ModelConfig(
    model_id="llama3-70b",
    n_layers=80,
    n_kv_heads=8,
    head_dim=128,
    bytes_per_element=2,
)


def _base_config() -> SimConfig:
    cfg = SimConfig.from_json(CONFIG_PATH)
    cfg.sim_duration_s = 30.0
    cfg.warmup_s = 5.0
    cfg.sim_start_time_s = 36000.0
    for p in cfg.profiles:
        p.arrival_rate_peak = 15
    return cfg


# ─── vLLM Adapter ───

class TestVLLMAdapter:

    def test_basic_conversion(self):
        """VLLMConfigAdapter converts a basic config dict."""
        adapter = VLLMConfigAdapter()
        cfg = adapter.from_vllm_config(
            {"block_size": 16, "max_num_seqs": 128, "enable_prefix_caching": True},
            model=MODEL,
        )
        assert cfg.service.n_prefill_slots == 128
        assert cfg.cache.eviction_policy == "lru"
        assert cfg.cache.block_size_tokens == 16

    def test_chunked_prefill_flag(self):
        """Chunked prefill shows in run_id."""
        adapter = VLLMConfigAdapter()
        cfg = adapter.from_vllm_config(
            {"enable_chunked_prefill": True},
            model=MODEL,
        )
        assert "chunked" in cfg.run_id

    def test_gpu_memory_scaling(self):
        """GPU memory utilization scales L1 capacity."""
        adapter = VLLMConfigAdapter()
        cfg = adapter.from_vllm_config(
            {"gpu_memory_utilization": 0.5, "gpu_memory_bytes": 80 * 1024**3},
            model=MODEL,
        )
        assert cfg.tiers[0].capacity_bytes == int(80 * 1024**3 * 0.5)

    def test_vllm_config_runs(self):
        """vLLM-adapted config runs through engine."""
        adapter = VLLMConfigAdapter()
        base = _base_config()
        cfg = adapter.from_vllm_config(
            {"block_size": 16, "max_num_seqs": 32, "enable_prefix_caching": True},
            model=MODEL,
            base_config=base,
        )
        m = SimEngine(cfg).run()
        assert sum(m.savings_events.values()) > 0

    def test_mapping_documentation(self):
        assert "caveat" in VLLMConfigAdapter.mapping_documentation()


# ─── LMCache Adapter ───

class TestLMCacheAdapter:

    def test_basic_conversion(self):
        """LMCacheConfigAdapter sets chunk dedup + demand pull."""
        adapter = LMCacheConfigAdapter()
        base = _base_config()
        cfg = adapter.from_lmcache_config(
            {"chunk_size": 256, "local_cpu": True, "max_local_cpu_size": 10.0},
            model=MODEL,
            base_config=base,
        )
        assert cfg.cache.deduplication == "chunk"
        assert cfg.cache.tier_migration == "demand_pull"
        assert cfg.cache.block_size_tokens == 256
        assert cfg.tiers[1].capacity_bytes == int(10.0 * 1024**3)

    def test_remote_enables_global_l3a(self):
        """Remote storage enables global L3A mode."""
        adapter = LMCacheConfigAdapter()
        base = _base_config()
        cfg = adapter.from_lmcache_config(
            {"remote_url": "redis://localhost", "remote_latency_us": 100_000},
            model=MODEL,
            base_config=base,
        )
        assert cfg.service.l3a_shared is True
        assert cfg.service.l3a_remote_latency_us == 100_000

    def test_lmcache_config_runs(self):
        """LMCache-adapted config runs through engine."""
        adapter = LMCacheConfigAdapter()
        base = _base_config()
        cfg = adapter.from_lmcache_config(
            {"chunk_size": 256, "local_cpu": True, "max_local_cpu_size": 5.0},
            model=MODEL,
            base_config=base,
        )
        m = SimEngine(cfg).run()
        assert sum(m.savings_events.values()) > 0
        assert m.chunk_total_logical > 0  # chunk mode active

    def test_tier_specs_documentation(self):
        adapter = LMCacheConfigAdapter()
        specs = adapter.to_tier_specs(
            {"chunk_size": 256, "local_cpu": True, "max_local_cpu_size": 5.0},
            model=MODEL,
        )
        assert len(specs) > 0
        assert specs[0]["agentsim_tier"] == "L2"

    def test_mapping_documentation(self):
        assert "caveat" in LMCacheConfigAdapter.mapping_documentation()


# ─── SGLang Adapter ───

class TestSGLangAdapter:

    def test_radix_attention_mapping(self):
        """RadixAttention maps to chunk dedup + tail_first eviction."""
        adapter = SGLangConfigAdapter()
        base = _base_config()
        cfg = adapter.from_sglang_config(
            {"radix_attention": True, "max_running_requests": 128},
            model=MODEL,
            base_config=base,
        )
        assert cfg.cache.deduplication == "chunk"
        assert cfg.cache.chunk_eviction == "tail_first"
        assert cfg.cache.tier_migration == "demand_pull"
        assert cfg.service.n_prefill_slots == 128

    def test_no_radix_basic_mode(self):
        """Without RadixAttention, uses basic LRU."""
        adapter = SGLangConfigAdapter()
        base = _base_config()
        cfg = adapter.from_sglang_config(
            {"radix_attention": False},
            model=MODEL,
            base_config=base,
        )
        assert cfg.cache.eviction_policy == "lru"
        assert cfg.cache.deduplication == "per_session"

    def test_sglang_config_runs(self):
        """SGLang-adapted config runs through engine."""
        adapter = SGLangConfigAdapter()
        base = _base_config()
        cfg = adapter.from_sglang_config(
            {"radix_attention": True, "max_running_requests": 32},
            model=MODEL,
            base_config=base,
        )
        m = SimEngine(cfg).run()
        assert sum(m.savings_events.values()) > 0

    def test_memory_fraction_scaling(self):
        """mem_fraction_static scales L1 capacity."""
        adapter = SGLangConfigAdapter()
        base = _base_config()
        cfg = adapter.from_sglang_config(
            {"mem_fraction_static": 0.5, "gpu_memory_bytes": 80 * 1024**3},
            model=MODEL,
            base_config=base,
        )
        assert cfg.tiers[0].capacity_bytes == int(80 * 1024**3 * 0.5)

    def test_mapping_documentation(self):
        assert "caveat" in SGLangConfigAdapter.mapping_documentation()
