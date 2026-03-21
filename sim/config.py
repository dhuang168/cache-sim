from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class TierConfig:
    name: str
    capacity_bytes: int
    bandwidth_bytes_per_s: int
    latency_floor_us: int
    block_size_bytes: int


@dataclass
class ModelConfig:
    model_id: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    bytes_per_element: int  # 2 = FP16, 1 = FP8


@dataclass
class WorkloadProfile:
    name: str
    arrival_rate_peak: float
    diurnal_peak_trough_ratio: float
    iat_mean_s: float
    iat_dist: str
    input_len_mean_tokens: int
    input_len_sigma_tokens: int
    output_len_pareto_alpha: float
    output_len_pareto_xmin: int
    context_growth_min_tokens: int
    context_growth_max_tokens: int
    prefix_stability_initial: float
    prefix_stability_final: float
    session_duration_mean_s: float
    session_duration_dist: str
    shared_system_prefix_tokens: int


@dataclass
class ServiceConfig:
    n_prefill_slots: int
    n_decode_slots: int
    prefill_queue_max: int
    decode_queue_max: int
    n_prefill_nodes: int = 1
    dispatch_algorithm: str = "push"
    inter_node_latency_us: int = 5
    inter_node_bandwidth_bytes_per_s: int = 100_000_000_000  # 100 GB/s NVLink
    l3a_shared: bool = True
    l3a_remote_latency_us: int = 50_000  # 50ms for remote/global L3A access


@dataclass
class SimConfig:
    run_id: str
    seed: int
    tiers: list[TierConfig]
    model: ModelConfig
    profiles: list[WorkloadProfile]
    profile_mix: dict[str, float]
    service: ServiceConfig
    ttl_l2_s: float
    ttl_l3a_s: float
    eviction_hbm_threshold: float
    eviction_ram_threshold: float
    sim_duration_s: float
    warmup_s: float
    epoch_report_interval_s: float
    enable_suffix_cache: bool = False
    enable_l3b_object_store: bool = False

    @classmethod
    def from_json(cls, path: str) -> SimConfig:
        with open(path) as f:
            raw = json.load(f)
        # Reconstruct nested dataclasses
        raw["tiers"] = [TierConfig(**t) for t in raw["tiers"]]
        raw["model"] = ModelConfig(**raw["model"])
        raw["profiles"] = [WorkloadProfile(**p) for p in raw["profiles"]]
        raw["service"] = ServiceConfig(**raw["service"])
        return cls(**raw)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
