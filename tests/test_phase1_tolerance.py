"""
Phase 1 tolerance tests — verify NEW agentsim engine matches golden baselines.

Same scenarios as test_phase05_baseline.py but using agentsim.core.des.engine.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from agentsim.core.des.config import SimConfig
from agentsim.core.des.engine import SimEngine

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "configs" / "heavy_coding.json")
GOLDEN_PATH = Path(__file__).resolve().parent.parent / "results" / "golden" / "phase05_reference.json"

PEAK = 15
SIM_DUR = 300.0
WARMUP = 30.0


def _load_golden():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _heavy_config(n_workers: int, l3a_shared: bool = True) -> SimConfig:
    cfg = SimConfig.from_json(CONFIG_PATH)
    cfg.sim_duration_s = SIM_DUR
    cfg.warmup_s = WARMUP
    cfg.epoch_report_interval_s = 30.0
    cfg.sim_start_time_s = 36000.0
    cfg.service.n_prefill_nodes = n_workers * 8
    cfg.service.n_gpus_per_worker = 8
    cfg.service.l3a_shared = l3a_shared
    cfg.service.dispatch_algorithm = "pull"
    for p in cfg.profiles:
        p.arrival_rate_peak = PEAK
    return cfg


def _run_and_extract(cfg):
    m = SimEngine(cfg).run()
    r = m.report()
    completed = sum(m.savings_events.values())
    hit = r.get("cache_hit_rate", {})
    combined_hit = 1.0 - hit.get("miss", 1.0)
    all_ttft = []
    for v in m.ttft_us.values():
        all_ttft.extend(v)
    return {
        "completed": completed,
        "combined_hit": combined_hit,
        "ttft_p50_s": np.percentile(all_ttft, 50) / 1e6 if all_ttft else 0,
        "recompute": np.mean(m.recompute_fraction) if m.recompute_fraction else 0,
    }


class TestNewEngineScenarioA:
    """Verify new engine matches Scenario A golden baseline."""

    @pytest.fixture(scope="class")
    def result(self):
        return _run_and_extract(_heavy_config(1, True))

    def test_completed_matches_golden(self, result):
        golden = _load_golden()["scenario_a"]
        assert result["completed"] == golden["completed"]

    def test_hit_rate_matches_golden(self, result):
        golden = _load_golden()
        ref = golden["scenario_a"]["combined_hit_rate"]
        tol = golden["tolerances"]["hit_rate_absolute"]
        assert abs(result["combined_hit"] - ref) <= tol

    def test_ttft_p50_within_tolerance(self, result):
        golden = _load_golden()
        ref = golden["scenario_a"]["ttft_p50_s"]
        tol = golden["tolerances"]["ttft_p50_relative"]
        assert abs(result["ttft_p50_s"] - ref) / max(ref, 0.1) <= tol


class TestNewEngineScenarioB:
    """Verify new engine global vs local at 4 workers."""

    @pytest.fixture(scope="class")
    def global_result(self):
        return _run_and_extract(_heavy_config(4, True))

    @pytest.fixture(scope="class")
    def local_result(self):
        return _run_and_extract(_heavy_config(4, False))

    def test_both_high_hit(self, global_result, local_result):
        assert global_result["combined_hit"] >= 0.99
        assert local_result["combined_hit"] >= 0.99

    def test_completed_matches(self, global_result, local_result):
        golden = _load_golden()["scenario_a"]
        assert global_result["completed"] == golden["completed"]
        assert local_result["completed"] == golden["completed"]


class TestNewEngineStress:
    """Stress scenario: 4W peak=5 — queue saturation plausibility."""

    def test_stress_runs_and_completes(self):
        cfg = _heavy_config(4, True)
        cfg.sim_duration_s = 120.0
        cfg.warmup_s = 15.0
        for p in cfg.profiles:
            p.arrival_rate_peak = 5
        m = SimEngine(cfg).run()
        completed = sum(m.savings_events.values())
        assert completed > 0, "No requests completed under stress"
