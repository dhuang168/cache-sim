"""
Phase 0 contract tests — verify all interface boundaries are correctly defined.

These tests enforce the 7 guiding principles from agentsim_final_plan.md.
All must pass before Phase 0.5 begins.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from dataclasses import fields, FrozenInstanceError

from agentsim.core.contracts import (
    ConfidenceLabel,
    CacheKey,
    SavingsEvent,
    TierSpec,
    CacheObject,
    TransferRecord,
    RequestResult,
    CacheOracleBase,
    DispatcherBase,
    EvictionPolicyBase,
    DESEventKind,
    DESEvent,
    ObserverBase,
    SweepEstimator,
    ReportMetadata,
)


# ─── Principle 7: Every hardware target carries a ConfidenceLabel ───

class TestConfidenceLabel:
    def test_three_levels_exist(self):
        assert ConfidenceLabel.CALIBRATED.value == "calibrated"
        assert ConfidenceLabel.SEMI_CALIBRATED.value == "semi-calibrated"
        assert ConfidenceLabel.ANALYTICAL_ONLY.value == "analytical-only"

    def test_is_string_enum(self):
        """ConfidenceLabel must be string-based for inclusion in reports."""
        assert isinstance(ConfidenceLabel.CALIBRATED, str)


# ─── Principle 4: Cache identity = exact CacheKey match ───

class TestCacheKey:
    def test_frozen(self):
        ck = CacheKey(model_id="llama3_70b", tokenizer_id="default", prefix_hash="abc123")
        with pytest.raises(FrozenInstanceError):
            ck.model_id = "changed"

    def test_all_fields_required(self):
        """CacheKey requires model_id, tokenizer_id, prefix_hash."""
        ck = CacheKey(model_id="m", tokenizer_id="t", prefix_hash="h")
        assert ck.model_id == "m"
        assert ck.tokenizer_id == "t"
        assert ck.prefix_hash == "h"

    def test_equality_requires_all_fields(self):
        """Same model but different prefix_hash must not match."""
        ck1 = CacheKey(model_id="m", tokenizer_id="t", prefix_hash="hash1")
        ck2 = CacheKey(model_id="m", tokenizer_id="t", prefix_hash="hash2")
        assert ck1 != ck2

    def test_same_fields_are_equal(self):
        ck1 = CacheKey(model_id="m", tokenizer_id="t", prefix_hash="h")
        ck2 = CacheKey(model_id="m", tokenizer_id="t", prefix_hash="h")
        assert ck1 == ck2

    def test_hashable(self):
        """CacheKey must be usable as dict key."""
        ck = CacheKey(model_id="m", tokenizer_id="t", prefix_hash="h")
        d = {ck: "value"}
        assert d[ck] == "value"

    def test_no_token_count_field(self):
        """CacheKey must NOT have a token_count field — principle 4."""
        field_names = {f.name for f in fields(CacheKey)}
        assert "token_count" not in field_names
        assert "tokens" not in field_names


# ─── Savings classification ───

class TestSavingsEvent:
    def test_all_five_classes_exist(self):
        assert SavingsEvent.HIT_L1.value == "hit_l1"
        assert SavingsEvent.HIT_L2_WIN.value == "hit_l2_win"
        assert SavingsEvent.HIT_L3_WIN.value == "hit_l3_win"
        assert SavingsEvent.HIT_L3_BREAK_EVEN.value == "hit_l3_break_even"
        assert SavingsEvent.MISS_RECOMPUTE.value == "miss_recompute"

    def test_classify_miss(self):
        assert SavingsEvent.classify(None, 0, 100_000) == SavingsEvent.MISS_RECOMPUTE

    def test_classify_l1(self):
        assert SavingsEvent.classify("L1", 100, 100_000) == SavingsEvent.HIT_L1

    def test_classify_l2_win(self):
        """Transfer faster than recompute → WIN."""
        assert SavingsEvent.classify("L2", 5_000, 100_000) == SavingsEvent.HIT_L2_WIN

    def test_classify_l2_loss(self):
        """Transfer slower than recompute → treated as MISS."""
        assert SavingsEvent.classify("L2", 200_000, 100_000) == SavingsEvent.MISS_RECOMPUTE

    def test_classify_l3a_win(self):
        """Transfer well below recompute → WIN."""
        assert SavingsEvent.classify("L3A", 30_000, 100_000) == SavingsEvent.HIT_L3_WIN

    def test_classify_l3a_break_even(self):
        """Transfer close to recompute (within margin) → BREAK_EVEN."""
        assert SavingsEvent.classify("L3A", 95_000, 100_000) == SavingsEvent.HIT_L3_BREAK_EVEN

    def test_classify_l3a_loss(self):
        """Transfer much more expensive than recompute → MISS."""
        assert SavingsEvent.classify("L3A", 200_000, 100_000) == SavingsEvent.MISS_RECOMPUTE

    def test_classify_has_classmethod(self):
        """classify() must exist as a classmethod on SavingsEvent."""
        assert hasattr(SavingsEvent, "classify")
        assert callable(SavingsEvent.classify)


# ─── Principle 2: Byte-based accounting ───

class TestTierSpec:
    def test_frozen(self):
        ts = TierSpec(
            name="L1", medium="HBM", capacity_bytes=80*1024**3,
            bandwidth_bps=3_000_000_000_000, block_size_bytes=5120,
            scope="per_gpu", latency_floor_us=1,
        )
        with pytest.raises(FrozenInstanceError):
            ts.capacity_bytes = 0

    def test_no_token_fields(self):
        """TierSpec must have NO token-related fields — principle 2."""
        field_names = {f.name for f in fields(TierSpec)}
        token_fields = {n for n in field_names if "token" in n.lower()}
        assert token_fields == set(), f"Token fields found in TierSpec: {token_fields}"

    def test_all_sizes_in_bytes(self):
        """All capacity/bandwidth fields must be in bytes."""
        field_names = {f.name for f in fields(TierSpec)}
        assert "capacity_bytes" in field_names
        assert "bandwidth_bps" in field_names
        assert "block_size_bytes" in field_names

    def test_latency_in_microseconds(self):
        field_names = {f.name for f in fields(TierSpec)}
        assert "latency_floor_us" in field_names


class TestCacheObject:
    def test_frozen(self):
        ck = CacheKey(model_id="m", tokenizer_id="t", prefix_hash="h")
        co = CacheObject(
            object_id="s1:t1", cache_key=ck, size_bytes=1000,
            token_count=10, created_at_us=0, last_access_us=0,
            tier="L1", worker_id=0, gpu_id=0,
        )
        with pytest.raises(FrozenInstanceError):
            co.size_bytes = 999

    def test_size_bytes_is_authoritative(self):
        """size_bytes must exist and be the primary accounting field."""
        field_names = {f.name for f in fields(CacheObject)}
        assert "size_bytes" in field_names

    def test_token_count_is_metadata_only(self):
        """token_count exists but is labeled as metadata in docstring."""
        field_names = {f.name for f in fields(CacheObject)}
        assert "token_count" in field_names
        # Verify docstring says metadata
        assert "metadata" in CacheObject.__doc__.lower()

    def test_cache_key_present(self):
        """CacheObject must reference a CacheKey for lookup."""
        field_names = {f.name for f in fields(CacheObject)}
        assert "cache_key" in field_names


# ─── Principle 5: Prefill and decode tracked separately ───

class TestRequestResult:
    def test_prefill_and_decode_separated(self):
        """Principle 5: prefill_latency_us and decode_latency_us are separate fields."""
        field_names = {f.name for f in fields(RequestResult)}
        assert "prefill_latency_us" in field_names
        assert "decode_latency_us" in field_names
        assert "queue_wait_prefill_us" in field_names
        assert "queue_wait_decode_us" in field_names

    def test_savings_event_always_present(self):
        field_names = {f.name for f in fields(RequestResult)}
        assert "savings_event" in field_names

    def test_cache_key_always_present(self):
        field_names = {f.name for f in fields(RequestResult)}
        assert "cache_key" in field_names

    def test_total_latency_property(self):
        ck = CacheKey(model_id="m", tokenizer_id="t", prefix_hash="h")
        rr = RequestResult(
            session_id="s1", request_id="r1",
            prefill_latency_us=100, queue_wait_prefill_us=50, ttft_us=150,
            decode_latency_us=200, queue_wait_decode_us=30,
            cache_tier="L1", cache_key=ck, savings_event=SavingsEvent.HIT_L1,
            bytes_loaded=1000, latency_source="bandwidth",
        )
        assert rr.total_latency_us == 150 + 200 + 30


# ─── Principle 1: Core DES abstractions ───

class TestCoreAbstractions:
    def test_cache_oracle_is_abc(self):
        assert issubclass(CacheOracleBase, type) or hasattr(CacheOracleBase, '__abstractmethods__')
        assert "prefill_latency_us" in CacheOracleBase.__abstractmethods__
        assert "decode_latency_us" in CacheOracleBase.__abstractmethods__
        assert "kv_transfer_latency_us" in CacheOracleBase.__abstractmethods__

    def test_dispatcher_is_abc(self):
        assert "select_node" in DispatcherBase.__abstractmethods__

    def test_eviction_policy_is_abc(self):
        assert "select_eviction_candidates" in EvictionPolicyBase.__abstractmethods__


# ─── Principle 3: Observation layer is downstream-only ───

class TestObservationContract:
    def test_des_event_kind_has_all_types(self):
        expected = {
            "request_arrival", "prefill_start", "cache_lookup",
            "tier_transfer", "eviction", "prefill_complete",
            "decode_start", "decode_complete", "request_drop",
        }
        actual = {e.value for e in DESEventKind}
        assert expected == actual

    def test_des_event_has_sim_time(self):
        field_names = {f.name for f in fields(DESEvent)}
        assert "sim_time_us" in field_names
        assert "kind" in field_names
        assert "payload" in field_names

    def test_observer_is_abc(self):
        assert "on_event" in ObserverBase.__abstractmethods__

    def test_observer_has_assert_read_only(self):
        """ObserverBase provides _assert_read_only() helper."""
        assert hasattr(ObserverBase, "_assert_read_only")

    def test_observer_assert_read_only_raises(self):
        """_assert_read_only must raise NotImplementedError."""
        class TestObs(ObserverBase):
            def on_event(self, event):
                pass
        obs = TestObs()
        with pytest.raises(NotImplementedError, match="Observer"):
            obs._assert_read_only()


# ─── Principle 4 (SimPy): Sweep tool constraints ───

class TestSweepContract:
    def test_sweep_estimator_is_abc(self):
        assert "estimate_turn_latency_ms" in SweepEstimator.__abstractmethods__

    def test_sweep_docstring_lists_prohibitions(self):
        """SweepEstimator docstring must list what SimPy is prohibited from doing."""
        doc = SweepEstimator.__doc__
        assert "PROHIBITED" in doc
        assert "TierStore" in doc
        assert "EvictionPolicyBase" in doc


# ─── Principle 7: ReportMetadata with confidence in headline ───

class TestReportMetadata:
    def test_confidence_in_headline(self):
        rm = ReportMetadata(
            chip_name="nvidia_a100_80g", model_name="llama3_70b",
            confidence=ConfidenceLabel.CALIBRATED,
            oracle_source="benchmarks/a100_70b.json",
            l3a_mode="global", sim_duration_s=3600.0,
            generated_at="2026-03-23", agentsim_version="0.2.0",
        )
        assert "CALIBRATED" in rm.headline
        assert "nvidia_a100_80g" in rm.headline
        assert "llama3_70b" in rm.headline

    def test_analytical_only_headline(self):
        rm = ReportMetadata(
            chip_name="custom_npu", model_name="llama3_70b",
            confidence=ConfidenceLabel.ANALYTICAL_ONLY,
            oracle_source="roofline",
            l3a_mode="local", sim_duration_s=600.0,
            generated_at="2026-03-23", agentsim_version="0.2.0",
        )
        assert "ANALYTICAL-ONLY" in rm.headline

    def test_l3a_mode_in_headline(self):
        rm = ReportMetadata(
            chip_name="a100", model_name="m",
            confidence=ConfidenceLabel.CALIBRATED,
            oracle_source="table", l3a_mode="global",
            sim_duration_s=100, generated_at="now", agentsim_version="0.2",
        )
        assert "global" in rm.headline


# ─── TransferRecord ───

class TestTransferRecord:
    def test_has_all_fields(self):
        field_names = {f.name for f in fields(TransferRecord)}
        assert "bytes_moved" in field_names
        assert "latency_us" in field_names
        assert "latency_source" in field_names
        assert "savings_event" in field_names
        assert "cache_key" in field_names
