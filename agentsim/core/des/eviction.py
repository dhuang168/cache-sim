from __future__ import annotations
import heapq
from typing import Optional

from agentsim.core.des.config import SimConfig
from agentsim.core.des.cache import CacheObject, TierStore, Tier
from agentsim.core.contracts import EvictionPolicyBase


class EvictionEngine:
    """
    Manages tier-to-tier movement of CacheObjects.

    Policy:
      L1 -> L2: triggered when L1 occupancy > eviction_hbm_threshold.
                Evict leaf nodes first. Among leaves, prefer ref_count==1
                and oldest last_accessed_at. Never evict shared-prefix nodes
                before private ones.

      L2 -> L3a: triggered by TTL or when L2 occupancy > eviction_ram_threshold.

      L3a cleanup: triggered when L3a occupancy > 0.90. Evict by LRU.
    """

    def __init__(self, config: SimConfig, tier_stores: dict[Tier, TierStore]):
        self.config = config
        self.stores = tier_stores

    def needs_l1_eviction(self) -> bool:
        store = self.stores[Tier.L1]
        return store.occupancy_pct > self.config.eviction_hbm_threshold

    def needs_l2_eviction(self) -> bool:
        store = self.stores[Tier.L2]
        return store.occupancy_pct > self.config.eviction_ram_threshold

    def needs_l3a_cleanup(self) -> bool:
        store = self.stores[Tier.L3A]
        return store.occupancy_pct > 0.90

    def evict_l1_to_l2(self, n_bytes_needed: int) -> list[str]:
        """
        Evict enough L1 CacheObjects to free n_bytes_needed.
        Returns list of evicted cache_keys.
        """
        l1 = self.stores[Tier.L1]
        l2 = self.stores[Tier.L2]

        # Use heapq.nsmallest to avoid full sort — only need enough to free n_bytes
        # Estimate how many objects we need (upper bound)
        avg_size = max(1, l1.used_bytes // max(1, len(l1.objects)))
        est_count = min(len(l1.objects), max(4, (n_bytes_needed // avg_size) * 2 + 4))
        candidates = heapq.nsmallest(
            est_count, l1.objects.items(),
            key=lambda kv: (kv[1].ref_count > 1, kv[1].last_accessed_at_us),
        )

        evicted = []
        freed = 0
        for key, obj in candidates:
            if freed >= n_bytes_needed:
                break
            alloc = obj.block_count * l1.block_size_bytes
            # Move to L2
            l1.remove(key)
            freed += alloc

            # Recompute blocks for L2 block size
            from agentsim.core.des.cache import allocated_blocks, TIER_TO_LAYOUT
            new_blocks = allocated_blocks(obj.size_bytes, l2.block_size_bytes)
            obj.tier = Tier.L2
            obj.block_count = new_blocks
            obj.block_layout = TIER_TO_LAYOUT[Tier.L2]

            if l2.can_fit(obj.size_bytes):
                l2.insert(key, obj)
            else:
                # L2 full — hibernate directly to L3A
                self.hibernate_l2_to_l3a_obj(key, obj)
            evicted.append(key)

        return evicted

    def hibernate_l2_to_l3a_obj(self, cache_key: str, obj) -> bool:
        """Move an already-removed object directly to L3A. Used when L2 is full."""
        l3a = self.stores[Tier.L3A]
        from agentsim.core.des.cache import allocated_blocks, TIER_TO_LAYOUT
        new_blocks = allocated_blocks(obj.size_bytes, l3a.block_size_bytes)
        obj.tier = Tier.L3A
        obj.block_count = new_blocks
        obj.block_layout = TIER_TO_LAYOUT[Tier.L3A]
        obj.is_hibernated = True

        if l3a.can_fit(obj.size_bytes):
            l3a.insert(cache_key, obj)
            return True
        cleaned = self.cleanup_l3a(new_blocks * l3a.block_size_bytes)
        if l3a.can_fit(obj.size_bytes):
            l3a.insert(cache_key, obj)
            return True
        return False  # object lost

    def hibernate_l2_to_l3a(self, cache_key: str) -> bool:
        """Move CacheObject from L2 to L3a storage. Returns True if successful."""
        l2 = self.stores[Tier.L2]
        l3a = self.stores[Tier.L3A]

        obj = l2.remove(cache_key)
        if obj is None:
            return False

        from agentsim.core.des.cache import allocated_blocks, TIER_TO_LAYOUT
        new_blocks = allocated_blocks(obj.size_bytes, l3a.block_size_bytes)
        obj.tier = Tier.L3A
        obj.block_count = new_blocks
        obj.block_layout = TIER_TO_LAYOUT[Tier.L3A]
        obj.is_hibernated = True

        if l3a.can_fit(obj.size_bytes):
            l3a.insert(cache_key, obj)
            return True
        else:
            # Need to clean up L3a first
            cleaned = self.cleanup_l3a(new_blocks * l3a.block_size_bytes)
            if l3a.can_fit(obj.size_bytes):
                l3a.insert(cache_key, obj)
                return True
            # Cannot fit even after cleanup — object is lost
            return False

    def cleanup_l3a(self, n_bytes_needed: int) -> list[str]:
        """Evict LRU objects from L3a. Returns permanently lost cache_keys."""
        l3a = self.stores[Tier.L3A]

        candidates = sorted(
            l3a.objects.items(),
            key=lambda kv: kv[1].last_accessed_at_us,
        )

        evicted = []
        freed = 0
        for key, obj in candidates:
            if freed >= n_bytes_needed:
                break
            alloc = obj.block_count * l3a.block_size_bytes
            l3a.remove(key)
            freed += alloc
            evicted.append(key)

        return evicted

    def find_ttl_expired_l2(self, current_time_us: int) -> list[str]:
        """Find L2 objects whose TTL has expired."""
        l2 = self.stores[Tier.L2]
        ttl_us = int(self.config.ttl_l2_s * 1_000_000)
        expired = []
        for key, obj in list(l2.objects.items()):
            if current_time_us - obj.last_accessed_at_us >= ttl_us:
                expired.append(key)
        return expired

    def find_ttl_expired_l3a(self, current_time_us: int) -> list[str]:
        """Find L3a objects whose TTL has expired."""
        l3a = self.stores[Tier.L3A]
        ttl_us = int(self.config.ttl_l3a_s * 1_000_000)
        expired = []
        for key, obj in list(l3a.objects.items()):
            if current_time_us - obj.last_accessed_at_us >= ttl_us:
                expired.append(key)
        return expired


class LRUEvictionPolicy(EvictionPolicyBase):
    """Contract-compliant wrapper around EvictionEngine's candidate selection."""

    def __init__(self, eviction_engine: EvictionEngine, tier_key: Tier):
        self._engine = eviction_engine
        self._tier_key = tier_key

    def select_eviction_candidates(self, tier, needed_bytes: int) -> list:
        """Select LRU candidates from the given tier."""
        store = self._engine.stores[self._tier_key]
        candidates = sorted(
            store.objects.items(),
            key=lambda kv: (kv[1].ref_count > 1, kv[1].last_accessed_at_us),
        )
        result = []
        freed = 0
        for key, obj in candidates:
            if freed >= needed_bytes:
                break
            freed += obj.block_count * store.block_size_bytes
            result.append(obj)
        return result
