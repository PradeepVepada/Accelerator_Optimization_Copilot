from typing import Dict, List
import pandas as pd


class TensorMeta:
    def __init__(self, tensor_id: str, size_mb: int, last_access: int):
        self.tensor_id = tensor_id
        self.size_mb = size_mb
        self.last_access = last_access


class CacheState:
    def __init__(self, max_cache_mb: int = 64):
        self.max_cache_mb = max_cache_mb
        self.cache_contents: Dict[str, TensorMeta] = {}
        self.usage_order: List[str] = []

    def current_usage(self) -> int:
        return sum(t.size_mb for t in self.cache_contents.values())


class MemorySimulator:
    def __init__(self, cache_size_mb=64, dram_latency_per_mb=0.1, dram_bandwidth=32):
        self.cache = CacheState(cache_size_mb)
        self.dram_latency_per_mb = dram_latency_per_mb
        self.dram_bandwidth = dram_bandwidth
        self.metrics = {
            "cache_hits": 0,
            "dram_accesses": 0,
            "total_latency": 0.0,
            "evictions": 0,
            "bandwidth_wait_time": 0.0
        }

    def access_tensor(self, tensor_id: str, size_mb: int, access_idx: int, policy="LRU"):
        if tensor_id in self.cache.cache_contents:
            self.metrics["cache_hits"] += 1
            self.metrics["total_latency"] += 0.01
            self.cache.cache_contents[tensor_id].last_access = access_idx

            if policy == "LRU":
                self.cache.usage_order.remove(tensor_id)
                self.cache.usage_order.append(tensor_id)
        else:
            self.metrics["dram_accesses"] += 1
            fetch_time = size_mb * self.dram_latency_per_mb + size_mb / self.dram_bandwidth
            self.metrics["total_latency"] += fetch_time
            self.metrics["bandwidth_wait_time"] += size_mb / self.dram_bandwidth

            # Skip caching if tensor is larger than cache capacity
            if size_mb > self.cache.max_cache_mb:
                # Tensor too large to cache - just count as DRAM access
                return
            
            while self.cache.current_usage() + size_mb > self.cache.max_cache_mb:
                oldest = self.cache.usage_order.pop(0)
                del self.cache.cache_contents[oldest]
                self.metrics["evictions"] += 1

            meta = TensorMeta(tensor_id, size_mb, access_idx)
            self.cache.cache_contents[tensor_id] = meta
            self.cache.usage_order.append(tensor_id)

    def simulate_trace(self, trace_df: pd.DataFrame, policy="LRU"):
        for idx, row in trace_df.iterrows():
            self.access_tensor(row["tensor_id"], row["size_mb"], idx, policy)
        
        # Return metrics with units for better clarity
        return {
            "cache_hits (count)": self.metrics["cache_hits"],
            "dram_accesses (count)": self.metrics["dram_accesses"],
            "total_latency (ms)": round(self.metrics["total_latency"], 3),
            "evictions (count)": self.metrics["evictions"],
            "bandwidth_wait_time (ms)": round(self.metrics["bandwidth_wait_time"], 3),
            "cache_hit_rate (%)": round(
                100 * self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["dram_accesses"]), 
                2
            )
        }

