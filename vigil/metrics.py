"""Instrumentation. Every number in the results section originates here.

One CSV row per completed chunk, written as the run proceeds so a crash at
hour five does not destroy five hours of data. Add instrumentation before you
need it; adding it afterwards means rerunning every experiment.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger("vigil.metrics")

CSV_FIELDS = [
    "chunk_index", "wall_t", "stream_t", "frames_seen", "frames_kept",
    "frame_retention_cum", "speech_ms", "peak_rms", "escalated", "tau",
    "images_sent", "payload_bytes", "tokens_in", "tokens_out", "tokens_cum",
    "latency_s", "queue_depth", "retries", "rate_limit_faults", "error",
    "description",
]


class RateLimiter:
    """Sliding-window limiter. Keeps calls under the provider's quota."""

    def __init__(self, max_calls: int, window_s: float):
        self.max_calls = max(int(max_calls), 1)
        self.window_s = float(window_s)
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        waited = 0.0
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > self.window_s:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return waited
                sleep_for = self.window_s - (now - self._calls[0]) + 0.01
                waited += sleep_for
                await asyncio.sleep(sleep_for)


@dataclass
class RunMetrics:
    run_id: str = "dev"
    started_at: float = field(default_factory=time.time)
    chunks_done: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    errors: int = 0
    rate_limit_faults: int = 0
    retries: int = 0
    latencies: list = field(default_factory=list)
    max_queue_depth: int = 0
    tokens_by_hour: dict = field(default_factory=dict)

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def record(self, tokens_in: int, tokens_out: int, latency: float,
               queue_depth: int, stream_t: float) -> None:
        self.chunks_done += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.latencies.append(latency)
        self.max_queue_depth = max(self.max_queue_depth, queue_depth)
        hour = int(stream_t // 3600)
        self.tokens_by_hour[hour] = (self.tokens_by_hour.get(hour, 0)
                                     + tokens_in + tokens_out)

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = min(int(len(s) * p / 100.0), len(s) - 1)
        return s[idx]

    def tokens_per_hour(self) -> float:
        hours = self.elapsed_s / 3600.0
        return self.tokens_total / hours if hours > 0 else 0.0

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id,
            "chunks_done": self.chunks_done,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_total,
            "tokens_per_hour": round(self.tokens_per_hour(), 1),
            "tokens_by_hour": self.tokens_by_hour,
            "p50_latency_s": round(self.percentile(50), 3),
            "p95_latency_s": round(self.percentile(95), 3),
            "max_queue_depth": self.max_queue_depth,
            "errors": self.errors,
            "rate_limit_faults": self.rate_limit_faults,
            "retries": self.retries,
            "elapsed_s": round(self.elapsed_s, 1),
        }


class CsvLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS,
                                 extrasaction="ignore")
        self._w.writeheader()
        self._fh.flush()

    def write(self, row: dict) -> None:
        try:
            self._w.writerow(row)
            self._fh.flush()   # survive a crash mid-run
        except Exception:
            log.exception("csv write failed")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def write_manifest(path: str, cfg, metrics: RunMetrics, gate_stats,
                   extra: dict | None = None) -> None:
    """One JSON per run describing exactly what produced the CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    doc = {
        "run_id": cfg.run_id,
        "config": cfg.to_dict(),
        "metrics": metrics.snapshot(),
        "gate": gate_stats.snapshot() if gate_stats else {},
    }
    if extra:
        doc["extra"] = extra
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
