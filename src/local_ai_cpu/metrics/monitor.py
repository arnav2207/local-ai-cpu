"""CPU and memory sampling during inference."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ResourceMetrics:
    duration_seconds: float
    peak_cpu_percent: float
    peak_rss_mb: float


class ResourceMonitor:
    """Sample process CPU and memory usage while a block of work runs."""

    def __init__(self, sample_interval_seconds: float = 0.1) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self.metrics: ResourceMetrics | None = None
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start = 0.0
        self._peak_cpu = 0.0
        self._peak_rss = 0

    def __enter__(self) -> ResourceMonitor:
        self._start = time.perf_counter()
        self._peak_cpu = 0.0
        self._peak_rss = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        duration = time.perf_counter() - self._start
        self.metrics = ResourceMetrics(
            duration_seconds=duration,
            peak_cpu_percent=self._peak_cpu,
            peak_rss_mb=self._peak_rss / (1024 * 1024),
        )

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                cpu = self._process.cpu_percent(interval=None)
                rss = self._process.memory_info().rss
            except psutil.Error:
                break

            self._peak_cpu = max(self._peak_cpu, cpu)
            self._peak_rss = max(self._peak_rss, rss)
            time.sleep(self.sample_interval_seconds)
