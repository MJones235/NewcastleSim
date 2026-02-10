"""
Performance monitoring utilities for simulation profiling.

This module provides timing utilities to identify bottlenecks in the
hybrid Concordia + JuPedSim simulation.
"""

import time
from contextlib import contextmanager


class PerformanceTimer:
    """Simple performance timer for profiling simulation bottlenecks."""

    def __init__(self):
        self.timings = {}
        self.counts = {}
        self.parallel_operations = set()  # Track which operations run in parallel

    def record(self, name: str, duration: float, is_parallel: bool = False):
        """Record a timing measurement."""
        if name not in self.timings:
            self.timings[name] = 0.0
            self.counts[name] = 0

        if is_parallel:
            # For parallel operations, store max duration instead of sum
            self.parallel_operations.add(name)
            self.timings[name] = max(self.timings[name], duration)
            self.counts[name] += 1
        else:
            # For sequential operations, sum as normal
            self.timings[name] += duration
            self.counts[name] += 1

    @contextmanager
    def measure(self, name: str, is_parallel: bool = False):
        """Context manager for timing a block of code.

        Args:
            name: Name of the operation
            is_parallel: If True, uses max instead of sum (for parallel operations)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record(name, duration, is_parallel=is_parallel)

    def report(self) -> str:
        """Generate performance report."""
        if not self.timings:
            return "No timings recorded"

        lines = ["\n=== PERFORMANCE PROFILE (Wall-Clock Time) ==="]
        total_time = sum(self.timings.values())

        # Sort by total time descending
        sorted_items = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        for name, total in sorted_items:
            count = self.counts[name]
            avg = total / count if count > 0 else 0
            percent = (total / total_time * 100) if total_time > 0 else 0

            # Add indicator for parallel operations
            parallel_mark = " [parallel]" if name in self.parallel_operations else ""

            lines.append(
                f"{name:30s}: {total:8.3f}s total | {avg:8.3f}s avg | "
                f"{count:5d} calls | {percent:5.1f}%{parallel_mark}"
            )

        lines.append(f"{'TOTAL':30s}: {total_time:8.3f}s")
        lines.append("=" * 80)

        return "\n".join(lines)
