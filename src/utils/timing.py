"""
Timing utilities for performance monitoring.
Provides decorators and context managers for measuring execution time.
"""

import functools
import time
from contextlib import contextmanager
from typing import Callable, Optional, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def timer(name: str = "Operation", log_level: str = "info"):
    """
    Context manager for timing code blocks.

    Usage:
        with timer("Database query"):
            # Your code here
            pass

    Args:
        name: Name of the operation being timed
        log_level: Log level for output (debug, info, warning, error)
    """
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

    log_func = getattr(logger, log_level.lower())
    log_func(
        f"{name} completed",
        extra={"duration_ms": round(elapsed, 2), "operation": name}
    )


def timed(func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
    """
    Decorator for timing function execution.

    Usage:
        @timed
        def my_function():
            pass

        @timed(name="Custom name")
        def my_function():
            pass

    Args:
        func: Function to wrap
        name: Custom name for logging (defaults to function name)

    Returns:
        Wrapped function
    """

    def decorator(f: Callable) -> Callable:
        operation_name = name or f.__name__

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(
                    f"{operation_name} completed",
                    extra={
                        "duration_ms": round(elapsed, 2),
                        "function": f.__name__,
                        "module": f.__module__
                    }
                )

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


class PerformanceTracker:
    """
    Track multiple timing measurements for reporting.

    Usage:
        tracker = PerformanceTracker()
        tracker.start("operation1")
        # ... do work ...
        tracker.stop("operation1")

        tracker.report()  # Print summary
    """

    def __init__(self):
        self.measurements: dict[str, list[float]] = {}
        self._active: dict[str, float] = {}

    def start(self, name: str):
        """Start timing an operation."""
        self._active[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        Stop timing an operation and record the duration.

        Args:
            name: Name of the operation

        Returns:
            Duration in milliseconds
        """
        if name not in self._active:
            raise ValueError(f"Timer '{name}' was not started")

        duration_ms = (time.perf_counter() - self._active[name]) * 1000
        del self._active[name]

        if name not in self.measurements:
            self.measurements[name] = []
        self.measurements[name].append(duration_ms)

        return duration_ms

    @contextmanager
    def measure(self, name: str):
        """
        Context manager for measuring a code block.

        Usage:
            with tracker.measure("database_query"):
                # Your code here
                pass
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def get_stats(self, name: str) -> dict[str, float]:
        """
        Get statistics for a specific operation.

        Args:
            name: Operation name

        Returns:
            Dict with min, max, mean, median, p95, p99 in ms
        """
        if name not in self.measurements:
            return {}

        measurements = sorted(self.measurements[name])
        n = len(measurements)

        if n == 0:
            return {}

        return {
            "count": n,
            "min": round(measurements[0], 2),
            "max": round(measurements[-1], 2),
            "mean": round(sum(measurements) / n, 2),
            "median": round(measurements[n // 2], 2),
            "p95": round(measurements[int(n * 0.95)], 2) if n > 1 else measurements[0],
            "p99": round(measurements[int(n * 0.99)], 2) if n > 1 else measurements[0],
        }

    def report(self) -> dict[str, dict[str, float]]:
        """
        Generate a report of all measurements.

        Returns:
            Dict mapping operation names to their statistics
        """
        report = {}
        for name in self.measurements:
            report[name] = self.get_stats(name)

        return report

    def log_report(self):
        """Log a formatted report of all measurements."""
        report = self.report()

        logger.info("=== Performance Report ===")
        for name, stats in report.items():
            logger.info(
                f"{name}: mean={stats['mean']}ms, p95={stats['p95']}ms, p99={stats['p99']}ms",
                extra={"operation": name, **stats}
            )

    def reset(self):
        """Clear all measurements."""
        self.measurements.clear()
        self._active.clear()


if __name__ == "__main__":
    # Test timing utilities
    import random

    # Test timer context manager
    with timer("Test operation"):
        time.sleep(0.1)

    # Test timed decorator
    @timed
    def slow_function():
        time.sleep(0.05)
        return "done"

    result = slow_function()

    # Test performance tracker
    tracker = PerformanceTracker()

    for i in range(10):
        with tracker.measure("api_call"):
            time.sleep(random.uniform(0.01, 0.05))

    tracker.log_report()
    print("\nDetailed stats:")
    print(tracker.get_stats("api_call"))
