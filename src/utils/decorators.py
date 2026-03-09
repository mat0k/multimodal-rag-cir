import functools
import time
from typing import Any, Callable


def timed_metric(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_seconds = time.perf_counter() - start_time

        if isinstance(result, dict):
            result["latency_seconds"] = elapsed_seconds
            return result

        if isinstance(result, tuple) and result and isinstance(result[0], dict):
            result[0]["latency_seconds"] = elapsed_seconds
            return result

        return result

    return wrapper
