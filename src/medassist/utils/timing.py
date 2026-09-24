"""Per-node latency measurement + logging setup."""
from __future__ import annotations

import functools
import logging
import time

logger = logging.getLogger("medassist")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def timed(name: str):
    """Decorate a LangGraph node: adds {'timings': {name: seconds}} to its output.

    LangSmith records latency for every node too; this copy is used by the CLI,
    the Streamlit sidebar and the benchmark script.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            start = time.perf_counter()
            out = dict(fn(state) or {})
            elapsed = round(time.perf_counter() - start, 3)
            logger.info("[timing] %s: %.2fs", name, elapsed)
            out["timings"] = {name: elapsed}
            return out

        return wrapper

    return decorator
