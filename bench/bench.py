"""Benchmark mojo-simsimd against the upstream package.

Run only through ``pixi run bench`` so the project-level flock is held.
"""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import simsimd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import mojo_simsimd as mojo  # noqa: E402


def best_time(function, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def measure(name, ours, theirs):
    ours()
    theirs()
    ours_time = best_time(ours)
    upstream_time = best_time(theirs)
    ratio = upstream_time / ours_time
    result = "faster" if ratio > 1 else "slower"
    print(f"| {name} | {ours_time * 1e3:.2f} ms | {upstream_time * 1e3:.2f} ms | {ratio:.2f}x {result} |")


def main():
    rng = np.random.default_rng(0)
    a = np.ascontiguousarray(rng.normal(size=(16_384, 256)))
    b = np.ascontiguousarray(rng.normal(size=(16_384, 256)))
    c = np.ascontiguousarray(rng.normal(size=(1024, 256)))
    d = np.ascontiguousarray(rng.normal(size=(1024, 256)))
    print(f"Machine: {platform.platform()} ({platform.processor() or 'unknown CPU'})")
    print("| case | mojo-simsimd | upstream simsimd | ratio |")
    print("| --- | ---: | ---: | ---: |")
    measure("dot (16,384 x 256)", lambda: mojo.dot(a, b), lambda: simsimd.dot(a, b))
    measure("sqeuclidean (16,384 x 256)", lambda: mojo.sqeuclidean(a, b), lambda: simsimd.sqeuclidean(a, b))
    measure("cosine (16,384 x 256)", lambda: mojo.cosine(a, b), lambda: simsimd.cosine(a, b))
    measure("cdist sqeuclidean (1,024 x 1,024 x 256)", lambda: mojo.cdist(c, d), lambda: simsimd.cdist(c, d, metric="sqeuclidean"))
    measure("wsum (4,194,304 values)", lambda: mojo.wsum(a.ravel(), b.ravel()), lambda: simsimd.wsum(a.ravel(), b.ravel()))


if __name__ == "__main__":
    main()
