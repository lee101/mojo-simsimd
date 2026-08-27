# mojo-simsimd

`mojo-simsimd` is a standalone Mojo implementation of a small, Float64-only subset of [simsimd](https://github.com/ashvardanian/simsimd). It is not a drop-in replacement for upstream: this port deliberately rejects complex, extended-precision, and unsafe output-dtype conversions.

```python
import numpy as np
import mojo_simsimd as simsimd

a = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
b = np.array([[4.0, 5.0, 6.0], [1.0, 1.0, 1.0]])

assert np.allclose(simsimd.cosine(a, b), [0.02536815, 0.22540333])
assert np.allclose(simsimd.cdist(a, b, metric="sqeuclidean"), [[27.0, 5.0], [56.0, 2.0]])
```

## Covered subset

| upstream API | status |
| --- | --- |
| `dot`, `inner` | covered for real numeric vectors and paired row matrices |
| `sqeuclidean`, `l2sq`, `euclidean`, `l2` | covered |
| `cosine`, `cos` | covered, including zero-norm behavior |
| `cdist` | covered for `dot`, `sqeuclidean`, `euclidean`, and `cosine` |
| `bilinear`, `mahalanobis` | covered for real numeric vectors and dense metric tensors |
| `wsum`, `fma` | covered for real numeric vectors |

Not covered: integer-bit metrics (`hamming`, `jaccard`), complex kernels (`vdot`), probability divergences (`kl`, `jensenshannon`), sorted-set intersection, capability controls, threaded API controls, and upstream raw-kernel pointers. Real inputs are normalized to contiguous `float64`, but integers outside Float64's exact range are rejected. `dtype`, `out_dtype`, and `out` must be Float64 when supplied. This port does not provide upstream's full dtype-dispatch matrix or its `DistancesTensor` result type.

## Install and test

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

The test suite installs upstream `simsimd` and checks numeric parity for every documented operation, output-buffer behavior, SIMD tails, empty inputs, and the Python-to-native validation boundary. The example above is executed as part of the release check.

## Performance

Measured on 2026-08-02 with `pixi run bench` on Linux 6.8.0-136-generic, x86_64, using upstream simsimd 6.5.16. Timings are best of five.

| case | mojo-simsimd | upstream simsimd | result |
| --- | ---: | ---: | --- |
| dot (16,384 x 256) | 7.33 ms | 7.14 ms | 0.97x slower |
| sqeuclidean (16,384 x 256) | 7.14 ms | 7.28 ms | 1.02x faster |
| cosine (16,384 x 256) | 7.40 ms | 7.83 ms | 1.06x faster |
| cdist sqeuclidean (1,024 x 1,024 x 256) | 22.50 ms | 127.30 ms | 5.66x faster |
| wsum (4,194,304 values) | 18.98 ms | 30.21 ms | 1.59x faster |

The pairwise kernel parallelizes independent output rows once the input size crosses its fixed threshold. Batched one-to-one metrics are one native call, not a Python loop.

## How it works

`src/kernels.mojo` is intentionally one compilation unit: it produces one shared library, `dist/libmojo-simsimd.so`. Core reductions use four-lane `Float64` SIMD plus scalar tails. Matrices are dense, row-major, C-contiguous buffers.

Python validates shapes and dtypes, makes C-contiguous Float64 input arrays, and keeps those arrays alive for the duration of each `ctypes` call. Empty inputs are handled in Python so a null/empty NumPy buffer never reaches the native ABI. Each `@export` Mojo wrapper reconstructs a `Pointer[Float64, AnyOrigin[mut=True]]`; Mojo does not allocate or own Python memory. A contiguous Float64 output can be written directly; other valid output views use a temporary and are copied back.

## License

MIT
