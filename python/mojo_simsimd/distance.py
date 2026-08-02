"""NumPy-facing API for the floating-point subset of :mod:`simsimd`."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ._lib import addr, f64, lib

_METRICS = {
    "sqeuclidean": ("mss_sqeuclidean", 0),
    "l2sq": ("mss_sqeuclidean", 0),
    "euclidean": ("mss_euclidean", 1),
    "l2": ("mss_euclidean", 1),
    "cosine": ("mss_cosine", 2),
    "cos": ("mss_cosine", 2),
    "dot": ("mss_dot", 3),
    "inner": ("mss_dot", 3),
}


def _check_dtype(dtype, out_dtype=None) -> None:
    """Reject dtype requests that the Float64-only ABI cannot honor."""
    for name, value in (("dtype", dtype), ("out_dtype", out_dtype)):
        if value is not None and np.dtype(value) != np.dtype(np.float64):
            raise TypeError(f"{name} must be float64; this port has a Float64-only native ABI")


def _check_out(out, shape: tuple[int, ...]) -> np.ndarray:
    target = np.asarray(out)
    if target.shape != shape:
        raise ValueError(f"out has shape {target.shape}, expected {shape}")
    if target.dtype != np.float64:
        raise TypeError("out must have dtype float64")
    if not target.flags.writeable:
        raise ValueError("out must be writable")
    return target


def _rows(value, name: str) -> tuple[np.ndarray, bool]:
    array = f64(value)
    if array.ndim == 1:
        return array.reshape(1, -1), True
    if array.ndim == 2:
        return array, False
    raise ValueError(f"{name} must be a vector or a two-dimensional row-major matrix")


def _paired(a, b) -> tuple[np.ndarray, np.ndarray, bool]:
    left, left_scalar = _rows(a, "a")
    right, right_scalar = _rows(b, "b")
    if left.shape[1] != right.shape[1]:
        raise ValueError("a and b must have equal final dimensions")
    rows = max(left.shape[0], right.shape[0])
    if left.shape[0] not in (1, rows) or right.shape[0] not in (1, rows):
        raise ValueError("a and b must have equal row counts or one input must be a vector")
    if left.shape[0] == 1 and rows != 1:
        left = np.broadcast_to(left, (rows, left.shape[1])).copy()
    if right.shape[0] == 1 and rows != 1:
        right = np.broadcast_to(right, (rows, right.shape[1])).copy()
    return left, right, left_scalar and right_scalar


def _result_shape(scalar: bool, rows: int) -> tuple[int, ...]:
    return () if scalar else (rows,)


def _metric(name: str) -> Callable:
    native, _ = _METRICS[name]

    def operation(a, b, /, dtype=None, *, out=None, out_dtype=None):
        _check_dtype(dtype, out_dtype)
        left, right, scalar = _paired(a, b)
        result = np.empty(_result_shape(scalar, left.shape[0]), dtype=np.float64)
        if left.shape[1] == 0:
            result.fill(0.0)
        elif scalar:
            kernel = getattr(lib(), native)
            result = np.asarray(kernel(addr(left), addr(right), left.shape[1]))
        else:
            lib().mss_paired(
                addr(left), addr(right), addr(result), left.shape[0], left.shape[1], _METRICS[name][1]
            )
        if out is not None:
            target = _check_out(out, result.shape)
            target[...] = result
            return None
        if scalar:
            return float(result)
        return result.astype(out_dtype or np.float64, copy=False)

    operation.__name__ = name
    return operation


dot = _metric("dot")
inner = _metric("inner")
sqeuclidean = _metric("sqeuclidean")
l2sq = _metric("l2sq")
euclidean = _metric("euclidean")
l2 = _metric("l2")
cosine = _metric("cosine")
cos = _metric("cos")


def cdist(a, b, /, metric="sqeuclidean", *, dtype=None, out=None, out_dtype=None, threads=1):
    """Pairwise distances between two matrices, as in ``simsimd.cdist``."""
    _check_dtype(dtype, out_dtype)
    if threads != 1:
        raise ValueError("threads is not configurable in this port")
    try:
        _, metric_id = _METRICS[metric]
    except KeyError as error:
        raise ValueError(f"unsupported metric {metric!r}") from error
    left, left_scalar = _rows(a, "a")
    right, right_scalar = _rows(b, "b")
    if left_scalar or right_scalar:
        raise ValueError("cdist inputs must be two-dimensional matrices")
    if left.shape[1] != right.shape[1]:
        raise ValueError("a and b must have equal final dimensions")
    result = np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
    if left.shape[0] and right.shape[0] and left.shape[1]:
        lib().mss_pairwise(
            addr(left), addr(right), addr(result), left.shape[0], right.shape[0], left.shape[1], metric_id
        )
    else:
        result.fill(0.0)
    if out is not None:
        target = _check_out(out, result.shape)
        target[...] = result
        return None
    return result.astype(out_dtype or np.float64, copy=False)


def bilinear(a, b, metric_tensor, /, dtype=None) -> float:
    _check_dtype(dtype)
    left = f64(a)
    right = f64(b)
    matrix = f64(metric_tensor)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("a and b must be vectors")
    if left.shape != right.shape or matrix.shape != (left.size, left.size):
        raise ValueError("a, b, and metric_tensor have incompatible shapes")
    if not left.size:
        return 0.0
    return float(lib().mss_bilinear(addr(left), addr(right), addr(matrix), left.size))


def mahalanobis(a, b, inverse_covariance, /, dtype=None) -> float:
    _check_dtype(dtype)
    left = f64(a)
    right = f64(b)
    matrix = f64(inverse_covariance)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("a and b must be vectors")
    if left.shape != right.shape or matrix.shape != (left.size, left.size):
        raise ValueError("a, b, and inverse_covariance have incompatible shapes")
    if not left.size:
        return 0.0
    return float(lib().mss_mahalanobis(addr(left), addr(right), addr(matrix), left.size))


def _vector_out(out, n: int, like: np.ndarray) -> np.ndarray:
    if out is None:
        return np.empty_like(like)
    target = _check_out(out, (n,))
    if target.dtype != np.float64 or not target.flags.c_contiguous:
        return np.empty(n, dtype=np.float64)
    return target


def wsum(a, b, /, dtype=None, *, alpha=1.0, beta=1.0, out=None):
    _check_dtype(dtype)
    left = f64(a)
    right = f64(b)
    if left.ndim != 1 or right.ndim != 1:
        raise ValueError("a and b must be vectors")
    if left.shape != right.shape:
        raise ValueError("a and b must have equal shapes")
    result = _vector_out(out, left.size, left)
    if left.size:
        lib().mss_wsum(addr(left), addr(right), addr(result), left.size, alpha, beta)
    if out is not None:
        np.asarray(out)[...] = result
        return None
    return result


def fma(a, b, c, /, dtype=None, *, alpha=1.0, beta=1.0, out=None):
    _check_dtype(dtype)
    left = f64(a)
    right = f64(b)
    third = f64(c)
    if left.ndim != 1 or right.ndim != 1 or third.ndim != 1:
        raise ValueError("a, b, and c must be vectors")
    if left.shape != right.shape or left.shape != third.shape:
        raise ValueError("a, b, and c must have equal shapes")
    result = _vector_out(out, left.size, left)
    if left.size:
        lib().mss_fma(addr(left), addr(right), addr(third), addr(result), left.size, alpha, beta)
    if out is not None:
        np.asarray(out)[...] = result
        return None
    return result
