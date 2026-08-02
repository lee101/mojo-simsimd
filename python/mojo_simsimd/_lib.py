"""Build and load the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_SIMSIMD_LIB") or os.path.join(
    ROOT, "dist", "libmojo-simsimd.so"
)

I = ctypes.c_int64
F = ctypes.c_double
_SIGNATURES = {
    "mss_dot": ([I, I, I], F),
    "mss_sqeuclidean": ([I, I, I], F),
    "mss_euclidean": ([I, I, I], F),
    "mss_cosine": ([I, I, I], F),
    "mss_pairwise": ([I, I, I, I, I, I, I], None),
    "mss_paired": ([I, I, I, I, I, I], None),
    "mss_bilinear": ([I, I, I, I], F),
    "mss_mahalanobis": ([I, I, I, I], F),
    "mss_wsum": ([I, I, I, I, F, F], None),
    "mss_fma": ([I, I, I, I, I, F, F], None),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJO_SIMSIMD_MOJO")
    if override:
        return override.split()
    if found := shutil.which("mojo"):
        return [found]
    if pixi := (shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("Mojo compiler not found; set MOJO_SIMSIMD_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_SIMSIMD_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = [
        os.path.join(directory, filename)
        for directory, _, filenames in os.walk(SRC)
        for filename in filenames
        if filename.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = _mojo_command() + [
        "build", "--emit", "shared-lib", os.path.join(SRC, "kernels.mojo"), "-o", LIB,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not os.path.exists(LIB):
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value) -> np.ndarray:
    """Return a C-contiguous float64 array without lossy coercion.

    The native ABI only understands Float64.  In particular, converting a
    complex or extended-precision array here would silently discard data
    before the pointer reaches Mojo.
    """
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("mojo-simsimd supports only real numeric inputs")
    if array.dtype.kind == "f" and array.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("mojo-simsimd cannot losslessly convert extended-precision inputs to float64")
    if array.dtype.kind in "iu" and array.size and np.max(array) > 2**53:
        raise TypeError("integer inputs above 2**53 cannot be represented exactly as float64")
    if array.dtype.kind == "i" and array.size and np.min(array) < -(2**53):
        raise TypeError("integer inputs below -2**53 cannot be represented exactly as float64")
    return np.ascontiguousarray(array, dtype=np.float64)


def addr(value: np.ndarray) -> int:
    address = value.ctypes.data
    if not address:
        raise ValueError("cannot pass a null buffer to the native kernels")
    return address
