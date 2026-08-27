"""Float64 SIMD distance kernels and their small C ABI."""

from max.algorithm import parallelize
from std.math import sqrt
from std.runtime import initialize_runtime


comptime W = 4
comptime UNROLL = 4
comptime PAIRWISE_PARALLEL_THRESHOLD = 1_000_000
comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]


def ptr(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def dot_kernel(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a.unsafe_load(i) * b.unsafe_load(i)
        i += 1
    return total


def sqeuclidean_kernel(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var delta = a.unsafe_load[width=W](i) - b.unsafe_load[width=W](i)
        acc += delta * delta
        i += W
    var total = acc.reduce_add()
    while i < n:
        var delta = a.unsafe_load(i) - b.unsafe_load(i)
        total += delta * delta
        i += 1
    return total


def cosine_kernel(a: Ptr, b: Ptr, n: Int) -> Float64:
    var dot_acc = SIMD[DType.float64, W](0.0)
    var a_acc = SIMD[DType.float64, W](0.0)
    var b_acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        var av = a.unsafe_load[width=W](i)
        var bv = b.unsafe_load[width=W](i)
        dot_acc += av * bv
        a_acc += av * av
        b_acc += bv * bv
        i += W
    var dot_total = dot_acc.reduce_add()
    var a_total = a_acc.reduce_add()
    var b_total = b_acc.reduce_add()
    while i < n:
        var av = a.unsafe_load(i)
        var bv = b.unsafe_load(i)
        dot_total += av * bv
        a_total += av * av
        b_total += bv * bv
        i += 1
    if a_total == 0.0 or b_total == 0.0:
        return 0.0 if a_total == b_total else 1.0
    return 1.0 - dot_total / sqrt(a_total * b_total)


def distance_kernel(a: Ptr, b: Ptr, n: Int, metric: Int) -> Float64:
    if metric == 0:
        return sqeuclidean_kernel(a, b, n)
    if metric == 1:
        return sqrt(sqeuclidean_kernel(a, b, n))
    if metric == 2:
        return cosine_kernel(a, b, n)
    return dot_kernel(a, b, n)


def pairwise_kernel(
    a: Ptr, b: Ptr, dst: Ptr, rows_a: Int, rows_b: Int, cols: Int, metric: Int
):
    if rows_a * rows_b * cols >= PAIRWISE_PARALLEL_THRESHOLD:

        def pairwise_row(i: Int) {imm}:
            for j in range(rows_b):
                dst.unsafe_store(
                    i * rows_b + j,
                    distance_kernel(
                        a.unsafe_offset(i * cols),
                        b.unsafe_offset(j * cols),
                        cols,
                        metric,
                    ),
                )

        parallelize(pairwise_row, rows_a, 8)
        return

    for i in range(rows_a):
        for j in range(rows_b):
            dst.unsafe_store(
                i * rows_b + j,
                distance_kernel(
                    a.unsafe_offset(i * cols),
                    b.unsafe_offset(j * cols),
                    cols,
                    metric,
                ),
            )


def paired_kernel(a: Ptr, b: Ptr, dst: Ptr, rows: Int, cols: Int, metric: Int):
    for i in range(rows):
        dst.unsafe_store(
            i,
            distance_kernel(
                a.unsafe_offset(i * cols),
                b.unsafe_offset(i * cols),
                cols,
                metric,
            ),
        )


def bilinear_kernel(a: Ptr, b: Ptr, matrix: Ptr, n: Int) -> Float64:
    var total = 0.0
    for i in range(n):
        var row_total = 0.0
        for j in range(n):
            row_total += matrix.unsafe_load(i * n + j) * b.unsafe_load(j)
        total += a.unsafe_load(i) * row_total
    return total


def mahalanobis_kernel(a: Ptr, b: Ptr, matrix: Ptr, n: Int) -> Float64:
    var total = 0.0
    for i in range(n):
        var row_total = 0.0
        for j in range(n):
            row_total += matrix.unsafe_load(i * n + j) * (
                b.unsafe_load(j) - a.unsafe_load(j)
            )
        total += (b.unsafe_load(i) - a.unsafe_load(i)) * row_total
    return sqrt(total) if total > 0.0 else 0.0


def wsum_kernel(
    a: Ptr, b: Ptr, dst: Ptr, n: Int, alpha: Float64, beta: Float64
):
    var va = SIMD[DType.float64, W](alpha)
    var vb = SIMD[DType.float64, W](beta)
    var i = 0
    while i + W * UNROLL <= n:
        dst.unsafe_store(
            i, va * a.unsafe_load[width=W](i) + vb * b.unsafe_load[width=W](i)
        )
        dst.unsafe_store(
            i + W,
            va * a.unsafe_load[width=W](i + W)
            + vb * b.unsafe_load[width=W](i + W),
        )
        dst.unsafe_store(
            i + 2 * W,
            va * a.unsafe_load[width=W](i + 2 * W)
            + vb * b.unsafe_load[width=W](i + 2 * W),
        )
        dst.unsafe_store(
            i + 3 * W,
            va * a.unsafe_load[width=W](i + 3 * W)
            + vb * b.unsafe_load[width=W](i + 3 * W),
        )
        i += W * UNROLL
    while i + W <= n:
        dst.unsafe_store(
            i, va * a.unsafe_load[width=W](i) + vb * b.unsafe_load[width=W](i)
        )
        i += W
    while i < n:
        dst.unsafe_store(i, alpha * a.unsafe_load(i) + beta * b.unsafe_load(i))
        i += 1


def fma_kernel(
    a: Ptr, b: Ptr, c: Ptr, dst: Ptr, n: Int, alpha: Float64, beta: Float64
):
    var va = SIMD[DType.float64, W](alpha)
    var vb = SIMD[DType.float64, W](beta)
    var i = 0
    while i + W * UNROLL <= n:
        dst.unsafe_store(
            i,
            va * a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)
            + vb * c.unsafe_load[width=W](i),
        )
        dst.unsafe_store(
            i + W,
            va * a.unsafe_load[width=W](i + W) * b.unsafe_load[width=W](i + W)
            + vb * c.unsafe_load[width=W](i + W),
        )
        dst.unsafe_store(
            i + 2 * W,
            va
            * a.unsafe_load[width=W](i + 2 * W)
            * b.unsafe_load[width=W](i + 2 * W)
            + vb * c.unsafe_load[width=W](i + 2 * W),
        )
        dst.unsafe_store(
            i + 3 * W,
            va
            * a.unsafe_load[width=W](i + 3 * W)
            * b.unsafe_load[width=W](i + 3 * W)
            + vb * c.unsafe_load[width=W](i + 3 * W),
        )
        i += W * UNROLL
    while i + W <= n:
        dst.unsafe_store(
            i,
            va * a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)
            + vb * c.unsafe_load[width=W](i),
        )
        i += W
    while i < n:
        dst.unsafe_store(
            i,
            alpha * a.unsafe_load(i) * b.unsafe_load(i)
            + beta * c.unsafe_load(i),
        )
        i += 1


@export("mss_dot")
def mss_dot(a: Int, b: Int, n: Int) abi("C") -> Float64:
    return dot_kernel(ptr(a), ptr(b), n)


@export("mss_sqeuclidean")
def mss_sqeuclidean(a: Int, b: Int, n: Int) abi("C") -> Float64:
    return sqeuclidean_kernel(ptr(a), ptr(b), n)


@export("mss_euclidean")
def mss_euclidean(a: Int, b: Int, n: Int) abi("C") -> Float64:
    return sqrt(sqeuclidean_kernel(ptr(a), ptr(b), n))


@export("mss_cosine")
def mss_cosine(a: Int, b: Int, n: Int) abi("C") -> Float64:
    return cosine_kernel(ptr(a), ptr(b), n)


@export("mss_pairwise")
def mss_pairwise(
    a: Int, b: Int, dst: Int, rows_a: Int, rows_b: Int, cols: Int, metric: Int
) abi("C"):
    initialize_runtime()
    pairwise_kernel(ptr(a), ptr(b), ptr(dst), rows_a, rows_b, cols, metric)


@export("mss_paired")
def mss_paired(
    a: Int, b: Int, dst: Int, rows: Int, cols: Int, metric: Int
) abi("C"):
    paired_kernel(ptr(a), ptr(b), ptr(dst), rows, cols, metric)


@export("mss_bilinear")
def mss_bilinear(a: Int, b: Int, matrix: Int, n: Int) abi("C") -> Float64:
    return bilinear_kernel(ptr(a), ptr(b), ptr(matrix), n)


@export("mss_mahalanobis")
def mss_mahalanobis(a: Int, b: Int, matrix: Int, n: Int) abi("C") -> Float64:
    return mahalanobis_kernel(ptr(a), ptr(b), ptr(matrix), n)


@export("mss_wsum")
def mss_wsum(
    a: Int, b: Int, dst: Int, n: Int, alpha: Float64, beta: Float64
) abi("C"):
    wsum_kernel(ptr(a), ptr(b), ptr(dst), n, alpha, beta)


@export("mss_fma")
def mss_fma(
    a: Int, b: Int, c: Int, dst: Int, n: Int, alpha: Float64, beta: Float64
) abi("C"):
    fma_kernel(ptr(a), ptr(b), ptr(c), ptr(dst), n, alpha, beta)
