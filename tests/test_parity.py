"""Parity checks against the published Python simsimd package."""

import numpy as np
import pytest
import simsimd

import mojo_simsimd as mojo


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(42)
    return rng.normal(size=(7, 257)), rng.normal(size=(7, 257))


@pytest.mark.parametrize("name", ["dot", "inner", "sqeuclidean", "l2sq", "euclidean", "l2", "cosine", "cos"])
def test_vector_metric_parity(vectors, name):
    a, b = vectors
    ours = getattr(mojo, name)(a, b)
    theirs = getattr(simsimd, name)(a, b)
    tolerance = 2e-7 if name in {"cosine", "cos"} else 2e-12
    assert np.allclose(ours, theirs, rtol=tolerance, atol=tolerance)
    assert getattr(mojo, name)(a[0], b[0]) == pytest.approx(
        float(np.asarray(theirs)[0]), rel=tolerance, abs=tolerance
    )


@pytest.mark.parametrize("name", ["dot", "sqeuclidean", "euclidean", "cosine"])
def test_vector_broadcast_and_out_parity(vectors, name):
    a, b = vectors
    expected = getattr(simsimd, name)(a, b[0])
    actual = getattr(mojo, name)(a, b[0])
    tolerance = 2e-7 if name == "cosine" else 2e-12
    assert np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    out = np.empty(a.shape[0])
    assert getattr(mojo, name)(a, b[0], out=out) is None
    assert np.allclose(out, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("metric", ["sqeuclidean", "euclidean", "cosine", "dot"])
def test_cdist_parity(vectors, metric):
    a, b = vectors
    expected = simsimd.cdist(a[:4], b[2:], metric=metric)
    actual = mojo.cdist(a[:4], b[2:], metric=metric)
    tolerance = 2e-7 if metric == "cosine" else 2e-12
    assert np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    out = np.empty((4, 5))
    assert mojo.cdist(a[:4], b[2:], metric=metric, out=out) is None
    assert np.allclose(out, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("rows", [8, 64])
def test_cdist_parallel_threshold(rows):
    rng = np.random.default_rng(rows)
    a = rng.normal(size=(rows, 256))
    b = rng.normal(size=(rows, 256))
    assert np.allclose(mojo.cdist(a, b), simsimd.cdist(a, b, metric="sqeuclidean"), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("rows", [3891, 3892])
def test_paired_parallel_threshold(rows):
    rng = np.random.default_rng(rows)
    a = rng.normal(size=(rows, 257))
    b = rng.normal(size=(rows, 257))
    expected = np.sum(a * b, axis=1)
    assert np.allclose(mojo.dot(a, b), expected, rtol=2e-12, atol=2e-12)


def test_bilinear_and_mahalanobis_parity(vectors):
    a, b = vectors
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    diagonal = np.linspace(0.2, 3.0, 16)
    metric = (q * diagonal) @ q.T
    assert mojo.bilinear(a[0, :16], b[0, :16], metric) == pytest.approx(
        simsimd.bilinear(a[0, :16], b[0, :16], metric), rel=2e-12, abs=2e-12
    )
    assert mojo.mahalanobis(a[0, :16], b[0, :16], metric) == pytest.approx(
        simsimd.mahalanobis(a[0, :16], b[0, :16], metric), rel=1e-3, abs=1e-3
    )


def test_wsum_and_fma_parity(vectors):
    a, b = vectors
    c = a[2]
    expected_wsum = simsimd.wsum(a[0], b[0], alpha=-0.75, beta=1.25)
    expected_fma = simsimd.fma(a[0], b[0], c, alpha=-0.75, beta=1.25)
    assert np.allclose(mojo.wsum(a[0], b[0], alpha=-0.75, beta=1.25), expected_wsum)
    assert np.allclose(mojo.fma(a[0], b[0], c, alpha=-0.75, beta=1.25), expected_fma)
    out = np.empty_like(c)
    assert mojo.fma(a[0], b[0], c, out=out) is None
    assert np.allclose(out, simsimd.fma(a[0], b[0], c))
    assert mojo.wsum(a[0], b[0], out=out) is None
    assert np.allclose(out, simsimd.wsum(a[0], b[0]))


@pytest.mark.parametrize("size", [3, 17, 257])
def test_transform_simd_tails(size):
    a = np.linspace(-1.0, 1.0, size)
    b = np.linspace(1.0, -1.0, size)
    c = np.linspace(0.5, 1.5, size)
    assert np.allclose(mojo.wsum(a, b, alpha=-0.75, beta=1.25), simsimd.wsum(a, b, alpha=-0.75, beta=1.25))
    assert np.allclose(mojo.fma(a, b, c, alpha=-0.75, beta=1.25), simsimd.fma(a, b, c, alpha=-0.75, beta=1.25))


@pytest.mark.parametrize("size", [2_097_152, 2_097_155])
def test_wsum_parallel_threshold_and_tail(size):
    a = np.linspace(-1.0, 1.0, size)
    b = np.linspace(1.0, -1.0, size)
    expected = -0.75 * a + 1.25 * b
    assert np.allclose(mojo.wsum(a, b, alpha=-0.75, beta=1.25), expected)


def test_zero_norm_cosine_matches_upstream():
    zero = np.zeros(8)
    nonzero = np.arange(8.0)
    for a, b in [(zero, zero), (zero, nonzero), (nonzero, zero)]:
        assert mojo.cosine(a, b) == simsimd.cosine(a, b)


def test_validation():
    with pytest.raises(ValueError):
        mojo.dot(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        mojo.cdist(np.ones(3), np.ones((2, 3)))
    with pytest.raises(ValueError):
        mojo.cdist(np.ones((2, 3)), np.ones((2, 3)), metric="hamming")
    with pytest.raises(TypeError):
        mojo.dot(np.ones(3, dtype=np.complex128), np.ones(3))
    with pytest.raises(TypeError):
        mojo.dot(np.ones(3), np.ones(3), out_dtype=np.float32)
    with pytest.raises(TypeError):
        mojo.dot(np.ones(3), np.ones(3), out=np.empty((), dtype=np.float32))
    with pytest.raises(TypeError):
        mojo.dot(np.array([2**53 + 1], dtype=np.uint64), np.ones(1))


@pytest.mark.parametrize("name", ["dot", "sqeuclidean", "euclidean", "cosine"])
def test_empty_inputs_do_not_cross_the_pointer_abi(name):
    empty = np.empty(0)
    assert getattr(mojo, name)(empty, empty) == pytest.approx(getattr(simsimd, name)(empty, empty))
    assert np.array_equal(mojo.cdist(np.empty((2, 0)), np.empty((3, 0)), metric=name), np.zeros((2, 3)))


def test_non_contiguous_float64_out_is_copied_back():
    storage = np.empty((2, 7))
    out = storage[:, 2]
    a = np.arange(2.0)
    b = np.arange(2.0, 4.0)
    assert mojo.wsum(a, b, out=out) is None
    assert np.allclose(out, simsimd.wsum(a, b))
