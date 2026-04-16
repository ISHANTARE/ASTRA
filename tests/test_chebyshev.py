import numpy as np
import numpy.polynomial.chebyshev as cheb
from astra.propagator import _eval_cheb_3d_njit


def test_chebyshev_clenshaw_njit():
    """
    [HIGH-01] Validates the custom Numba Clenshaw recurrence against numpy's standard chebval.
    """
    np.random.seed(42)
    # 10 coefficients for each of the 3 dimensions
    coeffs = np.random.randn(10, 3)

    # Test random normalized times
    t_norms = np.linspace(-1, 1, 50)
    for t_norm in t_norms:
        # Our custom Numba JIT-compiled evaluator
        ans_njit = _eval_cheb_3d_njit(t_norm, coeffs)

        # Reference NumPy evaluator
        ans_ref = np.zeros(3)
        for i in range(3):
            ans_ref[i] = cheb.chebval(t_norm, coeffs[:, i])

        np.testing.assert_allclose(ans_njit, ans_ref, rtol=1e-12, atol=1e-12)


def test_chebyshev_clenshaw_njit_edge_cases():
    """Test Clenshaw recurrence for small numbers of coefficients."""
    t_norm = 0.5

    # N = 1 (constant)
    c1 = np.array([[1.0, 2.0, 3.0]])
    ans1 = _eval_cheb_3d_njit(t_norm, c1)
    np.testing.assert_allclose(ans1, c1[0])

    # N = 2 (linear)
    c2 = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, -0.5]])
    ans2 = _eval_cheb_3d_njit(t_norm, c2)
    ref2 = [cheb.chebval(t_norm, c2[:, i]) for i in range(3)]
    np.testing.assert_allclose(ans2, ref2)
