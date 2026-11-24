"""
Hill (P) tensor of an ellipsoidal inclusion by numerical integration over the sphere.

    fC0(x, y, C4, a)       integrand,  x = (1 - cos(theta)) / 2 ,  y = phi / (2 pi)
                           x, y scalars -> one point (6,6);  x, y arrays of shape (N,) -> all points at once (N,6,6)
    int2v(f, ...)          double integral, tensor-product Gauss-Legendre, loop over the points (reference)
    int2v_vec(f, ...)      same integral, f called ONCE on the whole grid (vectorised)
    tenseurP(a1,a2,a3,C0,n,vect=True)
                           P = int2v(fC0, x in [0,1], y in [0,1/2]) * 1/2 ;  vect=True uses int2v_vec

Tensors are 6x6 Mandel matrices (see tensors.py).
"""
import numpy as np

from .tensors import _PAIRS, _W, mandel_to_t4

_I1 = np.array([p[0] for p in _PAIRS])
_I2 = np.array([p[1] for p in _PAIRS])


def fC0(x, y, C4, a):
    """
    Integrand of the P-tensor at the point (x, y), returned in Mandel notation.
    x, y : scalars (-> (6,6)) or arrays of the same shape (...) (-> (..., 6, 6)); the algebra is the same,
           np.einsum("...") and the stacked np.linalg.inv do the loop over the points.
    C4 : stiffness as a 3x3x3x3 array (converted once, outside the integral)
    a  : (a1, a2, a3)
    """
    theta = np.arccos(1.0 - 2.0 * x)
    phi = 2.0 * np.pi * y
    STH, CTH = np.sin(theta), np.cos(theta)
    SPH, CPH = np.sin(phi), np.cos(phi)

    X = np.stack([STH * CPH / a[0],
                  STH * SPH / a[1],
                  CTH / a[2]], axis=-1)                          # (..., 3)

    # CXX_ik = C_ijkl X_j X_l  and its inverse
    CXX = np.einsum("ijkl,...j,...l->...ik", C4, X, X)
    CXX = np.linalg.inv(CXX)

    # RES(I,J) with I=(i1,i2), J=(j1,j2): four-term symmetrisation
    i1, i2, j1, j2 = _I1[:, None], _I2[:, None], _I1[None, :], _I2[None, :]
    RES = (CXX[..., i1, j1] * X[..., i2] * X[..., j2] +
           CXX[..., i2, j1] * X[..., i1] * X[..., j2] +
           CXX[..., i1, j2] * X[..., i2] * X[..., j1] +
           CXX[..., i2, j2] * X[..., i1] * X[..., j1])
    return RES * _W[:, None] * _W[None, :]        # Mandel factors


def int2v(f, xmin, xmax, ymin, ymax, n=32):
    """Double integral of f(x, y) by tensor-product Gauss-Legendre (n x n)."""
    g, w = np.polynomial.legendre.leggauss(n)
    xs = 0.5 * (xmax - xmin) * g + 0.5 * (xmax + xmin)
    ys = 0.5 * (ymax - ymin) * g + 0.5 * (ymax + ymin)
    wx = 0.5 * (xmax - xmin) * w
    wy = 0.5 * (ymax - ymin) * w
    res = 0.0
    for xi, wxi in zip(xs, wx):
        for yj, wyj in zip(ys, wy):
            res = res + wxi * wyj * f(xi, yj)
    return res


def int2v_vec(f, xmin, xmax, ymin, ymax, n=32):
    """
    Same integral as int2v, but f(x, y) is called ONCE with the flattened (n*n,) arrays of all the nodes
    and must return an array of shape (n*n, ...); the weighted sum is one np.einsum.
    """
    g, w = np.polynomial.legendre.leggauss(n)
    xs = 0.5 * (xmax - xmin) * g + 0.5 * (xmax + xmin)
    ys = 0.5 * (ymax - ymin) * g + 0.5 * (ymax + ymin)
    wx = 0.5 * (xmax - xmin) * w
    wy = 0.5 * (ymax - ymin) * w
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    W = np.outer(wx, wy).ravel()
    return np.einsum("n,n...->...", W, f(X.ravel(), Y.ravel()))


def ptensor(a1, a2, a3, C0, n=32, vect=True):
    """
    P-tensor by direct calculation of the double integral.
    vect=True  : fC0 is evaluated on all n*n quadrature points at once (vectorised, ~15-20x faster)
    vect=False : point-by-point loop (reference)
    Same algebra in both cases; results agree to rounding (~1e-15 relative).
    """
    factor, ymax = 0.5, 0.5
    C4 = mandel_to_t4(C0)
    a = (a1, a2, a3)
    integrate = int2v_vec if vect else int2v
    return integrate(lambda x, y: fC0(x, y, C4, a), 0.0, 1.0, 0.0, ymax, n) * factor