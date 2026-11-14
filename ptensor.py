import numpy as np

S2 = np.sqrt(2.0)
_PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
_W = np.array([1, 1, 1, S2, S2, S2])


def mandel_to_t4(M):
    """6x6 Mandel -> 3x3x3x3."""
    T = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            v = M[I, J] / (_W[I] * _W[J])
            for (a, b) in {(i, j), (j, i)}:
                for (c, d) in {(k, l), (l, k)}:
                    T[a, b, c, d] = v
    return T


def t4_to_mandel(T):
    """3x3x3x3 -> 6x6 Mandel."""
    M = np.empty((6, 6))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            M[I, J] = _W[I] * _W[J] * T[i, j, k, l]
    return M


def rotate_mandel(C, R):
    """Rotate a Mandel 6x6 tensor with the 3x3 rotation matrix R."""
    Q = np.empty((6, 6))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            Q[I, J] = _W[I] * _W[J] * 0.5 * (R[i, k] * R[j, l] + R[i, l] * R[j, k])
    return Q @ C @ Q.T


def cubic_stiffness(c11, c12, c44):
    """Cubic stiffness (crystal frame) in Mandel notation."""
    C = np.zeros((6, 6))
    C[:3, :3] = c12
    C[range(3), range(3)] = c11
    C[3, 3] = C[4, 4] = C[5, 5] = 2 * c44
    return C


_I1 = np.array([p[0] for p in _PAIRS])
_I2 = np.array([p[1] for p in _PAIRS])


def fC0(x, y, C4, a):
    """
    Integrand of the P tensor at the point (x, y), returned in Mandel notation.
    C4 : stiffness as a 3x3x3x3 array (converted once, outside the integral)
    a  : (a1, a2, a3)
    """
    theta = np.arccos(1.0 - 2.0 * x)
    phi = 2.0 * np.pi * y
    STH, CTH = np.sin(theta), np.cos(theta)
    SPH, CPH = np.sin(phi), np.cos(phi)

    X = np.array([STH * CPH / a[0],
                  STH * SPH / a[1],
                  CTH / a[2]])

    # CXX_ik = C_ijkl X_j X_l  and its inverse
    CXX = np.einsum("ijkl,j,l->ik", C4, X, X)
    CXX = np.linalg.inv(CXX)

    # RES(I,J) with I=(i1,i2), J=(j1,j2): four-term symmetrisation
    i1, i2, j1, j2 = _I1[:, None], _I2[:, None], _I1[None, :], _I2[None, :]
    RES = (CXX[i1, j1] * X[i2] * X[j2] +
           CXX[i2, j1] * X[i1] * X[j2] +
           CXX[i1, j2] * X[i2] * X[j1] +
           CXX[i2, j2] * X[i1] * X[j1])
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


def tenseurP(a1, a2, a3, C0, n=32):
    """P tensor by direct calculation of the double integral."""
    factor, ymax = 0.5, 0.5
    C4 = mandel_to_t4(C0)
    a = (a1, a2, a3)
    return int2v(lambda x, y: fC0(x, y, C4, a), 0.0, 1.0, 0.0, ymax, n) * factor
