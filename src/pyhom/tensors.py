"""
Tensor algebra: conversions, rotations and elementary stiffness builders.

Convention: fourth-order tensors are 6x6 matrices in Mandel notation, MFront ordering
(xx, yy, zz, xy, xz, yz), with a factor sqrt(2) on shear components.  Products and
inversions of fourth-order tensors are then plain matrix products / np.linalg.inv.
"""
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


def isotropic_stiffness(k, mu):
    J = np.zeros((6, 6)); J[:3, :3] = 1 / 3
    return 3 * k * J + 2 * mu * (np.eye(6) - J)


def bulk_shear(L):
    """Isotropic part (k, mu) of a Mandel stiffness."""
    J = np.zeros((6, 6)); J[:3, :3] = 1 / 3
    K = np.eye(6) - J
    return np.sum(L * J) / 3.0, np.sum(L * K) / 10.0
