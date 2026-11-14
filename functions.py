"""
Homogenization estimates built on the P tensor of ptensor.py.

Tensors are 6x6 Mandel matrices (MFront ordering xx,yy,zz,xy,xz,yz).
"""
import numpy as np
from ptensor import tenseurP          # ptensor.py must be in the same folder


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def isotropic_stiffness(k, mu):
    J = np.zeros((6, 6)); J[:3, :3] = 1 / 3
    return 3 * k * J + 2 * mu * (np.eye(6) - J)


def bulk_shear(L):
    """Isotropic part (k, mu) of a Mandel stiffness."""
    J = np.zeros((6, 6)); J[:3, :3] = 1 / 3
    K = np.eye(6) - J
    return np.sum(L * J) / 3.0, np.sum(L * K) / 10.0


# ----------------------------------------------------------------------
# Bounds
# ----------------------------------------------------------------------
def voigt(fv, L_phases):
    return sum(f * L for f, L in zip(fv, L_phases))


def reuss(fv, L_phases):
    return np.linalg.inv(sum(f * np.linalg.inv(L) for f, L in zip(fv, L_phases)))


def hashin_shtrikman(f_incl, matrix, inclusion):
    """
    Closed-form HS (k, mu) for two isotropic phases.
    matrix, inclusion : (k, mu);  f_incl : volume fraction of the inclusion.
    Upper bound: matrix = stiffer phase.  Lower bound: matrix = softer phase.
    """
    km, mm = matrix
    ki, mi = inclusion
    fm = 1.0 - f_incl
    k = km + f_incl / (1.0 / (ki - km) + 3.0 * fm / (3.0 * km + 4.0 * mm))
    mu = mm + f_incl / (1.0 / (mi - mm) + 6.0 * (km + 2.0 * mm) * fm
                        / (5.0 * mm * (3.0 * km + 4.0 * mm)))
    return k, mu


# ----------------------------------------------------------------------
# Estimates
# ----------------------------------------------------------------------
def mori_tanaka(fv, L_phases, matrix=0, a=(1.0, 1.0, 1.0), n=32):
    """Mori-Tanaka: P computed once, in the stiffness of the matrix phase."""
    fv = np.asarray(fv, float)
    L0 = np.asarray(L_phases[matrix], float)
    P = tenseurP(a[0], a[1], a[2], L0, n)
    Q0 = np.linalg.inv(P) - L0
    L_moy = sum(f * np.linalg.inv(L + Q0) for f, L in zip(fv, L_phases))
    return np.linalg.inv(L_moy) - Q0


def selfconsistent(fv, L_rot, a=(1.0, 1.0, 1.0), n=32,
                   tol=1e-12, max_iter=500, verbose=False):
    """
    Self-consistent estimate.
    fv : volume fractions (Nph,),  L_rot : phase stiffnesses (Nph, 6, 6)
    a  : ellipsoid semi-axes,      n : quadrature order of the P integral
    Returns (L_sc, Q_star, n_iter)
    """
    fv = np.asarray(fv, float)
    L_rot = np.asarray(L_rot, float)

    L_sc = voigt(fv, L_rot)                            # initialisation
    for it in range(1, max_iter + 1):
        P = tenseurP(a[0], a[1], a[2], L_sc, n)
        Q_star = np.linalg.inv(P) - L_sc
        L_moy = sum(f * np.linalg.inv(L + Q_star) for f, L in zip(fv, L_rot))
        L_new = np.linalg.inv(L_moy) - Q_star

        err = np.max(np.abs(L_new - L_sc)) / np.max(np.abs(L_new))
        L_sc = L_new
        if verbose:
            print(f"iter {it:3d}  err = {err:.3e}")
        if err < tol:
            break
    else:
        print("warning: self-consistent loop did not converge")

    P = tenseurP(a[0], a[1], a[2], L_sc, n)
    Q_star = np.linalg.inv(P) - L_sc
    return L_sc, Q_star, it