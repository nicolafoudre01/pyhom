import numpy as np

from ..bounds import voigt
from ..hill import tenseurP


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
