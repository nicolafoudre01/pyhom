"""Classical bounds: Voigt, Reuss and Hashin-Shtrikman."""
import numpy as np


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
