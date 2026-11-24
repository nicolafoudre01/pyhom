import numpy as np

from ..hill import ptensor


def moritanaka(fv, L_phases, matrix=0, a=(1.0, 1.0, 1.0), n=32):
    """Mori-Tanaka: P computed once, in the stiffness of the matrix phase."""
    fv = np.asarray(fv, float)
    L0 = np.asarray(L_phases[matrix], float)
    P = ptensor(a[0], a[1], a[2], L0, n)
    Q0 = np.linalg.inv(P) - L0
    L_moy = sum(f * np.linalg.inv(L + Q0) for f, L in zip(fv, L_phases))
    return np.linalg.inv(L_moy) - Q0
