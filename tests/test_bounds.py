import numpy as np
from pyhom import isotropic_stiffness, bulk_shear, voigt, reuss, hashin_shtrikman


def test_voigt_reuss_single_phase_and_order():
    L = isotropic_stiffness(100, 40)
    assert np.abs(voigt([1.0], [L]) - L).max() < 1e-12
    assert np.abs(reuss([1.0], [L]) - L).max() < 1e-12
    Lp = [isotropic_stiffness(100, 40), isotropic_stiffness(200, 100)]
    f = [0.5, 0.5]
    assert np.linalg.eigvalsh(voigt(f, Lp) - reuss(f, Lp)).min() > 0


def test_hashin_shtrikman_between_voigt_reuss_and_limits():
    soft, stiff, f = (100.0, 40.0), (200.0, 100.0), 0.4
    Lp = [isotropic_stiffness(*soft), isotropic_stiffness(*stiff)]
    kv, mv = bulk_shear(voigt([1 - f, f], Lp))
    kr, mr = bulk_shear(reuss([1 - f, f], Lp))
    kl, ml = hashin_shtrikman(f, soft, stiff)          # soft matrix: lower bound
    ku, mu_ = hashin_shtrikman(1 - f, stiff, soft)     # stiff matrix: upper bound
    assert kr <= kl <= ku <= kv and mr <= ml <= mu_ <= mv
    assert hashin_shtrikman(0.0, soft, stiff) == soft  # no inclusion: matrix
