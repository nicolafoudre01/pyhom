import numpy as np
from helpers import rand_rotation
from pyhom import (selfconsistent, moritanaka, voigt, reuss, hashin_shtrikman, bulk_shear,
                   isotropic_stiffness, cubic_stiffness, rotate_mandel)

SOFT, STIFF = (100.0, 40.0), (200.0, 100.0)


# ---------------- self-consistent ----------------
def test_sc_trivial_cases():
    L1 = isotropic_stiffness(*SOFT)
    assert np.abs(selfconsistent([1.0], [L1])[0] - L1).max() < 1e-10
    assert np.abs(selfconsistent([0.3, 0.7], [L1, L1])[0] - L1).max() < 1e-10


def test_sc_sphere_closed_form_equations():
    f = [0.4, 0.6]
    ph = [SOFT, STIFF]
    Lsc = selfconsistent(f, [isotropic_stiffness(*p) for p in ph])[0]
    k, mu = bulk_shear(Lsc)
    rk = sum(fi * (ki - k) / (ki + 4 * mu / 3) for fi, (ki, mi) in zip(f, ph))
    rm = sum(fi * (mi - mu) / (mi + mu * (9 * k + 8 * mu) / (6 * (k + 2 * mu))) for fi, (ki, mi) in zip(f, ph))
    assert abs(rk) < 1e-8 and abs(rm) < 1e-8
    assert np.abs(Lsc - isotropic_stiffness(k, mu)).max() < 1e-8                 # isotropic


def test_sc_bounds_permutation_elongated_and_rotation():
    f = [0.4, 0.6]
    Lp = [isotropic_stiffness(*SOFT), isotropic_stiffness(*STIFF)]
    Lv, Lr = voigt(f, Lp), reuss(f, Lp)
    Lsc = selfconsistent(f, Lp)[0]
    assert np.linalg.eigvalsh(Lv - Lsc).min() > 0 and np.linalg.eigvalsh(Lsc - Lr).min() > 0
    assert np.abs(selfconsistent(f[::-1], Lp[::-1])[0] - Lsc).max() < 1e-10
    Le = selfconsistent(f, Lp, a=(5., 1., 1.), n=48)[0]                          # 5:1:1
    assert np.abs(Le - Le.T).max() < 1e-8
    assert max(abs(Le[1, 1] - Le[2, 2]), abs(Le[0, 1] - Le[0, 2])) / np.abs(Le).max() < 1e-8
    assert np.linalg.eigvalsh(Lv - Le).min() > 0 and np.linalg.eigvalsh(Le - Lr).min() > 0
    R, C = rand_rotation(7), cubic_stiffness(237.4, 134.7, 116.4)                # rotation covariance
    C2 = 0.7 * C + 20 * np.eye(6)
    A = selfconsistent([0.5, 0.5], [C, C2])[0]
    B = selfconsistent([0.5, 0.5], [rotate_mandel(C, R), rotate_mandel(C2, R)])[0]
    assert np.abs(B - rotate_mandel(A, R)).max() / np.abs(A).max() < 1e-8


# ---------------- Mori-Tanaka ----------------
def test_mt_spheres_equal_hashin_shtrikman():
    Ls, Lh = isotropic_stiffness(*SOFT), isotropic_stiffness(*STIFF)
    f = 0.35                                                       # fraction of the stiff phase
    k, mu = bulk_shear(moritanaka([1 - f, f], [Ls, Lh], 0))       # soft matrix: HS lower
    kh, mh = hashin_shtrikman(f, SOFT, STIFF)
    assert abs(k - kh) < 1e-10 and abs(mu - mh) < 1e-10
    k, mu = bulk_shear(moritanaka([1 - f, f], [Ls, Lh], 1))       # stiff matrix: HS upper
    kh, mh = hashin_shtrikman(1 - f, STIFF, SOFT)
    assert abs(k - kh) < 1e-10 and abs(mu - mh) < 1e-10


def test_mt_limits_and_elongated():
    Ls, Lh = isotropic_stiffness(*SOFT), isotropic_stiffness(*STIFF)
    assert np.abs(moritanaka([1., 0.], [Ls, Lh], 0) - Ls).max() < 1e-10
    assert np.abs(moritanaka([0., 1.], [Ls, Lh], 0) - Lh).max() < 1e-10
    f = [0.65, 0.35]
    Le = moritanaka(f, [Ls, Lh], 0, a=(5., 1., 1.), n=48)
    assert np.abs(Le - Le.T).max() < 1e-8
    assert np.linalg.eigvalsh(voigt(f, [Ls, Lh]) - Le).min() > 0
    assert np.linalg.eigvalsh(Le - reuss(f, [Ls, Lh])).min() > 0


def test_mt_voids_finite_stiffness():
    """Voids (zero stiffness): MT stays finite and decreases with porosity."""
    Lm, Lv = isotropic_stiffness(*STIFF), np.zeros((6, 6))
    ks = [bulk_shear(moritanaka([1 - p, p], [Lm, Lv], 0))[0] for p in (0.0, 0.1, 0.3, 0.6)]
    assert all(np.isfinite(ks)) and all(np.diff(ks) < 0)
