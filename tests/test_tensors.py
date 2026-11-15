import numpy as np
from helpers import triclinic, rand_rotation
from pyhom import (mandel_to_t4, t4_to_mandel, rotate_mandel, cubic_stiffness,
                   isotropic_stiffness, bulk_shear)


def test_roundtrip_mandel_t4_mandel():
    C = triclinic()
    assert np.linalg.norm(t4_to_mandel(mandel_to_t4(C)) - C) < 1e-12


def test_roundtrip_t4_mandel_t4():
    T = np.random.default_rng(1).normal(size=(3, 3, 3, 3))
    T = 0.5 * (T + T.transpose(1, 0, 2, 3)); T = 0.5 * (T + T.transpose(0, 1, 3, 2))
    T = 0.5 * (T + T.transpose(2, 3, 0, 1))
    assert np.abs(mandel_to_t4(t4_to_mandel(T)) - T).max() < 1e-12


def test_symmetries_of_t4():
    T = mandel_to_t4(triclinic())
    assert np.abs(T - T.transpose(1, 0, 2, 3)).max() < 1e-12      # minor
    assert np.abs(T - T.transpose(0, 1, 3, 2)).max() < 1e-12      # minor
    assert np.abs(T - T.transpose(2, 3, 0, 1)).max() < 1e-12      # major


def test_mandel_factors_and_quadratic_form():
    C = triclinic()
    T = mandel_to_t4(C)
    assert abs(T[0, 1, 0, 1] - C[3, 3] / 2) < 1e-12
    assert abs(T[0, 0, 1, 2] - C[0, 5] / np.sqrt(2)) < 1e-12
    eps = np.random.default_rng(2).normal(size=(3, 3)); eps = 0.5 * (eps + eps.T)
    em = np.array([eps[0, 0], eps[1, 1], eps[2, 2], np.sqrt(2) * eps[0, 1],
                   np.sqrt(2) * eps[0, 2], np.sqrt(2) * eps[1, 2]])
    assert abs(em @ C @ em - np.einsum("ij,ijkl,kl->", eps, T, eps)) < 1e-10


def test_cubic_stiffness():
    T = mandel_to_t4(cubic_stiffness(237.4, 134.7, 116.4))
    assert abs(T[0, 0, 0, 0] - 237.4) < 1e-12 and abs(T[1, 1, 1, 1] - 237.4) < 1e-12
    assert abs(T[0, 0, 1, 1] - 134.7) < 1e-12
    assert abs(T[0, 1, 0, 1] - 116.4) < 1e-12 and abs(T[1, 2, 1, 2] - 116.4) < 1e-12
    assert abs(T[0, 0, 0, 1]) < 1e-12


def test_isotropic_stiffness_and_bulk_shear():
    k, mu = 123.4, 56.7
    lam, d = k - 2 * mu / 3, np.eye(3)
    Tiso = lam * np.einsum("ij,kl->ijkl", d, d) + mu * (np.einsum("ik,jl->ijkl", d, d)
                                                       + np.einsum("il,jk->ijkl", d, d))
    L = isotropic_stiffness(k, mu)
    assert np.abs(mandel_to_t4(L) - Tiso).max() < 1e-10
    kk, mm = bulk_shear(L)
    assert abs(kk - k) < 1e-10 and abs(mm - mu) < 1e-10


def test_rotate_mandel():
    C, R = triclinic(3), rand_rotation(4)
    ref = t4_to_mandel(np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, mandel_to_t4(C)))
    assert np.abs(rotate_mandel(C, R) - ref).max() < 1e-10               # vs einsum
    assert np.abs(rotate_mandel(C, np.eye(3)) - C).max() < 1e-12         # identity
    assert np.abs(rotate_mandel(rotate_mandel(C, R), R.T) - C).max() < 1e-10   # inverse
    k0, k1 = bulk_shear(C), bulk_shear(rotate_mandel(C, R))              # invariants
    assert abs(k0[0] - k1[0]) < 1e-10 and abs(k0[1] - k1[1]) < 1e-10
