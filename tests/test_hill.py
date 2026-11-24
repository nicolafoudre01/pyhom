import numpy as np
import pytest
from helpers import spd_triclinic, rand_rotation
from pyhom import (fC0, int2v, int2v_vec, ptensor, mandel_to_t4, t4_to_mandel, rotate_mandel,
                   cubic_stiffness, isotropic_stiffness)


# ---------------- int2v ----------------
def test_int2v_polynomial_and_matrix_valued():
    assert abs(int2v(lambda x, y: x**3 * y**2 + 1.0, 0, 2, 1, 3, 8) - (4.0 * 26 / 3 + 4.0)) < 1e-10
    f = lambda x, y: np.array([[x * y, 1.0], [np.cos(x), y ** 2]])
    ref = np.array([[0.25, 1.0], [np.sin(1.0), 1 / 3]])
    assert np.abs(int2v(f, 0, 1, 0, 1, 16) - ref).max() < 1e-12
    assert abs(int2v(lambda x, y: 1.0, 0, 1, 0, 0.5, 4) - 0.5) < 1e-14


# ---------------- fC0 ----------------
def test_fC0_properties():
    C = spd_triclinic(5)
    C4, a = mandel_to_t4(C), (2.0, 1.0, 0.7)
    x, y = 0.3, 0.17
    F = fC0(x, y, C4, a)
    assert np.abs(F - F.T).max() < 1e-12                                  # symmetric
    th, ph = np.arccos(1 - 2 * x), 2 * np.pi * y                         # direct formula
    X = np.array([np.sin(th) * np.cos(ph) / a[0], np.sin(th) * np.sin(ph) / a[1], np.cos(th) / a[2]])
    N = np.linalg.inv(np.einsum("ijkl,j,l->ik", C4, X, X))
    T = (np.einsum("ik,j,l->ijkl", N, X, X) + np.einsum("jk,i,l->ijkl", N, X, X)
         + np.einsum("il,j,k->ijkl", N, X, X) + np.einsum("jl,i,k->ijkl", N, X, X))
    assert np.abs(F - t4_to_mandel(T)).max() < 1e-12
    assert np.abs(F - fC0(1 - x, y + 0.5, C4, a)).max() < 1e-12          # xi -> -xi
    assert np.abs(fC0(x, y, 2 * C4, a) - F / 2).max() < 1e-12            # homogeneity


# ---------------- tenseurP ----------------
def test_P_bcc_reference_cpp():
    """Reference values of the C++ TesttenseurP (6 digits)."""
    C = cubic_stiffness(237.4, 134.7, 116.4)
    p11, p12, p66 = 0.00283395, -0.00081998, 2 * 0.00117724
    Pref = np.diag([p11] * 3 + [p66] * 3)
    Pref[:3, :3] += p12 * (np.ones((3, 3)) - np.eye(3))
    assert np.abs(ptensor(1, 1, 1, C) - Pref).max() < 1e-8


def test_P_isotropic_sphere_closed_form():
    k, mu = 100.0, 40.0
    alpha = 1 / (3 * k + 4 * mu)
    beta = 3 * (k + 2 * mu) / (5 * mu * (3 * k + 4 * mu))
    J = np.zeros((6, 6)); J[:3, :3] = 1 / 3
    Pex = alpha * J + beta * (np.eye(6) - J)
    assert np.abs(ptensor(1, 1, 1, isotropic_stiffness(k, mu)) - Pex).max() < 1e-12


def test_P_scale_permutation_convergence():
    C = cubic_stiffness(237.4, 134.7, 116.4)
    assert np.abs(ptensor(3, 3, 3, C) - ptensor(1, 1, 1, C)).max() < 1e-12     # radius
    P1, P2 = ptensor(5, 1, 1, C, 64), ptensor(1, 5, 1, C, 64)
    idx = [1, 0, 2, 3, 5, 4]                                                    # x <-> y
    assert np.abs(P1[np.ix_(idx, idx)] - P2).max() < 1e-8
    assert np.abs(ptensor(5, 1, 1, C, 32) - P1).max() / np.abs(P1).max() < 1e-4  # convergence
    ev = np.linalg.eigvals(ptensor(5, 2, 1, C, 64) @ C).real                    # Eshelby S = P C
    assert ev.min() > 0 and ev.max() < 1


def test_P_triclinic_symmetry_and_rotation():
    C = spd_triclinic(1)
    a = (5.0, 2.0, 1.0)
    P = ptensor(*a, C, 48)
    assert np.max(np.abs(P - P.T)) / np.max(np.abs(P)) < 1e-8
    assert np.linalg.eigvalsh(0.5 * (P + P.T)).min() > 0
    R = rand_rotation(5)                                                        # sphere: any R
    P0, P1 = ptensor(1, 1, 1, C, 48), ptensor(1, 1, 1, rotate_mandel(C, R), 48)
    assert np.abs(P1 - rotate_mandel(P0, R)).max() / np.abs(P0).max() < 1e-8
    for Rpi in (np.diag([1., -1, -1]), np.diag([-1., 1, -1]), np.diag([-1., -1, 1])):   # ellipsoid, pi rotations
        P1 = ptensor(*a, rotate_mandel(C, Rpi), 48)
        assert np.abs(P1 - rotate_mandel(P, Rpi)).max() / np.abs(P).max() < 1e-8


# ---------------- vectorised integration (same algebra as the loop) ----------------
from pyhom import int2v_vec


def test_fC0_array_equals_point_by_point():
    C4, a = mandel_to_t4(spd_triclinic(2)), (2.0, 1.0, 0.7)
    x, y = np.array([0.1, 0.4, 0.8, 0.95]), np.array([0.05, 0.3, 0.45, 0.2])
    F = fC0(x, y, C4, a)
    assert F.shape == (4, 6, 6) and fC0(x[0], y[0], C4, a).shape == (6, 6)
    for i in range(4):
        assert np.abs(F[i] - fC0(x[i], y[i], C4, a)).max() < 1e-14
    G = fC0(x.reshape(2, 2), y.reshape(2, 2), C4, a)                      # any batch shape
    assert G.shape == (2, 2, 6, 6) and np.abs(G.reshape(4, 6, 6) - F).max() < 1e-14


def test_int2v_vec_equals_int2v():
    f = lambda x, y: np.array([[x * y, 1.0], [np.cos(x), y ** 2]])
    fv = lambda x, y: np.stack([np.stack([x * y, np.ones_like(x)], axis=-1),
                                np.stack([np.cos(x), y ** 2], axis=-1)], axis=-2)
    assert np.abs(int2v_vec(fv, 0, 1, 0, 1, 16) - int2v(f, 0, 1, 0, 1, 16)).max() < 1e-14
    assert abs(int2v_vec(lambda x, y: x ** 3 * y ** 2 + 1.0, 0, 2, 1, 3, 8) - (4.0 * 26 / 3 + 4.0)) < 1e-10


@pytest.mark.parametrize("a,n", [((1, 1, 1), 32), ((5, 1, 1), 48), ((5, 2, 1), 24)])
def test_tenseurP_vect_flag(a, n):
    for C in (cubic_stiffness(237.4, 134.7, 116.4), spd_triclinic(3)):
        P1, P2 = ptensor(*a, C, n), ptensor(*a, C, n, vect=False)
        assert np.abs(P1 - P2).max() / np.abs(P2).max() < 1e-13
        assert np.array_equal(P1, ptensor(*a, C, n, vect=True))          # vect=True is the default
