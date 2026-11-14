"""
Unit tests and graphs.

    python unitary_tests.py          # run all the tests
    python unitary_tests.py plot     # tests, then the biphasic graph:
                                     # saved to estimates_biphasic.png and shown
    python unitary_tests.py plot fig.png   # same, with another file name

Needs ptensor.py (P tensor, conversions) and functions.py (estimates, bounds).
"""
import sys
import numpy as np

from ptensor import (mandel_to_t4, t4_to_mandel, rotate_mandel, cubic_stiffness,
                     fC0, int2v, tenseurP)
from functions import (isotropic_stiffness, bulk_shear, voigt, reuss,
                       hashin_shtrikman, mori_tanaka, selfconsistent)


# ======================================================================
# Utilities
# ======================================================================
def _report(name, err, tol):
    status = "OK " if err < tol else "FAIL"
    print(f"[{status}] {name:58s} err = {err:.2e}  (tol {tol:.0e})")
    assert err < tol, f"{name} non OK"


def _triclinic(seed=0):
    A = np.random.default_rng(seed).normal(size=(6, 6))
    return A + A.T


def _rand_rotation(seed=0):
    q = np.random.default_rng(seed).normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


# ======================================================================
# ptensor.py
# ======================================================================
def test_convert():
    C = _triclinic()
    T = mandel_to_t4(C)
    _report("mandel_to_t4 -> t4_to_mandel round trip", np.linalg.norm(t4_to_mandel(T) - C), 1e-12)
    T2 = np.random.default_rng(1).normal(size=(3, 3, 3, 3))
    T2 = 0.5 * (T2 + T2.transpose(1, 0, 2, 3)); T2 = 0.5 * (T2 + T2.transpose(0, 1, 3, 2))
    T2 = 0.5 * (T2 + T2.transpose(2, 3, 0, 1))
    _report("t4_to_mandel -> mandel_to_t4 round trip", np.abs(mandel_to_t4(t4_to_mandel(T2)) - T2).max(), 1e-12)
    sym = max(np.abs(T - T.transpose(1, 0, 2, 3)).max(), np.abs(T - T.transpose(0, 1, 3, 2)).max(),
              np.abs(T - T.transpose(2, 3, 0, 1)).max())
    _report("mandel_to_t4: minor and major symmetries", sym, 1e-12)
    e = max(abs(T[0, 1, 0, 1] - C[3, 3] / 2), abs(T[0, 0, 1, 2] - C[0, 5] / np.sqrt(2)))
    _report("Mandel factors (1212 -> /2, 1123 -> /sqrt2)", e, 1e-12)
    eps = np.random.default_rng(2).normal(size=(3, 3)); eps = 0.5 * (eps + eps.T)
    em = np.array([eps[0, 0], eps[1, 1], eps[2, 2], np.sqrt(2) * eps[0, 1],
                   np.sqrt(2) * eps[0, 2], np.sqrt(2) * eps[1, 2]])
    e = abs(em @ C @ em - np.einsum("ij,ijkl,kl->", eps, T, eps))
    _report("quadratic form eps:C:eps (Mandel vs tensor)", e, 1e-10)


def test_cubic_stiffness():
    T = mandel_to_t4(cubic_stiffness(237.4, 134.7, 116.4))
    e = max(abs(T[0, 0, 0, 0] - 237.4), abs(T[1, 1, 1, 1] - 237.4), abs(T[0, 0, 1, 1] - 134.7),
            abs(T[0, 1, 0, 1] - 116.4), abs(T[1, 2, 1, 2] - 116.4), abs(T[0, 0, 0, 1]))
    _report("cubic_stiffness: C1111, C1122, C1212 and zeros", e, 1e-12)


def test_rotate():
    C, R = _triclinic(3), _rand_rotation(4)
    ref = t4_to_mandel(np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, mandel_to_t4(C)))
    _report("rotate_mandel vs einsum rotation of the tensor", np.abs(rotate_mandel(C, R) - ref).max(), 1e-10)
    _report("rotate_mandel with identity", np.abs(rotate_mandel(C, np.eye(3)) - C).max(), 1e-12)
    back = rotate_mandel(rotate_mandel(C, R), R.T)
    _report("rotate_mandel then inverse rotation", np.abs(back - C).max(), 1e-10)
    k0, k1 = bulk_shear(C), bulk_shear(rotate_mandel(C, R))
    _report("isotropic part (k, mu) invariant by rotation", max(abs(k0[0] - k1[0]), abs(k0[1] - k1[1])), 1e-10)


def test_int2v():
    g = lambda x, y: x**3 * y**2 + 1.0
    _report("int2v: scalar polynomial on [0,2]x[1,3]",
            abs(int2v(g, 0, 2, 1, 3, 8) - (4.0 * 26 / 3 + 4.0)), 1e-10)
    f = lambda x, y: np.array([[x * y, 1.0], [np.cos(x), y ** 2]])
    ref = np.array([[0.25, 1.0], [np.sin(1.0), 1 / 3]])
    _report("int2v: matrix-valued integrand on [0,1]^2", np.abs(int2v(f, 0, 1, 0, 1, 16) - ref).max(), 1e-12)
    _report("int2v: area of [0,1]x[0,0.5]", abs(int2v(lambda x, y: 1.0, 0, 1, 0, 0.5, 4) - 0.5), 1e-14)


def test_fC0():
    C = _triclinic(5); C = C @ C.T / 6 + 100 * np.eye(6)       # SPD triclinic
    C4, a = mandel_to_t4(C), (2.0, 1.0, 0.7)
    x, y = 0.3, 0.17
    F = fC0(x, y, C4, a)
    _report("fC0 symmetric 6x6", np.abs(F - F.T).max(), 1e-12)
    th, ph = np.arccos(1 - 2 * x), 2 * np.pi * y
    X = np.array([np.sin(th) * np.cos(ph) / a[0], np.sin(th) * np.sin(ph) / a[1], np.cos(th) / a[2]])
    N = np.linalg.inv(np.einsum("ijkl,j,l->ik", C4, X, X))
    T = (np.einsum("ik,j,l->ijkl", N, X, X) + np.einsum("jk,i,l->ijkl", N, X, X)
         + np.einsum("il,j,k->ijkl", N, X, X) + np.einsum("jl,i,k->ijkl", N, X, X))
    _report("fC0 vs direct 3x3x3x3 formula at one point", np.abs(F - t4_to_mandel(T)).max(), 1e-12)
    G = fC0(1 - x, y + 0.5, C4, a)
    _report("fC0(x, y) = fC0(1-x, y+1/2)  (xi -> -xi)", np.abs(F - G).max(), 1e-12)
    _report("fC0 homogeneity: C -> 2C gives fC0/2",
            np.abs(fC0(x, y, 2 * C4, a) - F / 2).max(), 1e-12)


def test_tenseurP():
    C = cubic_stiffness(237.4, 134.7, 116.4)
    p11, p12, p66 = 0.00283395, -0.00081998, 2 * 0.00117724
    Pref = np.diag([p11] * 3 + [p66] * 3)
    Pref[:3, :3] += p12 * (np.ones((3, 3)) - np.eye(3))
    _report("tenseurP BCC sphere vs C++ reference (6 digits)", np.abs(tenseurP(1, 1, 1, C) - Pref).max(), 1e-8)
    k, mu = 100.0, 40.0
    alpha = 1 / (3 * k + 4 * mu)
    beta = 3 * (k + 2 * mu) / (5 * mu * (3 * k + 4 * mu))
    J = np.zeros((6, 6)); J[:3, :3] = 1 / 3
    Pex = alpha * J + beta * (np.eye(6) - J)
    _report("tenseurP isotropic sphere vs closed form",
            np.abs(tenseurP(1, 1, 1, isotropic_stiffness(k, mu)) - Pex).max(), 1e-12)
    _report("tenseurP invariant under a -> 3a",
            np.abs(tenseurP(3, 3, 3, C) - tenseurP(1, 1, 1, C)).max(), 1e-12)
    P1, P2 = tenseurP(5, 1, 1, C, 64), tenseurP(1, 5, 1, C, 64)
    idx = [1, 0, 2, 3, 5, 4]
    _report("tenseurP axis permutation x<->y", np.abs(P1[np.ix_(idx, idx)] - P2).max(), 1e-8)
    _report("tenseurP 5:1:1 convergence n=32 vs n=64 (relative)",
            np.abs(tenseurP(5, 1, 1, C, 32) - P1).max() / np.abs(P1).max(), 1e-4)
    ev = np.linalg.eigvals(tenseurP(5, 2, 1, C, 64) @ C).real
    _report("eigenvalues of P*C in ]0,1[ (violation if > 0)", max(0.0, -ev.min(), ev.max() - 1.0), 1e-12)


# ======================================================================
# functions.py
# ======================================================================
def test_stiffness_helpers():
    k, mu = 123.4, 56.7
    lam, d = k - 2 * mu / 3, np.eye(3)
    Tiso = lam * np.einsum("ij,kl->ijkl", d, d) + mu * (np.einsum("ik,jl->ijkl", d, d)
                                                       + np.einsum("il,jk->ijkl", d, d))
    _report("isotropic_stiffness vs lambda, mu tensor",
            np.abs(mandel_to_t4(isotropic_stiffness(k, mu)) - Tiso).max(), 1e-10)
    kk, mm = bulk_shear(isotropic_stiffness(k, mu))
    _report("bulk_shear(isotropic_stiffness(k, mu)) = (k, mu)", max(abs(kk - k), abs(mm - mu)), 1e-10)


def test_selfconsistent():
    L1 = isotropic_stiffness(100.0, 40.0)
    Lsc, Q, it = selfconsistent([1.0], [L1])
    _report("selfconsistent: single phase returns its stiffness", np.abs(Lsc - L1).max(), 1e-10)
    Lsc, Q, it = selfconsistent([0.3, 0.7], [L1, L1])
    _report("selfconsistent: two identical phases", np.abs(Lsc - L1).max(), 1e-10)

    ph, f = [(100.0, 40.0), (200.0, 100.0)], [0.4, 0.6]
    Lsc, Q, it = selfconsistent(f, [isotropic_stiffness(*p) for p in ph])
    k, mu = bulk_shear(Lsc)
    rk = sum(fi * (ki - k) / (ki + 4 * mu / 3) for fi, (ki, mi) in zip(f, ph))
    rm = sum(fi * (mi - mu) / (mi + mu * (9 * k + 8 * mu) / (6 * (k + 2 * mu))) for fi, (ki, mi) in zip(f, ph))
    _report("selfconsistent sphere: bulk modulus equation residual", abs(rk), 1e-8)
    _report("selfconsistent sphere: shear modulus equation residual", abs(rm), 1e-8)
    _report("selfconsistent sphere: result is isotropic",
            np.abs(Lsc - isotropic_stiffness(k, mu)).max(), 1e-8)

    Lp = [isotropic_stiffness(*p) for p in ph]
    Lv, Lr = voigt(f, Lp), reuss(f, Lp)
    viol = max(0.0, -np.linalg.eigvalsh(Lv - Lsc).min(), -np.linalg.eigvalsh(Lsc - Lr).min())
    _report("selfconsistent between Reuss and Voigt bounds (violation)", viol, 1e-10)

    L2, _, _ = selfconsistent(f[::-1], Lp[::-1])
    _report("selfconsistent: invariant under phase permutation", np.abs(L2 - Lsc).max(), 1e-10)

    Le, _, _ = selfconsistent(f, Lp, a=(5., 1., 1.), n=48)
    _report("selfconsistent 5:1:1: symmetric", np.abs(Le - Le.T).max(), 1e-8)
    _report("selfconsistent 5:1:1: L22 = L33 and L12 = L13 (relative)",
            max(abs(Le[1, 1] - Le[2, 2]), abs(Le[0, 1] - Le[0, 2])) / np.abs(Le).max(), 1e-8)
    viol = max(0.0, -np.linalg.eigvalsh(Lv - Le).min(), -np.linalg.eigvalsh(Le - Lr).min())
    _report("selfconsistent 5:1:1: between Reuss and Voigt (violation)", viol, 1e-10)

    R = _rand_rotation(7)
    C = cubic_stiffness(237.4, 134.7, 116.4)
    C2 = 0.7 * C + 20 * np.eye(6)
    A = selfconsistent([0.5, 0.5], [C, C2], n=32)[0]
    B = selfconsistent([0.5, 0.5], [rotate_mandel(C, R), rotate_mandel(C2, R)], n=32)[0]
    _report("selfconsistent sphere: rotation covariance",
            np.abs(B - rotate_mandel(A, R)).max() / np.abs(A).max(), 1e-8)


def test_mori_tanaka():
    soft, stiff = (100.0, 40.0), (200.0, 100.0)
    Ls, Lh = isotropic_stiffness(*soft), isotropic_stiffness(*stiff)
    f = 0.35                                    # fraction of the stiff phase
    k, mu = bulk_shear(mori_tanaka([1 - f, f], [Ls, Lh], 0))
    kh, mh = hashin_shtrikman(f, soft, stiff)
    _report("Mori-Tanaka (soft matrix) = HS lower bound (k, mu)", max(abs(k - kh), abs(mu - mh)), 1e-10)
    k, mu = bulk_shear(mori_tanaka([1 - f, f], [Ls, Lh], 1))
    kh, mh = hashin_shtrikman(1 - f, stiff, soft)
    _report("Mori-Tanaka (stiff matrix) = HS upper bound (k, mu)", max(abs(k - kh), abs(mu - mh)), 1e-10)
    _report("Mori-Tanaka, no inclusion = matrix", np.abs(mori_tanaka([1., 0.], [Ls, Lh], 0) - Ls).max(), 1e-10)
    _report("Mori-Tanaka, all inclusion = inclusion", np.abs(mori_tanaka([0., 1.], [Ls, Lh], 0) - Lh).max(), 1e-10)
    Lmt = mori_tanaka([1 - f, f], [Ls, Lh], 0)
    k, mu = bulk_shear(Lmt)
    _report("Mori-Tanaka with spheres is isotropic", np.abs(Lmt - isotropic_stiffness(k, mu)).max(), 1e-8)
    Le = mori_tanaka([1 - f, f], [Ls, Lh], 0, a=(5., 1., 1.), n=48)
    viol = max(0.0, -np.linalg.eigvalsh(voigt([1 - f, f], [Ls, Lh]) - Le).min(),
               -np.linalg.eigvalsh(Le - reuss([1 - f, f], [Ls, Lh])).min())
    _report("Mori-Tanaka 5:1:1: symmetric", np.abs(Le - Le.T).max(), 1e-8)
    _report("Mori-Tanaka 5:1:1: between Reuss and Voigt (violation)", viol, 1e-10)


def test_bounds_ordering():
    fs, c = biphasic_curves((100.0, 40.0), (200.0, 100.0), n_f=11, n=24)
    chain = ["Reuss", "HS lower", "Self-consistent", "HS upper", "Voigt"]
    viol = 0.0
    for lo, hi in zip(chain[:-1], chain[1:]):
        viol = max(viol, np.max(c[lo] - c[hi] - 1e-9))
    _report("Reuss <= HS- <= self-consistent <= HS+ <= Voigt (k and mu)", max(viol, 0.0), 1e-9)
    viol = max(np.max(c["MT (soft matrix)"] - c["MT (stiff matrix)"] - 1e-9), 0.0)
    _report("MT(soft matrix) <= MT(stiff matrix)", viol, 1e-9)


# ======================================================================
# Graphs
# ======================================================================
def biphasic_curves(soft, stiff, n_f=21, n=24):
    """
    All estimates and bounds (k, mu) vs the volume fraction f of the stiff phase,
    for two isotropic phases with spherical inclusions.  soft, stiff : (k, mu)
    """
    fs = np.linspace(0.0, 1.0, n_f)
    Lp = [isotropic_stiffness(*soft), isotropic_stiffness(*stiff)]
    out = {name: [] for name in ("Voigt", "Reuss", "HS upper", "HS lower",
                                 "MT (soft matrix)", "MT (stiff matrix)", "Self-consistent")}
    for f in fs:
        fv = [1.0 - f, f]
        out["Voigt"].append(bulk_shear(voigt(fv, Lp)))
        out["Reuss"].append(bulk_shear(reuss(fv, Lp)))
        if 0.0 < f < 1.0:
            out["HS upper"].append(hashin_shtrikman(1 - f, stiff, soft))   # stiff matrix
            out["HS lower"].append(hashin_shtrikman(f, soft, stiff))       # soft matrix
        else:
            ref = soft if f == 0.0 else stiff
            out["HS upper"].append(ref); out["HS lower"].append(ref)
        out["MT (soft matrix)"].append(bulk_shear(mori_tanaka(fv, Lp, 0, n=n)))
        out["MT (stiff matrix)"].append(bulk_shear(mori_tanaka(fv, Lp, 1, n=n)))
        out["Self-consistent"].append(bulk_shear(selfconsistent(fv, Lp, n=n)[0]))
    return fs, {k: np.array(v) for k, v in out.items()}


def plot_biphasic(soft=(100.0, 40.0), stiff=(200.0, 100.0),
                  filename="estimates_biphasic.png", show=True):
    """Always saves the figure to `filename` (open the PNG in VS Code), then
    tries to display it if `show` is True (needs an interactive backend)."""
    import matplotlib.pyplot as plt
    fs, c = biphasic_curves(soft, stiff)
    style = {"Voigt": ("k", "-"), "Reuss": ("k", "--"),
             "HS upper": ("tab:blue", "-"), "HS lower": ("tab:blue", "--"),
             "MT (soft matrix)": ("tab:green", ":"), "MT (stiff matrix)": ("tab:orange", ":"),
             "Self-consistent": ("tab:red", "-")}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, name in enumerate(["Bulk modulus k", "Shear modulus mu"]):
        for key, (col, ls) in style.items():
            ax[j].plot(fs, c[key][:, j], color=col, ls=ls,
                       lw=2.2 if key == "Self-consistent" else 1.6, label=key)
        ax[j].set_xlabel("volume fraction of the stiff phase")
        ax[j].set_ylabel(name)
        ax[j].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Biphasic isotropic: soft (k, mu) = {soft}, stiff = {stiff}, spherical inclusions")
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=150)
        print(f"figure saved to {filename}")
    if show:
        plt.show()
    return fs, c


# ======================================================================
if __name__ == "__main__":
    for t in (test_convert, test_cubic_stiffness, test_rotate, test_int2v, test_fC0,
              test_tenseurP, test_stiffness_helpers, test_selfconsistent,
              test_mori_tanaka, test_bounds_ordering):
        print(f"--- {t.__name__}")
        t()
    print("\nAll tests OK")

    if "plot" in sys.argv:
        i = sys.argv.index("plot")
        name = sys.argv[i + 1] if len(sys.argv) > i + 1 else "estimates_biphasic.png"
        plot_biphasic(filename=name)