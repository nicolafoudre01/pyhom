# pyhom: library overview

Homogenization of the effective elastic stiffness of heterogeneous materials by mean-field schemes
(Mori-Tanaka, self-consistent) and classical bounds. Pure `numpy` (about 350 lines), 29 tests.
Everything rests on one object, the Hill tensor of an ellipsoid (see `polarization.md`).

## Convention (used everywhere)

Fourth-order tensors are **6×6 matrices in Mandel notation**, order (xx, yy, zz, xy, xz, yz), with a factor $\sqrt2$ on the shear components.
A double contraction is then a plain matrix product, and an inversion is `np.linalg.inv`.
Rotations are $3\times3$ matrices $R$ with $v_{sample}=R\,v_{local}$.

## Layout and dependencies

```
src/pyhom/
├── tensors.py        no dependency
├── hill.py           <- tensors
├── bounds.py         no dependency
├── schemes/
│   ├── mori_tanaka.py      <- hill
│   └── self_consistent.py  <- hill, bounds
└── phase.py          <- tensors
tests/   examples/   pyproject.toml
```
Dependencies go one way. Nothing in `pyhom` knows about plasticity, so other projects (e.g. `pyvpsc`) build on it.

## Module by module

| file | content |
|---|---|
| `tensors.py` | `mandel_to_t4`, `t4_to_mandel` (6×6 ↔ 3×3×3×3), `rotate_mandel(C, R)`, builders `cubic_stiffness(c11, c12, c44)`, `isotropic_stiffness(k, mu)`, and `bulk_shear(L)` (isotropic part $(k,\mu)$ of a stiffness) |
| `hill.py` | the Hill tensor: `fC0` (integrand), `int2v` / `int2v_vec` (double Gauss-Legendre integral, loop / vectorised), `tenseurP(a1, a2, a3, C0, n=32, vect=True)`. Details in `P_tensor.md` |
| `bounds.py` | `voigt`, `reuss` (any number of phases), `hashin_shtrikman(f_incl, matrix, inclusion)` (closed form, two isotropic phases) |
| `schemes/mori_tanaka.py` | `mori_tanaka(fv, L_phases, matrix=0, a, n)` |
| `schemes/self_consistent.py` | `selfconsistent(fv, L_rot, a, n, tol, max_iter)` returns `(L_sc, Q_star, n_iter)` |
| `phase.py` | data structures (below) |
| `__init__.py` | public API (everything above) and the convention docstring |

Arguments shared by the schemes: `fv` volume fractions, a list or array of stiffnesses (Nph, 6, 6), `a = (a1, a2, a3)` semi-axes of the (common) ellipsoidal shape, `n` quadrature order of the Hill integral.

## The two schemes

With $\mathbb Q^*=\mathbb P(\mathbb C_{ref},a)^{-1}-\mathbb C_{ref}$:

* **Mori-Tanaka**: reference medium = the matrix phase, $\mathbb P$ computed once:
  $\;\mathbb C_{MT}=\big\langle(\mathbb C_r+\mathbb Q^*)^{-1}\big\rangle^{-1}-\mathbb Q^*$.
* **Self-consistent**: reference medium = the effective medium itself, so the same formula is iterated,
  $\bar{\mathbb C}\leftarrow\big\langle(\mathbb C_r+\mathbb Q^*(\bar{\mathbb C}))^{-1}\big\rangle^{-1}-\mathbb Q^*(\bar{\mathbb C})$,
  starting from the Voigt average, until $\max|\Delta\bar{\mathbb C}|/\max|\bar{\mathbb C}|<$ `tol` ($10^{-12}$).
  It also returns $\mathbb Q^*$ (the interaction tensor), which other models reuse.

For spherical inclusions in an isotropic matrix, Mori-Tanaka with the softer (stiffer) phase as matrix equals the lower (upper) Hashin-Shtrikman bound (tested).
Voids work as inclusions in Mori-Tanaka (zero stiffness is fine, only the matrix enters $\mathbb P$).

## Data structures (`phase.py`)

* `Ellipsoid(a, R)`: semi-axes and the frame of the axes.
* `LinearElastic(L)`: a behaviour; anything with a `.stiffness()` method (local frame) can be used instead.
* `Phase(id, fraction, behaviour, shape, orientation, external, internal)`: one object.
  `orientation` (lattice frame) rotates the stiffness, `Phase.stiffness()` returns it in the sample frame; `shape` is separate.
  `external` / `internal` are free dicts for the variables of a scheme.
* `Aggregate(phases)`: checks that the fractions sum to 1 and gives array views (`fractions`, `stiffnesses`, `semi_axes`), `same_shape()`, and `get` / `set` for state variables.
  The schemes still take plain arrays: `selfconsistent(agg.fractions, agg.stiffnesses, a=agg.semi_axes[0])`.

## Typical use

```python
from pyhom import isotropic_stiffness, bulk_shear, mori_tanaka, selfconsistent

L = [isotropic_stiffness(100, 40), isotropic_stiffness(200, 100)]
fv = [0.5, 0.5]
L_sc, Q_star, n_iter = selfconsistent(fv, L)
L_mt = mori_tanaka(fv, L, matrix=0, a=(5, 1, 1), n=48)      # 5:1:1 inclusions
print(bulk_shear(L_sc), bulk_shear(L_mt))
```

## Tests (`pytest`, 29)

`test_tensors` (round trips, symmetries, rotation against `einsum`), `test_hill` (integrand, quadrature, BCC reference values, isotropic closed form, triclinic symmetry and rotation covariance, vectorised = loop), `test_bounds`, `test_schemes` (closed-form self-consistent equations, MT = HS, rotation covariance, voids), `test_phase`.
`examples/biphasic_estimates.py` plots all estimates and bounds against volume fraction.

## Extending

A new scheme goes in `schemes/`, with the same signature `(fv, L_phases, ..., a, n)` and `tenseurP` as its only link to the geometry.
Example: a differential scheme is a loop of small `mori_tanaka` steps, with the current effective stiffness as the matrix.

An interesting feature would be to extend this library to support finite strain and textural evolution.

## Limits

* Linear elasticity only; one common ellipsoid shape for all phases, with its axes along the global axes (a rotated shape needs $\mathbb C$ rotated into the ellipsoid frame, then $\mathbb P$ rotated back).
* $\mathbb C_{ref}$ must be positive definite. Self-consistent estimates with voids degenerate at the percolation threshold, where $\bar{\mathbb C}\to0$ and $\mathbb P^{-1}$ is singular.
