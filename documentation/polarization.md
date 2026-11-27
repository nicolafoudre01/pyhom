# Hill tensor $\mathbb P$ of an ellipsoid: numerical integration

Everything else in `pyhom` (Mori-Tanaka, self-consistent, interaction tensor $\mathbb Q^*$) is built on this one tensor.
Code: `pyhom/hill.py` &mdash; `fC0`, `int2v`, `int2v_vec`, `tenseurP`.

## 1. Definition

Ellipsoidal inclusion of semi-axes $(a_1,a_2,a_3)$ (along the global axes) in an infinite medium of stiffness $\mathbb C^0$ (anisotropic in general).
For a unit vector $\mathbf n$ define

$$
\xi_i = \frac{n_i}{a_i},\qquad
K_{ik}(\boldsymbol\xi) = C^0_{ijkl}\,\xi_j\,\xi_l,\qquad
\mathbf N(\boldsymbol\xi) = \mathbf K(\boldsymbol\xi)^{-1}
$$

($\mathbf K$ is the acoustic tensor). The Hill tensor is the average over the unit sphere $S^2$ of the symmetrised product $N_{ik}\xi_j\xi_l$, also known as the modified Green tensor associated to $C^0$:

$$
P_{ijkl} = \frac1{4\pi}\int_{S^2} \tfrac14\, F_{ijkl}^{C^0}(\mathbf n)\; d\Omega,
\qquad
F_{ijkl}^{C^0} = N_{ik}\,\xi_j\xi_l + N_{jk}\,\xi_i\xi_l + N_{il}\,\xi_j\xi_k + N_{jl}\,\xi_i\xi_k .
$$

Related quantities: Eshelby tensor $\mathbb S=\mathbb P:\mathbb C^0$; interaction tensor $\mathbb Q^*=\mathbb P^{-1}-\mathbb C^0$ (used by the schemes).

## 2. Parametrisation of the sphere (as in the code)

$\mathbf n=(\sin\theta\cos\varphi,\ \sin\theta\sin\varphi,\ \cos\theta)$ and

$$
x=\frac{1-\cos\theta}{2}\in[0,1],\qquad y=\frac{\varphi}{2\pi}\in[0,1]
\quad\Longrightarrow\quad \frac{d\Omega}{4\pi}=dx\,dy .
$$

So a uniform grid in $(x,y)$ is a uniform sampling of the sphere, and $\mathbb P=\tfrac14\int_0^1 \int_0^1 F\,dx\,dy$.
Since $F(-\mathbf n)=F(\mathbf n)$, half of the sphere is enough ($\varphi\in[0,\pi]$, i.e. $y\in[0,\tfrac12]$):

$$
\boxed{\;\mathbb P=\frac12\int_0^1 \int_0^{1/2} F_{ijkl}(x,y)\;dy\,dx\;}
$$

(the factor $\tfrac12$ is `factor = 0.5`, the upper limit $\tfrac12$ is `ymax = 0.5` in `tenseurP`).

## 3. Quadrature

Tensor-product Gauss-Legendre with $n$ points per direction (`np.polynomial.legendre.leggauss`). With nodes $g_p$ and weights $w_p$ on $[-1,1]$:

$$
x_p=\tfrac{1+g_p}{2},\ w^x_p=\tfrac{w_p}{2};\qquad
y_q=\tfrac{1+g_q}{4},\ w^y_q=\tfrac{w_q}{4};\qquad
\mathbb P\simeq\frac12\sum_{p=1}^{n}\sum_{q=1}^{n} w^x_p\,w^y_q\;F(x_p,y_q).
$$

$n^2$ evaluations of $F$ (default $n=32$: 1024 points).

## 4. Implementation (Mandel notation)

Tensors are $6\times6$ Mandel matrices (order xx, yy, zz, xy, xz, yz): $\tilde P_{IJ}=W_IW_J\,P_{ijkl}$ with $W=1$ for $I\in\{1,2,3\}$ and $W=\sqrt2$ for shears.

| step | in `fC0(x, y, C4, a)` |
|---|---|
| once, outside the integral | `C4 = mandel_to_t4(C0)` ($\mathbb C^0\to C^0_{ijkl}$) |
| $\theta=\arccos(1-2x)$, $\varphi=2\pi y$ | `theta`, `phi` |
| $\boldsymbol\xi=(\sin\theta\cos\varphi/a_1,\ \sin\theta\sin\varphi/a_2,\ \cos\theta/a_3)$ | `X` |
| $K_{ik}=C^0_{ijkl}\xi_j\xi_l$ | `np.einsum("ijkl,...j,...l->...ik", C4, X, X)` |
| $\mathbf N=\mathbf K^{-1}$ | `np.linalg.inv` |
| four-term sum $F_{IJ}$, $I=(i_1,i_2)$, $J=(j_1,j_2)$ | `CXX[i1,j1]*X[i2]*X[j2] + CXX[i2,j1]*X[i1]*X[j2] + CXX[i1,j2]*X[i2]*X[j1] + CXX[i2,j2]*X[i1]*X[j1]` |
| Mandel factors | `RES * W[:,None] * W[None,:]` |

### Vectorisation
`fC0` accepts scalars $(x,y)$ (returns $6\times6$) or arrays of shape $(N,)$ (returns $(N,6,6)$): same algebra, the loop over the points is done by the `...` of `einsum` and by the stacked `np.linalg.inv`.
`int2v_vec` calls `fC0` once on the whole $n\times n$ grid and sums with one `einsum`.

```python
ptensor(a1, a2, a3, C0, n=32, vect=True)    # vectorised (default), about 3 ms at n = 32
ptensor(a1, a2, a3, C0, n=32, vect=False)   # point-by-point loop (reference), about 50 ms
```

Both give the same result to rounding ($\sim10^{-15}$ relative).

## 5. Checks (in `tests/test_hill.py`)

* **Isotropic sphere**, closed form, to $10^{-17}$:
  $\;\mathbb P=\alpha\,\mathbb J+\beta\,\mathbb K,\quad \alpha=\dfrac{1}{3k+4\mu},\quad \beta=\dfrac{3(k+2\mu)}{5\mu(3k+4\mu)}$
  ($\mathbb J=\tfrac13\mathbf 1\otimes\mathbf 1$, $\mathbb K=\mathbb I-\mathbb J$).
* **Cubic (BCC) sphere**: $P_{1111}=0.00283395$, $P_{1122}=-0.00081998$, $P_{1212}=0.00117724$ for $(c_{11},c_{12},c_{44})=(237.4,134.7,116.4)$ (reference of the C++ code).
* Symmetric and positive definite for a triclinic $\mathbb C^0$; eigenvalues of $\mathbb S=\mathbb P:\mathbb C^0$ in $]0,1[$.
* Rotation covariance (sphere: any $R$; ellipsoid: $\pi$-rotations about the axes), axis permutation, $a\to\lambda a$ invariance, vectorised = loop.

## 6. Accuracy

Relative error against a $256\times256$ reference (cubic stiffness above):

| shape $a_1\!:\!a_2\!:\!a_3$ | $n=16$ | $n=32$ | $n=64$ |
|---|---|---|---|
| 5 : 1 : 1 | $7\cdot10^{-4}$ | $4\cdot10^{-6}$ | $2\cdot10^{-10}$ |
| 10 : 1 : 1 | $3\cdot10^{-3}$ | $2\cdot10^{-4}$ | $1\cdot10^{-6}$ |
| 1 : 1 : 0.1 (disc) | $5\cdot10^{-2}$ | $3\cdot10^{-3}$ | $2\cdot10^{-5}$ |

Spheres and moderate aspect ratios converge fast; needles and discs need a larger $n$ (the integrand becomes sharply peaked).

## 7. Limits

* The ellipsoid axes are the global axes. For a rotated ellipsoid: rotate $\mathbb C^0$ into the ellipsoid frame, compute $\mathbb P$, rotate back (not implemented).
* $\mathbb C^0$ must be positive definite so that $\mathbf K(\boldsymbol\xi)$ is invertible in every direction. A void medium ($\mathbb C^0=0$) is not allowed; voids as *inclusions* are fine (they do not enter $\mathbb P$).
* Nearly incompressible media (finite penalty bulk modulus, as in the Hutchinson treatment) are fine: $\mathbb P$ stays finite when $k\to\infty$.
