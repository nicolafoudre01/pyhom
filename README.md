# pyhom

Homogenization of the effective elastic properties of heterogeneous materials.
Only depends on `numpy`.

```python
from pyhom import isotropic_stiffness, bulk_shear, mori_tanaka, selfconsistent

L = [isotropic_stiffness(100, 40), isotropic_stiffness(200, 100)]
fv = [0.5, 0.5]
print(bulk_shear(selfconsistent(fv, L)[0]))
print(bulk_shear(mori_tanaka(fv, L, matrix=0)))
```

**Convention**: fourth-order tensors are 6x6 Mandel matrices, MFront ordering
(xx, yy, zz, xy, xz, yz), factor sqrt(2) on shear components.

Install and test:

    pip install -e .[dev]
    pytest
