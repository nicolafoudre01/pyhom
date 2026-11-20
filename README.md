![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative)

# pyhom

Homogenization of the effective elastic properties of heterogeneous materials.
Only depends on `numpy`.

```python
from pyhom import isotropic_stiffness, bulk_shear, moritanaka, selfconsistent

L = [isotropic_stiffness(100, 40), isotropic_stiffness(200, 100)]
fv = [0.5, 0.5]
print(bulk_shear(selfconsistent(fv, L)[0]))
print(bulk_shear(moritanaka(fv, L, matrix=0)))
```

**Convention**: fourth-order tensors are 6x6 Mandel matrices, MFront ordering
(xx, yy, zz, xy, xz, yz), factor sqrt(2) on shear components.

Install and test:

    pip install -e .[dev]
    pytest

For usage as a library in another project, run :

    pip install git+https://github.com/nicolafoudre01/pyhom.git

## License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details
