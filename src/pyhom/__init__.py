"""
pyhom -- homogenization of the effective properties of heterogeneous materials.

Convention (everywhere): fourth-order tensors are 6x6 matrices in Mandel notation, MFront
ordering (xx, yy, zz, xy, xz, yz), with a factor sqrt(2) on shear components.

    tensors   conversions, rotation, stiffness builders
    hill      Hill tensor of an ellipsoid (numerical integration)
    bounds    Voigt, Reuss, Hashin-Shtrikman
    schemes   Mori-Tanaka, self-consistent
    phase     Phase / Aggregate data structures
"""
from .tensors import (mandel_to_t4, t4_to_mandel, rotate_mandel, cubic_stiffness,
                      isotropic_stiffness, bulk_shear)
from .hill import fC0, int2v, ptensor
from .bounds import voigt, reuss, hashin_shtrikman
from .schemes import moritanaka, selfconsistent
from .phase import Phase, Ellipsoid, LinearElastic, Aggregate

__all__ = [
    "mandel_to_t4", "t4_to_mandel", "rotate_mandel", "cubic_stiffness",
    "isotropic_stiffness", "bulk_shear",
    "fC0", "int2v", "ptensor",
    "voigt", "reuss", "hashin_shtrikman",
    "moritanaka", "selfconsistent",
    "Phase", "Ellipsoid", "LinearElastic", "Aggregate",
]
__version__ = "0.1.0"
