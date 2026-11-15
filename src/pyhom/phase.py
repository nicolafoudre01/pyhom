"""
Data structures: Phase (one object) and Aggregate (a set of phases, with array views).

Rotations are 3x3 matrices R with  v_sample = R @ v_local.
The lattice `orientation` rotates the stiffness; the `shape` frame is the frame of the
ellipsoid axes (the two are different objects and evolve differently).
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .tensors import rotate_mandel


@dataclass
class Ellipsoid:
    """Morphology: semi-axes a and the frame of the axes (R: shape frame -> sample frame)."""
    a: tuple = (1.0, 1.0, 1.0)
    R: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class LinearElastic:
    """Behaviour: a stiffness in the local (lattice) frame, 6x6 Mandel."""
    L: np.ndarray

    def stiffness(self):
        return self.L


@dataclass
class Phase:
    id: Any                                   # label (name or index)
    fraction: float                           # volume fraction
    behaviour: Any                            # object with .stiffness() (local frame)
    shape: Ellipsoid = field(default_factory=Ellipsoid)
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))   # lattice -> sample
    external: dict = field(default_factory=dict)   # imposed by the loading / the scheme
    internal: dict = field(default_factory=dict)   # state variables of the scheme

    def stiffness(self):
        """Stiffness in the SAMPLE frame (rotated by the lattice orientation)."""
        return rotate_mandel(self.behaviour.stiffness(), self.orientation)


class Aggregate:
    """Phases with array views: fractions (N,), stiffnesses (N,6,6), semi-axes (N,3)."""

    def __init__(self, phases):
        self.phases = list(phases)
        f = self.fractions
        if abs(f.sum() - 1.0) > 1e-9:
            raise ValueError(f"volume fractions sum to {f.sum()}, not 1")

    @property
    def ids(self):
        return [p.id for p in self.phases]

    @property
    def fractions(self):
        return np.array([p.fraction for p in self.phases])

    @property
    def stiffnesses(self):
        return np.array([p.stiffness() for p in self.phases])

    @property
    def semi_axes(self):
        return np.array([p.shape.a for p in self.phases])

    def same_shape(self):
        """True if all phases share one ellipsoid (semi-axes and frame)."""
        a0, R0 = self.phases[0].shape.a, self.phases[0].shape.R
        return all(np.allclose(p.shape.a, a0) and np.allclose(p.shape.R, R0)
                   for p in self.phases)

    def get(self, key, where="internal"):
        """Stack a state variable of all phases into one array."""
        return np.array([getattr(p, where)[key] for p in self.phases])

    def set(self, key, values, where="internal"):
        for p, v in zip(self.phases, values):
            getattr(p, where)[key] = v

    def __len__(self):
        return len(self.phases)
