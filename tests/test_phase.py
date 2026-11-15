import numpy as np
import pytest
from helpers import rand_rotation
from pyhom import (Phase, Ellipsoid, LinearElastic, Aggregate, selfconsistent,
                   isotropic_stiffness, cubic_stiffness, rotate_mandel, bulk_shear)


def test_aggregate_arrays_and_selfconsistent():
    ph = [Phase("soft", .5, LinearElastic(isotropic_stiffness(100, 40))),
          Phase("stiff", .5, LinearElastic(isotropic_stiffness(200, 100)))]
    agg = Aggregate(ph)
    assert agg.ids == ["soft", "stiff"] and agg.same_shape() and len(agg) == 2
    L = selfconsistent(agg.fractions, agg.stiffnesses, a=agg.semi_axes[0])[0]
    k, mu = bulk_shear(L)
    assert abs(k - 139.3512) < 1e-3 and abs(mu - 63.5754) < 1e-3


def test_orientation_rotates_stiffness():
    C = LinearElastic(cubic_stiffness(237.4, 134.7, 116.4))
    R = rand_rotation(1)
    p = Phase(0, 1.0, C, orientation=R)
    assert np.abs(p.stiffness() - rotate_mandel(C.L, R)).max() < 1e-12


def test_fractions_check_state_and_shapes():
    C = LinearElastic(isotropic_stiffness(1, 1))
    with pytest.raises(ValueError):
        Aggregate([Phase("a", .3, C)])
    agg = Aggregate([Phase(i, .5, C) for i in range(2)])
    agg.set("sigma", np.zeros((2, 6)))
    assert agg.get("sigma").shape == (2, 6)
    agg.phases[1].shape = Ellipsoid(a=(5., 1., 1.))
    assert not agg.same_shape()
