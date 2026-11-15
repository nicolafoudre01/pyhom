import numpy as np


def triclinic(seed=0):
    """Random symmetric 6x6 Mandel matrix (21 independent components)."""
    A = np.random.default_rng(seed).normal(size=(6, 6))
    return A + A.T


def spd_triclinic(seed=0, scale=100.0):
    A = np.random.default_rng(seed).normal(size=(6, 6))
    return scale * (A @ A.T / 6.0 + 0.5 * np.eye(6))


def rand_rotation(seed=0):
    q = np.random.default_rng(seed).normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
