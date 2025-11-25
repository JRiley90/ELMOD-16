import numpy as np
from .utils import make_constellation

def classify_constellation(a_k, phi_k):
    """
    Very simple heuristic classifier.
    Returns one of:
        'qam-like', 'apsk-like', 'ring-hybrid',
        'asymmetric-hybrid', 'clustered-multi-ring',
        'fractured-non-euclidean'
    """
    const = make_constellation(a_k, phi_k)
    x = const.real
    y = const.imag

    # radial distances
    r = np.sqrt(x**2 + y**2)

    # basic stats
    r_unique = len(np.unique(np.round(r, 2)))
    x_unique = len(np.unique(np.round(x, 2)))
    y_unique = len(np.unique(np.round(y, 2)))

    # symmetry checks
    sym_x = np.allclose(np.sort(x), np.sort(-x))
    sym_y = np.allclose(np.sort(y), np.sort(-y))

    if sym_x and sym_y and r_unique <= 3 and x_unique > 2 and y_unique > 2:
        return "qam-like"

    if r_unique >= 2 and (x_unique <= 8 and y_unique <= 8):
        if sym_x and sym_y:
            return "apsk-like"
        else:
            return "ring-hybrid"

    if r_unique >= 3:
        return "clustered-multi-ring"

    if not sym_x or not sym_y:
        return "asymmetric-hybrid"

    return "fractured-non-euclidean"
