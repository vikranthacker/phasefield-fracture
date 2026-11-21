"""Closed-form reference solutions used to verify the solver.

References
----------
* B. Bourdin, G.A. Francfort, J-J. Marigo, *The variational approach to
  fracture*, J. Elasticity 91 (2008).
* K. Pham, H. Amor, J-J. Marigo, C. Maurini, *Gradient damage models and their
  use to approximate brittle fracture*, Int. J. Damage Mech. 20 (2011).
* E. Tanne, T. Li, B. Bourdin, J-J. Marigo, C. Maurini, *Crack nucleation in
  variational phase-field models of brittle fracture*, JMPS 110 (2018).
"""
from __future__ import annotations

import numpy as np

SQRT3 = np.sqrt(3.0)


# --- homogeneous 1-D critical (peak) stress & strain ------------------
def critical_stress(E, Gc, ell, variant="AT2"):
    """Homogeneous uniaxial peak stress of the 1-D gradient-damage model."""
    if variant == "AT1":
        return np.sqrt(3.0 * E * Gc / (8.0 * ell))
    elif variant == "AT2":
        return (3.0 / 16.0) * SQRT3 * np.sqrt(E * Gc / ell)   # ~0.3248 sqrt(EGc/l)
    raise ValueError(variant)


def critical_strain(E, Gc, ell, variant="AT2"):
    if variant == "AT1":
        return np.sqrt(3.0 * Gc / (8.0 * ell * E))
    elif variant == "AT2":
        return np.sqrt(Gc / (3.0 * E * ell))
    raise ValueError(variant)


# --- optimal localized crack (damage) profile -------------------------
def crack_profile(x, ell, variant="AT2"):
    """Optimal 1-D damage profile centred at x = 0 (d(0) = 1)."""
    xi = np.abs(x) / ell
    if variant == "AT2":
        return np.exp(-xi)
    elif variant == "AT1":
        d = (1.0 - xi / 2.0) ** 2
        return np.where(np.abs(x) <= 2.0 * ell, d, 0.0)
    raise ValueError(variant)


def homogeneous_response(E, Gc, ell, eps, variant="AT2"):
    """Homogeneous stress for a strain array (no gradient term)."""
    a = Gc / ell
    Ee2 = E * eps ** 2
    if variant == "AT2":
        d = Ee2 / (Ee2 + a)
    else:  # AT1: elastic until threshold, then damage grows
        d = np.clip(1.0 - 3.0 * Gc / (8.0 * ell * np.maximum(Ee2, 1e-300)), 0.0, 1.0)
    return (1.0 - d) ** 2 * E * eps, d


# --- Griffith / LEFM estimate for single-edge-notched tension ---------
def sent_K_geometry_factor(a_over_W):
    """Mode-I geometry factor Y for single-edge cracked plate in tension."""
    s = a_over_W
    return (1.12 - 0.231 * s + 10.55 * s ** 2
            - 21.72 * s ** 3 + 30.39 * s ** 4)


def griffith_critical_stress(E, nu, Gc, a, W, plane_strain=True):
    """Remote stress at which a single edge crack of length ``a`` propagates.

    G = K_I^2 / E'  with  K_I = sigma sqrt(pi a) Y(a/W);  set G = Gc.
    """
    Ep = E / (1.0 - nu ** 2) if plane_strain else E
    Y = sent_K_geometry_factor(a / W)
    Kc = np.sqrt(Gc * Ep)
    return Kc / (Y * np.sqrt(np.pi * a))
