"""Minimal 1-D phase-field bar solver (linear elements) for verification.

Displacement-controlled uniaxial tension of a bar of unit cross-section.  Used
to check the homogeneous critical stress and the localized crack profile
against the closed forms in :mod:`phasefield.analytic`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .model import PhaseFieldModel
from .fem import solve_bounded, solve_bounded_psor, apply_dirichlet


def fracture_energy_1d(x, d, model: PhaseFieldModel):
    """Dissipated fracture energy (Gc/cw) int [w(d)/l + l d'^2] of a 1-D field.

    For the optimal crack profile this equals Gc exactly (Gamma-convergence),
    so it is a clean consistency check on the regularization constants.
    """
    Gc, ell = model.material.Gc, model.ell
    h = x[1] - x[0]
    dmid = 0.5 * (d[:-1] + d[1:])
    dprime = (d[1:] - d[:-1]) / h
    if model.variant == "AT2":
        w = dmid ** 2
        pref = Gc / 2.0
    else:
        w = dmid
        pref = 3.0 * Gc / 8.0
    integrand = w / ell + ell * dprime ** 2
    return float(pref * np.sum(integrand * h))


def optimal_profile_1d(model: PhaseFieldModel, L=1.0, n_elem=800):
    """Minimizer of the fracture functional with d=1 enforced at the centre.

    Returns the optimal 1-D damage profile, i.e. the Gamma-convergence crack
    profile:  exp(-|x|/l) for AT2,  (1-|x|/2l)^2 for AT1.  Validates the
    gradient/mass discretization in isolation from the elastic driving force.
    """
    mat = model.material
    h = L / n_elem
    x = np.linspace(-L / 2, L / 2, n_elem + 1)
    Gc_e = np.full(n_elem, mat.Gc)

    Kd_base = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
    M_base = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    Nint = h / 2.0

    grad_c, mass_c, source_c = model.damage_coeffs(Gc_e)
    # H = 0 (no elastic drive); minimize pure fracture energy
    A_blocks = (grad_c[:, None, None] * Kd_base + mass_c[:, None, None] * M_base)
    A = _assemble_1d(n_elem, h, A_blocks)
    b = np.zeros(n_elem + 1)
    bval = -source_c * Nint
    np.add.at(b, np.arange(n_elem), bval)
    np.add.at(b, np.arange(1, n_elem + 1), bval)

    # hard Dirichlet d=1 at the centre node (removes the pure-Laplacian
    # nullspace for AT1), then a box-constrained solve for d in [0, 1].
    cnode = n_elem // 2
    A, b = apply_dirichlet(A, b, [cnode], [1.0])
    lo = np.zeros(n_elem + 1)
    lo[cnode] = 1.0
    hi = np.ones(n_elem + 1)
    # projected SOR grows the damaged band outward from the pinned centre and
    # (unlike a pointwise active-set method) reaches the true obstacle contact.
    d = solve_bounded_psor(A, b, lo, hi)
    return x, d


def _assemble_1d(n_elem, h, vals):
    """Tridiagonal assembly from per-element 2x2 blocks ``vals`` (n_elem,2,2)."""
    N = n_elem + 1
    rows = np.zeros((n_elem, 2, 2), int)
    cols = np.zeros((n_elem, 2, 2), int)
    idx = np.arange(n_elem)
    conn = np.column_stack([idx, idx + 1])
    for a in range(2):
        for b in range(2):
            rows[:, a, b] = conn[:, a]
            cols[:, a, b] = conn[:, b]
    return sp.coo_matrix((vals.reshape(-1), (rows.reshape(-1), cols.reshape(-1))),
                         shape=(N, N)).tocsr()


def run_tension_test(model: PhaseFieldModel, L=1.0, n_elem=400,
                     eps_max=None, n_steps=400, imperfection=1e-3):
    """Ramp an imposed end displacement and record the homogeneous stress.

    Returns dict with strain, stress, peak stress, node coords and final damage.
    """
    mat = model.material
    E = mat.E
    h = L / n_elem
    x = np.linspace(0.0, L, n_elem + 1)

    # slightly weaken the central element to localize the crack there
    Gc_e = np.full(n_elem, mat.Gc)
    Gc_e[n_elem // 2] *= (1.0 - imperfection)

    # element stiffness/mass bases (linear, 1 element)
    Kd_base = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h          # d' v'
    M_base = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])     # d v
    Nint = h / 2.0                                             # int N_i

    d = np.zeros(n_elem + 1)
    d_committed = np.zeros(n_elem + 1)
    H_committed = np.zeros(n_elem)

    if eps_max is None:
        from .analytic import critical_strain
        eps_max = 2.5 * critical_strain(E, mat.Gc, model.ell, model.variant)

    disps = np.linspace(0.0, eps_max * L, n_steps + 1)[1:]
    strain_hist, stress_hist = [], []

    for U in disps:
        for _ in range(300):
            # ---- elasticity: (1-d)^2 E u' = F  -> u from FEM
            d_mid = 0.5 * (d[:-1] + d[1:])
            g = model.g(d_mid)
            ke = (g * E / h)[:, None, None] * np.array([[1.0, -1.0], [-1.0, 1.0]])
            Kg = _assemble_1d(n_elem, h, ke)
            # BC u(0)=0, u(L)=U
            f = np.zeros(n_elem + 1)
            free = np.arange(1, n_elem)
            u = np.zeros(n_elem + 1)
            u[-1] = U
            f = -Kg.dot(u)
            Kff = Kg[free][:, free]
            u[free] = spsolve(Kff.tocsc(), f[free])

            # ---- history & damage
            du = (u[1:] - u[:-1]) / h
            psi = 0.5 * E * du ** 2
            H = np.maximum(H_committed, psi)

            grad_c, mass_c, source_c = model.damage_coeffs(Gc_e)
            A_blocks = (grad_c[:, None, None] * Kd_base
                        + (mass_c + 2.0 * H)[:, None, None] * M_base)
            A = _assemble_1d(n_elem, h, A_blocks)
            b = np.zeros(n_elem + 1)
            bval = (2.0 * H - source_c) * Nint
            np.add.at(b, np.arange(n_elem), bval)
            np.add.at(b, np.arange(1, n_elem + 1), bval)

            d_old = d.copy()
            d = solve_bounded(A, b, d_committed, np.ones(n_elem + 1))
            if np.max(np.abs(d - d_old)) < 1e-4:
                break

        H_committed = H
        d_committed = d.copy()
        # homogeneous stress = reaction force / area (unit area)
        stress = float(Kg.dot(u)[-1])
        strain_hist.append(U / L)
        stress_hist.append(stress)

    strain_hist = np.array(strain_hist)
    stress_hist = np.array(stress_hist)
    return dict(strain=strain_hist, stress=stress_hist,
                peak_stress=float(stress_hist.max()),
                x=x, damage=d, Gc_e=Gc_e)
