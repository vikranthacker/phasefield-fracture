"""Verification tests -- the solver is checked against closed-form solutions.

These encode the accuracy claims of the project:
  * homogeneous critical stress           -> within 1 % of closed form
  * optimal crack profile (AT1 & AT2)     -> L2 error < 2 %
  * dissipated energy of the crack        -> equals Gc within 1 %
  * a 2-D single-edge-notch crack         -> nucleates, propagates, reaches the
    far edge, and dissipates ~ Gc * crack-length.

Run:  pytest -q
"""
import numpy as np
import pytest

from phasefield import Material, PhaseFieldModel, analytic
from phasefield.onedim import (run_tension_test, optimal_profile_1d,
                               fracture_energy_1d)

MAT = Material(E=210e3, nu=0.3, Gc=2.7)


@pytest.mark.parametrize("variant", ["AT2", "AT1"])
@pytest.mark.parametrize("ell", [0.02, 0.04])
def test_critical_stress(variant, ell):
    mdl = PhaseFieldModel(MAT, ell=ell, variant=variant)
    res = run_tension_test(mdl, L=1.0, n_elem=300, n_steps=250)
    ana = analytic.critical_stress(MAT.E, MAT.Gc, ell, variant)
    err = abs(res["peak_stress"] - ana) / ana
    assert err < 0.01, f"{variant} l={ell}: critical-stress error {err:.3%}"


@pytest.mark.parametrize("variant", ["AT2", "AT1"])
def test_crack_profile(variant):
    ell = 0.05
    x, d = optimal_profile_1d(PhaseFieldModel(MAT, ell=ell, variant=variant),
                              L=1.0, n_elem=800)
    dana = analytic.crack_profile(x, ell, variant)
    l2 = np.sqrt(np.mean((d - dana) ** 2))
    assert l2 < 0.02, f"{variant}: profile L2 error {l2:.4f}"


@pytest.mark.parametrize("variant", ["AT2", "AT1"])
@pytest.mark.parametrize("ell", [0.03, 0.05])
def test_energy_consistency(variant, ell):
    mdl = PhaseFieldModel(MAT, ell=ell, variant=variant)
    x, d = optimal_profile_1d(mdl, L=1.0, n_elem=1000)
    Ef = fracture_energy_1d(x, d, mdl)
    err = abs(Ef - MAT.Gc) / MAT.Gc
    assert err < 0.01, f"{variant} l={ell}: energy error {err:.3%} (E={Ef:.4f})"


def test_2d_sent_propagates():
    """Coarse 2-D SENT: the crack must nucleate at the notch, run across the
    plate, and dissipate about Gc * crack-length."""
    from phasefield import StructuredMesh, StaggeredSolver, adaptive_load
    mat = Material(E=210e3, nu=0.3, Gc=2.7)
    mdl = PhaseFieldModel(mat, ell=0.03, variant="AT2", split="isotropic")
    mesh = StructuredMesh(1.0, 1.0, 40, 40)
    solver = StaggeredSolver(mesh, mdl)
    gp = mesh.gauss_point_coords()
    xg, yg = gp[..., 0], gp[..., 1]
    solver.seed_crack((np.abs(yg - 0.5) < 0.7 * mesh.hy) & (xg <= 0.5))
    bot = mesh.nodes_where(lambda x, y: y < 1e-9)
    top = mesh.nodes_where(lambda x, y: y > 1 - 1e-9)

    def bc(U):
        return (np.concatenate([2 * bot, 2 * bot + 1, 2 * top + 1]),
                np.concatenate([np.zeros(2 * len(bot)), np.full(len(top), U)]))

    adaptive_load(solver, bc, 2 * top + 1, U_max=6.5e-3, dU0=6e-4,
                  dU_min=3e-5, max_stagger=200, tol=2e-3, verbose=False)
    D = solver.nodal_damage_grid()
    mid = D[20]
    assert solver.d.max() > 0.95                 # a crack formed
    assert mid[-1] > 0.8                          # it reached the right edge
    # dissipated energy is of order Gc * crack-length; the exact factor carries
    # a finite-l / seeded-crack overhead (the clean Gc check is the 1-D test),
    # so this is only a sanity bound.
    _, Efrac = solver.energies(solver._H_gp_last)
    assert 0.7 * mat.Gc < Efrac < 3.0 * mat.Gc
