"""Example 3 -- Crack branching / path selection among competing weak channels.

Two symmetric weak channels (reduced Gc) diverge ahead of the notch tip, so the
crack faces a genuine branch point.  In the *quasi-static* energy-minimizing
model the crack selects ONE channel: as soon as one branch starts to open it
relaxes the crack-driving force for the other (symmetry breaking), so the second
branch remains incipient.  This is the physically correct quasi-static picture,
and it demonstrates how heterogeneity controls the branch that is taken.

True simultaneous (symmetric) branching is an *inertial* phenomenon -- it occurs
when a running crack exceeds a critical fraction of the Rayleigh wave speed --
and needs a dynamic solver (mass term + explicit time stepping), listed as
future work in the README.

Run:  python examples/ex03_branching.py --n 120
"""
from __future__ import annotations

import argparse
import os
import numpy as np

import _pf_path  # noqa: F401
from phasefield import (StructuredMesh, Material, PhaseFieldModel,
                        StaggeredSolver, adaptive_load)
from phasefield.postprocess import plot_damage, toughness_map
from pfkit import seed_horizontal_notch, tension_bc_factory
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def weak_channels(mesh, Gc0, tip=(0.5, 0.5), angles=(25.0, -25.0),
                  weak=0.35, half_width=0.03):
    """Reduce Gc along two diverging rays from the notch tip."""
    c = mesh.element_centroids()
    Gc = np.full(mesh.n_elem, Gc0)
    tx, ty = tip
    for a in angles:
        th = np.deg2rad(a)
        dx, dy = np.cos(th), np.sin(th)
        # projection of (c-tip) onto ray direction
        px = c[:, 0] - tx
        py = c[:, 1] - ty
        s = px * dx + py * dy                      # along-ray distance
        perp = np.abs(-px * dy + py * dx)          # perpendicular distance
        band = (s > 0) & (perp < half_width)
        Gc[band] = np.minimum(Gc[band], weak * Gc0)
    return Gc


def run(n=120, ell=0.012, U_max=9.0e-3, dU0=4e-4):
    mat = Material(E=210e3, nu=0.3, Gc=2.7)
    mdl = PhaseFieldModel(mat, ell=ell, variant="AT2", split="amor")
    mesh = StructuredMesh(1.0, 1.0, n, n)
    Gc = weak_channels(mesh, mat.Gc)
    solver = StaggeredSolver(mesh, mdl, Gc_elem=Gc)
    seed_horizontal_notch(mesh, solver, y0=0.5, x_to=0.5)
    bc_fn, bot, top = tension_bc_factory(mesh)
    adaptive_load(solver, bc_fn, 2 * top + 1, U_max=U_max, dU0=dU0,
                  dU_min=dU0 / 25, max_stagger=300, tol=1e-3)

    os.makedirs(RESULTS, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    toughness_map(mesh, Gc, ax=ax[0], title="competing weak channels ($0.35\\,G_c$)")
    plot_damage(mesh, solver.d, ax=ax[1],
                title="crack selects one branch (quasi-static)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "branching.png"), dpi=130)
    np.savez(os.path.join(RESULTS, "branching.npz"), damage=solver.d, Gc=Gc, n=n)
    print("saved:", os.path.join(RESULTS, "branching.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--ell", type=float, default=0.012)
    args = p.parse_args()
    run(n=args.n, ell=args.ell)
