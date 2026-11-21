"""Example 4 -- Crack deflection by tough circular inclusions.

A notch propagates through a matrix seeded with stiff, *tough* circular
inclusions (high Gc).  The crack is repelled by the tough particles and weaves
between them -- a canonical crack/microstructure-interaction problem.

Run:  python examples/ex04_inclusions.py --n 120
"""
from __future__ import annotations

import argparse
import os
import numpy as np

import _pf_path  # noqa: F401
from phasefield import (StructuredMesh, Material, PhaseFieldModel,
                        StaggeredSolver, adaptive_load, materials)
from phasefield.postprocess import plot_damage, toughness_map
from pfkit import seed_horizontal_notch, tension_bc_factory
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run(n=120, ell=0.012, U_max=8.0e-3, dU0=4e-4):
    mat = Material(E=210e3, nu=0.3, Gc=2.7)
    mdl = PhaseFieldModel(mat, ell=ell, variant="AT2", split="isotropic")
    mesh = StructuredMesh(1.0, 1.0, n, n)
    inclusions = [(0.62, 0.56, 0.08), (0.80, 0.44, 0.07),
                  (0.46, 0.60, 0.055), (0.72, 0.68, 0.05)]
    Gc = materials.circular_inclusions(mesh, Gc_matrix=mat.Gc,
                                       inclusions=inclusions,
                                       Gc_inclusion=5.0 * mat.Gc)
    solver = StaggeredSolver(mesh, mdl, Gc_elem=Gc)
    seed_horizontal_notch(mesh, solver, y0=0.5, x_to=0.35)
    bc_fn, bot, top = tension_bc_factory(mesh)
    adaptive_load(solver, bc_fn, 2 * top + 1, U_max=U_max, dU0=dU0,
                  dU_min=dU0 / 25, max_stagger=300, tol=1e-3)

    os.makedirs(RESULTS, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    toughness_map(mesh, Gc, ax=ax[0], title="tough inclusions ($5\\,G_c$)")
    plot_damage(mesh, solver.d, ax=ax[1], title="crack deflected by inclusions")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "inclusions.png"), dpi=130)
    np.savez(os.path.join(RESULTS, "inclusions.npz"), damage=solver.d, Gc=Gc, n=n)
    print("saved:", os.path.join(RESULTS, "inclusions.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--ell", type=float, default=0.012)
    args = p.parse_args()
    run(n=args.n, ell=args.ell)
