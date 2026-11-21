"""Example 2 -- Crack propagation in a heterogeneous (random) toughness field.

A single-edge notch propagates through a plate whose fracture toughness Gc(x,y)
is a spatially correlated log-normal random field.  The crack no longer runs
straight: it deflects toward weaker material.  Running two different random
seeds illustrates crack-path sensitivity to the microstructure.

Run:  python examples/ex02_heterogeneous.py --n 110
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


def run_one(n, ell, seed, cov=0.35, corr=0.06, U_max=7.5e-3, dU0=4e-4):
    mat = Material(E=210e3, nu=0.3, Gc=2.7)
    mdl = PhaseFieldModel(mat, ell=ell, variant="AT2", split="isotropic")
    mesh = StructuredMesh(1.0, 1.0, n, n)
    Gc = materials.gaussian_random_field(mesh, mat.Gc, cov=cov,
                                         corr_length=corr, seed=seed)
    solver = StaggeredSolver(mesh, mdl, Gc_elem=Gc)
    seed_horizontal_notch(mesh, solver, y0=0.5, x_to=0.4)
    bc_fn, bot, top = tension_bc_factory(mesh)
    adaptive_load(solver, bc_fn, 2 * top + 1, U_max=U_max, dU0=dU0,
                  dU_min=dU0 / 25, max_stagger=300, tol=1e-3)
    return mesh, solver, Gc


def run(n=110, ell=0.012):
    os.makedirs(RESULTS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    mesh, solver, Gc = run_one(n, ell, seed=1)
    toughness_map(mesh, Gc, ax=axes[0], title="toughness field $G_c(x,y)$  (seed 1)")
    plot_damage(mesh, solver.d, ax=axes[1], title="crack path (seed 1)")
    np.savez(os.path.join(RESULTS, "hetero_seed1.npz"), damage=solver.d, Gc=Gc, n=n)

    mesh2, solver2, Gc2 = run_one(n, ell, seed=7)
    plot_damage(mesh2, solver2.d, ax=axes[2], title="crack path (seed 7)")
    np.savez(os.path.join(RESULTS, "hetero_seed7.npz"), damage=solver2.d, Gc=Gc2, n=n)

    fig.suptitle("Crack-path sensitivity to heterogeneous toughness", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "heterogeneous.png"), dpi=130,
                bbox_inches="tight")
    print("saved:", os.path.join(RESULTS, "heterogeneous.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=110)
    p.add_argument("--ell", type=float, default=0.012)
    args = p.parse_args()
    run(n=args.n, ell=args.ell)
