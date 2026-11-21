"""Example 1 -- Single-Edge-Notched Tension (SENT).

The classic Miehe (2010) brittle-fracture benchmark: a square plate with a
horizontal pre-crack from the left edge to the centre, pulled in tension at the
top edge.  The crack runs straight across to the right edge.  We record the
force-displacement response (peak load + brittle snap) and the final crack.

Run:  python examples/ex01_sent.py --n 100 --steps-ell 0.01
"""
from __future__ import annotations

import argparse
import os
import numpy as np

import _pf_path  # noqa: F401  (adds repo root to sys.path)
from phasefield import (StructuredMesh, Material, PhaseFieldModel,
                        StaggeredSolver, adaptive_load)
from phasefield.postprocess import plot_damage, plot_force_displacement
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def seed_notch(mesh, solver, y0=0.5, x_tip=0.5):
    gp = mesh.gauss_point_coords()
    xg, yg = gp[..., 0], gp[..., 1]
    seed = (np.abs(yg - y0) < 0.7 * mesh.hy) & (xg <= x_tip)
    solver.seed_crack(seed)
    return seed


def run(n=100, ell=0.01, U_max=7.0e-3, dU0=4e-4, variant="AT2", tag="sent"):
    mat = Material(E=210e3, nu=0.3, Gc=2.7)
    mdl = PhaseFieldModel(mat, ell=ell, variant=variant, split="isotropic")
    mesh = StructuredMesh(1.0, 1.0, n, n)
    solver = StaggeredSolver(mesh, mdl)
    seed_notch(mesh, solver)

    bot = mesh.nodes_where(lambda x, y: y < 1e-9)
    top = mesh.nodes_where(lambda x, y: y > 1.0 - 1e-9)

    def bc_fn(U):
        fixed = np.concatenate([2 * bot, 2 * bot + 1, 2 * top + 1])
        values = np.concatenate([np.zeros(2 * len(bot)), np.full(len(top), U)])
        return fixed, values

    hist = adaptive_load(solver, bc_fn, reaction_dofs=2 * top + 1,
                         U_max=U_max, dU0=dU0, dU_min=dU0 / 25,
                         max_stagger=300, tol=1e-3)

    disp = np.array([h.disp for h in hist])
    force = np.array([h.reaction for h in hist])
    os.makedirs(RESULTS, exist_ok=True)
    np.savez(os.path.join(RESULTS, f"{tag}_data.npz"),
             disp=disp, force=force, damage=solver.d, n=n, ell=ell)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    plot_force_displacement(disp, force, ax=ax[0], color="crimson", lw=2)
    ax[0].set_title(f"SENT force-displacement ({variant}, $\\ell$={ell})")
    ax[0].annotate(f"peak = {force.max():.1f} N",
                   xy=(disp[force.argmax()], force.max()),
                   xytext=(0.4, 0.8), textcoords="axes fraction",
                   arrowprops=dict(arrowstyle="->"))
    plot_damage(mesh, solver.d, ax=ax[1], title="final crack")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"{tag}.png"), dpi=130)
    print("peak force =", round(float(force.max()), 2), "N")
    print("saved:", os.path.join(RESULTS, f"{tag}.png"))
    return hist, solver


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--ell", type=float, default=0.01)
    p.add_argument("--variant", default="AT2")
    args = p.parse_args()
    run(n=args.n, ell=args.ell, variant=args.variant)
