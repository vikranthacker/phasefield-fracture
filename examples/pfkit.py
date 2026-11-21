"""Small shared helpers for the example scripts."""
from __future__ import annotations

import numpy as np

import _pf_path  # noqa: F401
from phasefield import StructuredMesh, StaggeredSolver


def seed_horizontal_notch(mesh: StructuredMesh, solver: StaggeredSolver,
                          y0, x_from=0.0, x_to=0.5, width_cells=0.7):
    gp = mesh.gauss_point_coords()
    xg, yg = gp[..., 0], gp[..., 1]
    seed = (np.abs(yg - y0) < width_cells * mesh.hy) & (xg >= x_from) & (xg <= x_to)
    solver.seed_crack(seed)
    return seed


def tension_bc_factory(mesh: StructuredMesh):
    """Bottom edge clamped, top edge pulled in +y (ux free on top)."""
    bot = mesh.nodes_where(lambda x, y: y < 1e-9)
    top = mesh.nodes_where(lambda x, y: y > mesh.Ly - 1e-9)

    def bc_fn(U):
        fixed = np.concatenate([2 * bot, 2 * bot + 1, 2 * top + 1])
        values = np.concatenate([np.zeros(2 * len(bot)), np.full(len(top), U)])
        return fixed, values

    return bc_fn, bot, top
