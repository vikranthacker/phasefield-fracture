"""Staggered (alternate-minimization) phase-field fracture solver.

Each load step alternates between
  (1) a linear elasticity solve for the displacement at frozen damage, and
  (2) a damage sub-problem at frozen displacement (history field H for
      irreversibility, plus a hard d >= d_old bound),
iterating until the damage field converges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.sparse.linalg import spsolve

from .mesh import StructuredMesh
from .model import PhaseFieldModel
from .fem import Assembler, apply_dirichlet, solve_bounded


@dataclass
class StepResult:
    disp: float
    reaction: float
    elastic_energy: float
    fracture_energy: float
    max_damage: float
    stagger_iters: int


class StaggeredSolver:
    def __init__(self, mesh: StructuredMesh, model: PhaseFieldModel,
                 Gc_elem: Optional[np.ndarray] = None):
        self.mesh = mesh
        self.model = model
        self.asm = Assembler(mesh, model)
        if Gc_elem is None:
            Gc_elem = np.full(mesh.n_elem, model.material.Gc)
        self.Gc_elem = np.asarray(Gc_elem, float)

        self.d = np.zeros(mesh.n_nodes)           # nodal damage
        self.d_committed = np.zeros(mesh.n_nodes)  # irreversibility lower bound
        self.u = np.zeros(2 * mesh.n_nodes)
        self.H_committed = np.zeros((mesh.n_elem, 4))
        self.H_seed = np.zeros((mesh.n_elem, 4))

    # ------------------------------------------------------------------
    def seed_crack(self, gp_mask: np.ndarray, factor: float = 1.0e3):
        """Seed an initial (phase-field) crack at the flagged Gauss points.

        A large history value forces d -> 1 there; because it is stored as
        history it persists and enforces the pre-crack throughout loading.
        """
        Hval = factor * self.model.material.Gc / self.model.ell
        self.H_seed[gp_mask] = Hval
        self.H_committed = np.maximum(self.H_committed, self.H_seed)
        # initialise damage consistent with the seed
        self._update_damage(self.H_committed, lo=np.zeros(self.mesh.n_nodes))
        self.d_committed = self.d.copy()

    # ------------------------------------------------------------------
    def _update_damage(self, H_gp, lo):
        A, b = self.asm.assemble_damage(H_gp, self.Gc_elem)
        hi = np.ones(self.mesh.n_nodes)
        self.d = solve_bounded(A, b, lo, hi)

    # ------------------------------------------------------------------
    def solve_step(self, fixed, values, reaction_dofs,
                   max_stagger=200, tol=1e-3, verbose=False):
        mesh = self.mesh
        lo = self.d_committed.copy()
        K_last = None
        for it in range(max_stagger):
            # (1) elasticity at frozen damage
            K = self.asm.assemble_elasticity(self.d, u_prev=self.u)
            K_last = K
            Kbc, fbc = apply_dirichlet(K, np.zeros(2 * mesh.n_nodes),
                                       fixed, values)
            self.u = spsolve(Kbc, fbc)

            # (2) history + damage at frozen displacement
            eps = self.asm.strain_at_gp(self.u)
            psi = self.model.psi_plus(eps)
            H_gp = np.maximum(self.H_committed, psi)
            H_gp = np.maximum(H_gp, self.H_seed)

            d_old = self.d.copy()
            self._update_damage(H_gp, lo)
            delta = np.max(np.abs(self.d - d_old))
            if verbose:
                print(f"    stagger {it:3d}  d-change={delta:.2e}")
            if delta < tol:
                self._H_gp_last = H_gp
                break
        else:
            self._H_gp_last = H_gp

        # commit state
        self.H_committed = H_gp
        self.d_committed = self.d.copy()

        reaction = float(K_last.dot(self.u)[reaction_dofs].sum())
        Eel, Efrac = self.energies(self._H_gp_last)
        return StepResult(disp=float(np.max(np.abs(values))) if len(values) else 0.0,
                          reaction=reaction,
                          elastic_energy=Eel,
                          fracture_energy=Efrac,
                          max_damage=float(self.d.max()),
                          stagger_iters=it + 1)

    # ------------------------------------------------------------------
    def energies(self, H_gp):
        mesh, mdl = self.mesh, self.model
        # elastic stored energy  int g(d) psi+  (+ psi- not tracked separately here)
        d_gp = self.asm.gp_damage(self.d)
        eps = self.asm.strain_at_gp(self.u)
        psi = mdl.psi_plus(eps)
        w = mesh.gp_w                                   # (4,)
        Eel = float(np.einsum('eg,eg,g->', mdl.g(d_gp), psi, w))

        # fracture energy  (Gc/cw) int [ w(d)/ell + ell |grad d|^2 ]
        d_e = self.d[mesh.elements]                     # (n_elem,4)
        grad_term = np.einsum('ei,ij,ej->e', d_e, self.asm.Kgrad_sum, d_e)
        if mdl.variant == "AT2":
            local = np.einsum('eg,g->e', d_gp ** 2, w) / mdl.ell
            Efrac = float(np.sum(0.5 * self.Gc_elem * (local + mdl.ell * grad_term)))
        else:  # AT1
            local = np.einsum('eg,g->e', d_gp, w) / mdl.ell
            Efrac = float(np.sum((3.0 / 8.0) * self.Gc_elem * (local + mdl.ell * grad_term)))
        return Eel, Efrac

    # ------------------------------------------------------------------
    def nodal_damage_grid(self):
        return self.d.reshape(self.mesh.ny + 1, self.mesh.nx + 1)


def adaptive_load(solver: "StaggeredSolver", bc_fn, reaction_dofs,
                  U_max, dU0, dU_min=None, dU_max=None,
                  max_stagger=300, tol=1e-3, verbose=True,
                  lo_iter=8, hi_iter=40):
    """Displacement-controlled loading with forward step-size adaptation.

    ``bc_fn(U)`` returns ``(fixed_dofs, values)`` for imposed displacement U.
    The increment grows while steps converge quickly (elastic regime) and
    shrinks when a step needs many staggered iterations (crack propagation),
    which keeps the total solve count low without a monolithic scheme.
    """
    dU = dU0
    dU_min = dU_min if dU_min is not None else dU0 / 40.0
    dU_max = dU_max if dU_max is not None else dU0 * 3.0
    U = 0.0
    history = []
    step = 0
    while U < U_max - 1e-12:
        dU = min(dU, U_max - U)
        U_try = U + dU
        fixed, values = bc_fn(U_try)
        res = solver.solve_step(fixed, values, reaction_dofs,
                                max_stagger=max_stagger, tol=tol)
        res.disp = U_try
        history.append(res)
        U = U_try
        step += 1
        if verbose:
            print(f"step {step:3d}  U={U:.5f}  F={res.reaction:10.2f}  "
                  f"dmax={res.max_damage:.3f}  Efrac={res.fracture_energy:.4f}  "
                  f"iters={res.stagger_iters}  dU={dU:.2e}", flush=True)
        # forward adaptation
        if res.stagger_iters > hi_iter:
            dU = max(dU * 0.5, dU_min)
        elif res.stagger_iters < lo_iter:
            dU = min(dU * 1.3, dU_max)
    return history
