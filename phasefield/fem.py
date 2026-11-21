"""Vectorized finite-element assembly for the staggered phase-field solver.

Because the structured mesh has identical rectangular cells, all element
operators are pre-computed once from the mesh's Gauss-point data and then
scaled per element by the degradation g(d) (elasticity) or the local history
field H and material coefficients (damage).  Global assembly is a single COO
scatter.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .mesh import StructuredMesh
from .model import PhaseFieldModel


class Assembler:
    def __init__(self, mesh: StructuredMesh, model: PhaseFieldModel):
        self.mesh = mesh
        self.model = model
        self._build_bases()

    # ------------------------------------------------------------------
    def _build_bases(self):
        m, mdl = self.mesh, self.model
        D = mdl.material.D_planestrain()
        lam, mu = mdl.material.lame
        K = mdl.material.bulk_2d

        # Voigt constitutive splits (plane strain, 2-D trace)
        C_vol_unit = np.array([[1.0, 1.0, 0.0],
                               [1.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0]])          # scaled by K
        C_dev = np.array([[mu, -mu, 0.0],
                          [-mu, mu, 0.0],
                          [0.0, 0.0, mu]])

        self.Ke_iso = np.zeros((4, 8, 8))    # full (isotropic) per gp
        self.Ke_vol = np.zeros((4, 8, 8))    # K * B^T C_vol B per gp
        self.Ke_dev = np.zeros((4, 8, 8))    # B^T C_dev B per gp
        for g in range(4):
            B = m.gp_B[g]
            w = m.gp_w[g]
            self.Ke_iso[g] = w * (B.T @ D @ B)
            self.Ke_vol[g] = w * (B.T @ (K * C_vol_unit) @ B)
            self.Ke_dev[g] = w * (B.T @ C_dev @ B)

        # damage bases
        self.Kgrad_sum = m.gp_Kgrad.sum(axis=0)   # (4,4)
        self.Mass_gp = m.gp_Mass                  # (4,4,4)
        self.Nw_gp = m.gp_w[:, None] * m.gp_N     # (4,4)

    # ------------------------------------------------------------------
    def strain_at_gp(self, u):
        """Return Voigt strains at every (elem, gp): shape (n_elem, 4, 3)."""
        u_e = u[self.mesh.edof]                              # (n_elem, 8)
        return np.einsum('gki,ei->egk', self.mesh.gp_B, u_e)

    def gp_damage(self, d):
        """Interpolate nodal damage to Gauss points: (n_elem, 4)."""
        d_e = d[self.mesh.elements]                          # (n_elem, 4)
        return d_e @ self.mesh.gp_N.T                        # (n_elem, 4)

    # ------------------------------------------------------------------
    def assemble_elasticity(self, d, u_prev=None):
        """Degraded stiffness matrix K(d).

        ``u_prev`` supplies the strain state used to decide tension/compression
        for the Amor split (lagged one staggered iteration).  Ignored for the
        isotropic split.
        """
        m, mdl = self.mesh, self.model
        g_gp = mdl.g(self.gp_damage(d))                      # (n_elem, 4)

        if mdl.split == "isotropic":
            Ke = np.einsum('eg,gij->eij', g_gp, self.Ke_iso)
        elif mdl.split == "amor":
            if u_prev is None:
                tr_pos = np.ones((m.n_elem, 4), dtype=bool)
            else:
                eps = self.strain_at_gp(u_prev)
                tr = eps[..., 0] + eps[..., 1]
                tr_pos = tr >= 0.0
            vol_fac = np.where(tr_pos, g_gp, 1.0)            # degrade vol only in tension
            Ke = (np.einsum('eg,gij->eij', vol_fac, self.Ke_vol)
                  + np.einsum('eg,gij->eij', g_gp, self.Ke_dev))
        else:
            raise ValueError(mdl.split)

        data = Ke.reshape(-1)
        Kmat = sp.coo_matrix((data, (m.vec_rows, m.vec_cols)),
                             shape=(2 * m.n_nodes, 2 * m.n_nodes)).tocsr()
        return Kmat

    # ------------------------------------------------------------------
    def assemble_damage(self, H_gp, Gc_elem):
        """Assemble the damage system A d = b.

        ``H_gp``    : history field at Gauss points (n_elem, 4).
        ``Gc_elem`` : per-element fracture toughness (n_elem,) or scalar.
        Returns (A_csr, b).
        """
        m, mdl = self.mesh, self.model
        Gc = np.broadcast_to(np.asarray(Gc_elem, float), (m.n_elem,))
        grad, mass, source = mdl.damage_coeffs(Gc)           # per element arrays
        grad = np.broadcast_to(grad, (m.n_elem,))
        mass = np.broadcast_to(mass, (m.n_elem,))
        source = np.broadcast_to(source, (m.n_elem,))

        # A_elem = grad * Kgrad_sum + sum_g (mass + 2H_g) Mass_g
        A = grad[:, None, None] * self.Kgrad_sum[None]       # (n_elem,4,4)
        coeff = mass[:, None] + 2.0 * H_gp                   # (n_elem,4) per gp
        A = A + np.einsum('eg,gij->eij', coeff, self.Mass_gp)

        # b_elem = sum_g (2H_g - source) Nw_g
        bcoeff = 2.0 * H_gp - source[:, None]                # (n_elem,4)
        b_elem = bcoeff @ self.Nw_gp                         # (n_elem,4)

        A_mat = sp.coo_matrix((A.reshape(-1), (m.sca_rows, m.sca_cols)),
                              shape=(m.n_nodes, m.n_nodes)).tocsr()
        b = np.zeros(m.n_nodes)
        np.add.at(b, m.elements.reshape(-1), b_elem.reshape(-1))
        return A_mat, b


# ----------------------------------------------------------------------
def solve_bounded_psor(A, b, lo, hi, maxit=200000, omega=1.9, tol=1e-11,
                       d0=None, check_every=50):
    """Vectorized red-black projected SOR for the box-constrained QP.

    Globally and monotonically convergent for an SPD M-matrix (the regularized
    damage operator), and immune to the stalling a pointwise active-set method
    suffers when the damaged region must grow connectedly from a poor initial
    guess (e.g. the AT1 obstacle profile).  Red-black ordering makes each sweep
    a pair of sparse mat-vecs, so it is fast for the 1-D (tridiagonal) solves
    used in verification.
    """
    A = A.tocsr()
    diag = A.diagonal()
    n = len(b)
    lo = np.broadcast_to(np.asarray(lo, float), (n,)).copy()
    hi = np.broadcast_to(np.asarray(hi, float), (n,)).copy()
    d = lo.copy() if d0 is None else np.clip(d0, lo, hi)

    idx = np.arange(n)
    red = idx % 2 == 0
    black = ~red
    A_red = A[red]
    A_black = A[black]
    dr, db = diag[red], diag[black]

    for it in range(maxit):
        # update red using current d, then black using updated d (Gauss-Seidel)
        r = b[red] - A_red.dot(d)
        d[red] = np.clip(d[red] + omega * r / dr, lo[red], hi[red])
        r = b[black] - A_black.dot(d)
        d[black] = np.clip(d[black] + omega * r / db, lo[black], hi[black])
        if it % check_every == 0:
            res = b - A.dot(d)
            # KKT residual: zero where interior, one-sided at the bounds
            res = np.where((d <= lo + 1e-12) & (res < 0), 0.0, res)
            res = np.where((d >= hi - 1e-12) & (res > 0), 0.0, res)
            if np.max(np.abs(res)) < tol * max(1.0, np.max(np.abs(b))):
                break
    return d


def apply_dirichlet(K, f, fixed, values):
    """Symmetric Dirichlet elimination. Returns (K_bc, f_bc)."""
    n = K.shape[0]
    u_bc = np.zeros(n)
    u_bc[fixed] = values
    f = f - K.dot(u_bc)
    mask = np.ones(n)
    mask[fixed] = 0.0
    Dm = sp.diags(mask)
    K = Dm @ K @ Dm
    K = K + sp.diags(1.0 - mask)
    f[fixed] = values
    return K.tocsr(), f


def solve_bounded(A, b, lo, hi, maxit=50):
    """Primal-dual active-set solver for the box-constrained QP

        min 1/2 d^T A d - b^T d   s.t.   lo <= d <= hi,   A SPD.

    This is the standard semismooth-Newton active-set method; it converges in a
    handful of iterations and, unlike a naive projected solve, finds the true
    constrained minimizer (correct obstacle contact / damage irreversibility).
    """
    A = A.tocsr()
    lo = np.broadcast_to(np.asarray(lo, float), (len(b),)).copy()
    hi = np.broadcast_to(np.asarray(hi, float), (len(b),)).copy()
    n = len(b)
    all_idx = np.arange(n)
    c = A.diagonal().mean()
    if not np.isfinite(c) or c <= 0:
        c = 1.0

    def objective(dv):
        return 0.5 * float(dv @ (A.dot(dv))) - float(b @ dv)

    d = np.clip(spsolve(A.tocsc(), b), lo, hi)
    lam = A.dot(d) - b
    best_d = d.copy()
    best_obj = objective(d)
    seen = set()
    for _ in range(maxit):
        at_lo = (lam + c * (lo - d)) > 0
        at_hi = (lam + c * (hi - d)) < 0
        at_hi &= ~at_lo
        inact = ~(at_lo | at_hi)

        d = d.copy()
        d[at_lo] = lo[at_lo]
        d[at_hi] = hi[at_hi]
        if inact.any():
            idx = all_idx[inact]
            fixed = all_idx[~inact]
            rhs = b[idx]
            if fixed.size:
                rhs = rhs - A[idx][:, fixed].dot(d[fixed])
            d[idx] = spsolve(A[idx][:, idx].tocsc(), rhs)
        d = np.clip(d, lo, hi)
        lam = A.dot(d) - b

        obj = objective(d)
        if obj < best_obj:
            best_obj = obj
            best_d = d.copy()

        key = (at_lo.tobytes(), at_hi.tobytes())
        if key in seen:          # active set converged or started cycling
            break
        seen.add(key)
    return best_d
