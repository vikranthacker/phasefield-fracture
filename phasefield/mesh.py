"""Structured bilinear (Q1) quadrilateral mesh for 2-D plane-strain problems.

Every cell is an identical axis-aligned rectangle, so the reference element
operators (B-matrices, shape values, gradient/mass matrices) are evaluated
*once* at the four 2x2 Gauss points and reused for every element.  Global
assembly then reduces to scaling these base operators by per-element material
factors and scattering them with pre-computed COO index arrays -- which is what
makes a pure NumPy/SciPy phase-field solver fast enough to be useful.
"""
from __future__ import annotations

import numpy as np

# 2x2 Gauss quadrature on the reference square [-1, 1]^2
_G = 1.0 / np.sqrt(3.0)
_GP = np.array([(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)])
_GW = np.array([1.0, 1.0, 1.0, 1.0])


def _shape(xi, eta):
    """Bilinear shape functions and their reference-coordinate derivatives.

    Node ordering (counter-clockwise):  0=(-1,-1) 1=(+1,-1) 2=(+1,+1) 3=(-1,+1)
    """
    N = 0.25 * np.array([(1 - xi) * (1 - eta),
                         (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta),
                         (1 - xi) * (1 + eta)])
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dN_dxi, dN_deta


class StructuredMesh:
    """A ``nx`` by ``ny`` grid of Q1 cells over ``[0, Lx] x [0, Ly]``."""

    def __init__(self, Lx: float, Ly: float, nx: int, ny: int):
        self.Lx, self.Ly, self.nx, self.ny = Lx, Ly, nx, ny
        self.hx, self.hy = Lx / nx, Ly / ny
        self.n_nodes = (nx + 1) * (ny + 1)
        self.n_elem = nx * ny
        self.h = min(self.hx, self.hy)

        # Node coordinates, indexed node = j*(nx+1) + i
        ii, jj = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
        self.coords = np.column_stack([ii.ravel() * self.hx,
                                       jj.ravel() * self.hy])

        # Element -> node connectivity (n_elem, 4), CCW
        ei, ej = np.meshgrid(np.arange(nx), np.arange(ny))
        ei, ej = ei.ravel(), ej.ravel()
        n0 = ej * (nx + 1) + ei
        self.elements = np.column_stack([n0, n0 + 1,
                                         n0 + 1 + (nx + 1), n0 + (nx + 1)])

        self._build_reference_operators()
        self._build_index_maps()

    # ------------------------------------------------------------------
    def _build_reference_operators(self):
        """Pre-compute per-Gauss-point element operators (identical cells)."""
        hx, hy = self.hx, self.hy
        detJ = 0.25 * hx * hy  # constant for an axis-aligned rectangle

        self.gp_N = np.zeros((4, 4))          # shape values  (gp, node)
        self.gp_B = np.zeros((4, 3, 8))       # strain-disp   (gp, 3, 8)
        self.gp_dN = np.zeros((4, 2, 4))      # phys gradients (gp, dim, node)
        self.gp_w = np.zeros(4)               # detJ * weight
        for g, (xi, eta) in enumerate(_GP):
            N, dN_dxi, dN_deta = _shape(xi, eta)
            dN_dx = dN_dxi / (0.5 * hx)
            dN_dy = dN_deta / (0.5 * hy)
            B = np.zeros((3, 8))
            B[0, 0::2] = dN_dx
            B[1, 1::2] = dN_dy
            B[2, 0::2] = dN_dy
            B[2, 1::2] = dN_dx
            self.gp_N[g] = N
            self.gp_B[g] = B
            self.gp_dN[g, 0] = dN_dx
            self.gp_dN[g, 1] = dN_dy
            self.gp_w[g] = detJ * _GW[g]

        # Base scalar operators (sum over gp handled by caller when H varies)
        # Gradient (stiffness) and mass matrices per gp for the damage field.
        self.gp_Kgrad = np.zeros((4, 4, 4))
        self.gp_Mass = np.zeros((4, 4, 4))
        for g in range(4):
            dN = self.gp_dN[g]
            N = self.gp_N[g]
            w = self.gp_w[g]
            self.gp_Kgrad[g] = w * (np.outer(dN[0], dN[0]) + np.outer(dN[1], dN[1]))
            self.gp_Mass[g] = w * np.outer(N, N)

        self.detJ = detJ

    # ------------------------------------------------------------------
    def _build_index_maps(self):
        """COO scatter indices for vector (2-dof) and scalar (1-dof) fields."""
        el = self.elements                       # (n_elem, 4)
        # scalar field: 4x4 blocks
        self.sca_rows = np.repeat(el, 4, axis=1).reshape(-1)
        self.sca_cols = np.tile(el, 4).reshape(-1)
        # vector field: dof = 2*node + {0,1}
        edof = np.empty((self.n_elem, 8), dtype=np.int64)
        edof[:, 0::2] = 2 * el
        edof[:, 1::2] = 2 * el + 1
        self.edof = edof
        self.vec_rows = np.repeat(edof, 8, axis=1).reshape(-1)
        self.vec_cols = np.tile(edof, 8).reshape(-1)

    # ------------------------------------------------------------------
    # convenience node selectors (boolean masks on nodes)
    def nodes_where(self, func):
        x, y = self.coords[:, 0], self.coords[:, 1]
        return np.where(func(x, y))[0]

    def element_centroids(self):
        return self.coords[self.elements].mean(axis=1)

    def gauss_point_coords(self):
        """Physical coordinates of every (element, gauss-point)."""
        # coords of 4 nodes per element: (n_elem, 4, 2); N: (4gp, 4node)
        nc = self.coords[self.elements]               # (n_elem, 4, 2)
        return np.einsum('gn,end->egd', self.gp_N, nc)  # (n_elem, 4gp, 2)
