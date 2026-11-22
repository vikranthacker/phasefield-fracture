"""Phase-field model configuration: material, regularization, energy split.

Two Ambrosio-Tortorelli regularizations are supported (the standard ones from
the variational phase-field fracture literature):

* ``AT2`` (Bourdin):   w(d) = d^2 ,  no elastic stage, damage from zero load.
* ``AT1`` (Pham-Marigo): w(d) = d ,  genuine elastic stage up to a threshold.

Degradation function  g(d) = (1-d)^2 + k_res.

Energy splits controlling the tensile driving force psi+:

* ``isotropic`` : psi+ = whole elastic energy (good for pure tension, exact for
  the 1-D closed-form benchmarks).
* ``amor``      : volumetric-deviatoric split (Amor et al., 2009) -- only the
  tensile volumetric part and the deviatoric part drive damage, preventing
  crack growth and interpenetration under compression.  Needed for branching
  and mixed-mode problems.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Material:
    E: float = 210.0e3     # Young's modulus  [MPa]  (default ~ steel, N/mm^2)
    nu: float = 0.3        # Poisson ratio
    Gc: float = 2.7        # critical energy release rate [N/mm = mJ/mm^2]

    @property
    def lame(self):
        E, nu = self.E, self.nu
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lam, mu

    @property
    def bulk_2d(self):
        """Plane-strain bulk modulus K = lambda + mu."""
        lam, mu = self.lame
        return lam + mu

    def D_planestrain(self):
        E, nu = self.E, self.nu
        c = E / ((1 + nu) * (1 - 2 * nu))
        return c * np.array([[1 - nu, nu, 0.0],
                             [nu, 1 - nu, 0.0],
                             [0.0, 0.0, (1 - 2 * nu) / 2.0]])


@dataclass
class PhaseFieldModel:
    material: Material
    ell: float = 0.015             # regularization length [mm]
    variant: str = "AT2"           # "AT1" or "AT2"
    split: str = "isotropic"       # "isotropic" or "amor"
    k_res: float = 1.0e-9          # residual stiffness

    # ---- degradation -------------------------------------------------
    def g(self, d):
        return (1.0 - d) ** 2 + self.k_res

    # ---- AT coefficients ---------------------------------------------
    @property
    def cw(self):
        return 2.0 if self.variant == "AT2" else 8.0 / 3.0

    def damage_coeffs(self, Gc):
        """Assembly coefficients of the (linearized) damage sub-problem.

        The damage weak form is written as

            grad * (grad d . grad v) + (mass + 2H) * d*v  =  (2H - source) * v

        Derived from  d/dd[g(d) psi+] + (Gc/cw) d/dd[w(d)/ell + ell |grad d|^2]=0
        with the history field H replacing psi+ for irreversibility.

        * AT2 (w=d^2, cw=2):  grad = Gc*ell,      mass = Gc/ell,  source = 0
        * AT1 (w=d,   cw=8/3): grad = 3Gc*ell/4,  mass = 0,       source = 3Gc/(8 ell)

        ``Gc`` may be a scalar or a per-element array (heterogeneous media).
        """
        ell = self.ell
        if self.variant == "AT2":
            grad = Gc * ell
            mass = Gc / ell
            source = 0.0 * Gc
        elif self.variant == "AT1":
            grad = 0.75 * Gc * ell
            mass = 0.0 * Gc
            source = 3.0 * Gc / (8.0 * ell)
        else:
            raise ValueError(f"unknown variant '{self.variant}'")
        return grad, mass, source

    @property
    def has_elastic_stage(self):
        """AT1 has a true damage threshold and needs a box-constrained solve."""
        return self.variant == "AT1"

    # ---- tensile strain-energy density psi+ at Gauss points ----------
    def psi_plus(self, strain_gp):
        """strain_gp : (..., 3) Voigt strains [exx, eyy, gxy].

        Returns psi+ with the configured split.
        """
        mat = self.material
        lam, mu = mat.lame
        exx = strain_gp[..., 0]
        eyy = strain_gp[..., 1]
        exy = 0.5 * strain_gp[..., 2]   # tensorial shear

        if self.split == "isotropic":
            tr = exx + eyy
            # psi = 0.5*lam*tr^2 + mu*(eps:eps)
            eps2 = exx ** 2 + eyy ** 2 + 2.0 * exy ** 2
            return 0.5 * lam * tr ** 2 + mu * eps2

        if self.split == "amor":
            K = mat.bulk_2d
            tr = exx + eyy
            tr_p = np.maximum(tr, 0.0)
            # deviatoric strain (plane strain, 2-D trace)
            edxx = exx - tr / 2.0
            edyy = eyy - tr / 2.0
            edev2 = edxx ** 2 + edyy ** 2 + 2.0 * exy ** 2
            return 0.5 * K * tr_p ** 2 + mu * edev2

        raise ValueError(f"unknown split '{self.split}'")
