"""phasefield -- a self-contained phase-field brittle-fracture solver.

A compact, dependency-light (NumPy/SciPy) implementation of the Ambrosio-
Tortorelli (AT1/AT2) variational phase-field model for brittle fracture, with a
staggered solver, heterogeneous-material support, and verification against
closed-form solutions.  Model choices follow the standard variational
phase-field fracture literature, so results are directly comparable.
"""
from .mesh import StructuredMesh
from .model import Material, PhaseFieldModel
from .fem import Assembler, apply_dirichlet, solve_bounded
from .solver import StaggeredSolver, StepResult, adaptive_load
from . import analytic, materials, onedim

__all__ = [
    "StructuredMesh", "Material", "PhaseFieldModel", "Assembler",
    "apply_dirichlet", "solve_bounded", "StaggeredSolver", "StepResult",
    "adaptive_load", "analytic", "materials", "onedim",
]

__version__ = "0.1.0"
