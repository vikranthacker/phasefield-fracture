"""Spatially varying fracture-toughness fields for heterogeneous media.

Each generator returns a per-element array ``Gc_elem`` (length ``mesh.n_elem``)
built from the element centroids, suitable for :class:`StaggeredSolver`.
"""
from __future__ import annotations

import numpy as np

from .mesh import StructuredMesh


def homogeneous(mesh: StructuredMesh, Gc: float):
    return np.full(mesh.n_elem, Gc)


def circular_inclusions(mesh: StructuredMesh, Gc_matrix: float,
                        inclusions, Gc_inclusion: float):
    """Tough (or weak) circular inclusions.

    ``inclusions`` : list of (cx, cy, radius).
    """
    c = mesh.element_centroids()
    Gc = np.full(mesh.n_elem, Gc_matrix)
    for (cx, cy, r) in inclusions:
        inside = (c[:, 0] - cx) ** 2 + (c[:, 1] - cy) ** 2 <= r ** 2
        Gc[inside] = Gc_inclusion
    return Gc


def sinusoidal(mesh: StructuredMesh, Gc_mean: float, amplitude: float,
               wavelength: float, direction=(1.0, 0.0)):
    """Smoothly layered toughness:  Gc = Gc_mean (1 + A sin(2 pi k . x))."""
    c = mesh.element_centroids()
    k = np.array(direction, float)
    k = k / (np.linalg.norm(k) + 1e-30) / wavelength
    phase = 2.0 * np.pi * (c[:, 0] * k[0] + c[:, 1] * k[1])
    return Gc_mean * (1.0 + amplitude * np.sin(phase))


def gaussian_random_field(mesh: StructuredMesh, Gc_mean: float, cov: float,
                          corr_length: float, seed: int = 0, clip=(0.2, 3.0)):
    """Log-normal random toughness with a prescribed correlation length.

    A white-noise grid is smoothed by a Gaussian kernel (spectral filter) to
    impose spatial correlation, then mapped through exp() for positivity.
    """
    rng = np.random.default_rng(seed)
    nx, ny = mesh.nx, mesh.ny
    white = rng.standard_normal((ny, nx))

    # Gaussian smoothing in Fourier space
    fx = np.fft.fftfreq(nx, d=mesh.hx)
    fy = np.fft.fftfreq(ny, d=mesh.hy)
    FX, FY = np.meshgrid(fx, fy)
    kernel = np.exp(-2.0 * (np.pi * corr_length) ** 2 * (FX ** 2 + FY ** 2))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * kernel))
    field = (field - field.mean()) / (field.std() + 1e-30)

    sigma = np.sqrt(np.log(1.0 + cov ** 2))
    Gc = Gc_mean * np.exp(sigma * field - 0.5 * sigma ** 2)
    Gc = np.clip(Gc, clip[0] * Gc_mean, clip[1] * Gc_mean)
    return Gc.reshape(-1)
