"""Plotting / export helpers (matplotlib, Agg-friendly)."""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_damage(mesh, d, ax=None, title=None, Gc_elem=None, show_toughness=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.2))
    D = d.reshape(mesh.ny + 1, mesh.nx + 1)
    extent = [0, mesh.Lx, 0, mesh.Ly]
    im = ax.imshow(D, origin="lower", extent=extent, cmap="inferno",
                   vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="damage $d$")
    if show_toughness and Gc_elem is not None:
        Gg = Gc_elem.reshape(mesh.ny, mesh.nx)
        ax.contour(np.linspace(0, mesh.Lx, mesh.nx),
                   np.linspace(0, mesh.Ly, mesh.ny),
                   Gg, levels=6, colors="cyan", linewidths=0.4, alpha=0.5)
    if title:
        ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return ax


def plot_force_displacement(disps, forces, ax=None, label=None, **kw):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.plot(disps, forces, label=label, **kw)
    ax.set_xlabel("imposed displacement")
    ax.set_ylabel("reaction force")
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    return ax


def toughness_map(mesh, Gc_elem, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.2))
    G = Gc_elem.reshape(mesh.ny, mesh.nx)
    im = ax.imshow(G, origin="lower", extent=[0, mesh.Lx, 0, mesh.Ly],
                   cmap="viridis", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$G_c$")
    if title:
        ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    return ax
