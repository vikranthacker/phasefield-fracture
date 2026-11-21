"""Validation 2 -- optimal crack (damage) profile vs closed form.

The minimizer of the fracture functional with d=1 at the crack must reproduce
    AT2:  d(x) = exp(-|x|/l)
    AT1:  d(x) = (1 - |x|/2l)^2  for |x| <= 2l
This isolates the gradient/mass discretization and the Gamma-convergence
constant.  Also reports mesh convergence of the L2 error.
"""
from __future__ import annotations

import os
import numpy as np

import _pf_path  # noqa: F401
from phasefield import Material, PhaseFieldModel, analytic
from phasefield.onedim import optimal_profile_1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    mat = Material(E=210e3, nu=0.3, Gc=2.7)
    ell = 0.05
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    for variant, color in [("AT2", "crimson"), ("AT1", "navy")]:
        x, d = optimal_profile_1d(PhaseFieldModel(mat, ell=ell, variant=variant),
                                  L=1.0, n_elem=1000)
        dana = analytic.crack_profile(x, ell, variant)
        l2 = np.sqrt(np.mean((d - dana) ** 2))
        m = np.abs(x) < 4 * ell
        ax[0].plot(x[m] / ell, d[m], color=color, lw=2, label=f"{variant} FEM")
        ax[0].plot(x[m] / ell, dana[m], "--", color=color, lw=1.2,
                   label=f"{variant} analytic")
        print(f"{variant}: profile L2 error = {l2:.4f}")

    ax[0].set_xlabel("$x/\\ell$"); ax[0].set_ylabel("damage $d$")
    ax[0].set_title(f"Optimal crack profile ($\\ell$={ell})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # mesh convergence (AT2)
    ncells = [100, 200, 400, 800, 1600]
    errs = []
    for nc in ncells:
        x, d = optimal_profile_1d(PhaseFieldModel(mat, ell=ell, variant="AT2"),
                                  L=1.0, n_elem=nc)
        dana = analytic.crack_profile(x, ell, "AT2")
        errs.append(np.sqrt(np.mean((d - dana) ** 2)))
    hvals = 1.0 / np.array(ncells)
    ax[1].loglog(hvals / ell, errs, "o-", color="crimson")
    ax[1].set_xlabel("$h/\\ell$"); ax[1].set_ylabel("profile $L_2$ error")
    ax[1].set_title("Mesh convergence (AT2)")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(os.path.join(RESULTS, "val_crack_profile.png"), dpi=130)
    print("saved:", os.path.join(RESULTS, "val_crack_profile.png"))


if __name__ == "__main__":
    main()
