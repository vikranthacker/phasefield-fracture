"""Validation 1 -- homogeneous critical stress vs closed form.

Checks the 1-D peak (critical) stress of the AT1 and AT2 models against the
analytical results
    AT1:  sigma_c = sqrt(3 E Gc / (8 l))
    AT2:  sigma_c = (3/16) sqrt(3) sqrt(E Gc / l)
for a range of regularization lengths.  Also plots the full homogeneous
stress-strain response against the closed-form curve.
"""
from __future__ import annotations

import os
import numpy as np

import _pf_path  # noqa: F401
from phasefield import Material, PhaseFieldModel, analytic
from phasefield.onedim import run_tension_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    mat = Material(E=210e3, nu=0.3, Gc=2.7)
    ells = [0.01, 0.02, 0.03, 0.05]
    rows = []
    for variant in ["AT2", "AT1"]:
        for ell in ells:
            mdl = PhaseFieldModel(mat, ell=ell, variant=variant)
            res = run_tension_test(mdl, L=1.0, n_elem=400, n_steps=300)
            num = res["peak_stress"]
            ana = analytic.critical_stress(mat.E, mat.Gc, ell, variant)
            err = 100 * abs(num - ana) / ana
            rows.append((variant, ell, num, ana, err))
            print(f"{variant}  l={ell:5.3f}  num={num:9.2f}  ana={ana:9.2f}  err={err:5.2f}%")

    # full stress-strain curve for one case
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for variant, color in [("AT2", "crimson"), ("AT1", "navy")]:
        mdl = PhaseFieldModel(mat, ell=0.02, variant=variant)
        res = run_tension_test(mdl, L=1.0, n_elem=400, n_steps=300)
        eps = res["strain"]
        ana_sig, _ = analytic.homogeneous_response(mat.E, mat.Gc, 0.02, eps, variant)
        ax[0].plot(eps, res["stress"], color=color, lw=2, label=f"{variant} (FEM)")
        ax[0].plot(eps, ana_sig, "--", color=color, lw=1.2, label=f"{variant} (analytic)")
    ax[0].set_xlabel("strain"); ax[0].set_ylabel("stress [MPa]")
    ax[0].set_title("Homogeneous response ($\\ell$=0.02)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    errs = np.array([r[4] for r in rows])
    ax[1].bar(range(len(errs)), errs,
              color=["crimson"] * len(ells) + ["navy"] * len(ells))
    ax[1].set_xticks(range(len(errs)))
    ax[1].set_xticklabels([f"{r[0]}\n{r[1]}" for r in rows], fontsize=7)
    ax[1].set_ylabel("critical-stress error [%]")
    ax[1].set_title("Error vs closed form")
    ax[1].axhline(2.0, color="k", ls=":", lw=1)
    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(os.path.join(RESULTS, "val_critical_stress.png"), dpi=130)
    print("max error: %.2f%%" % errs.max())
    print("saved:", os.path.join(RESULTS, "val_critical_stress.png"))
    return rows


if __name__ == "__main__":
    main()
