"""Validation 4 -- mesh (h) convergence and the sigma_c ~ ell^{-1/2} scaling.

Two quantitative, closed-form-anchored checks that round out the verification:

Part A (h-convergence at fixed ell): the optimal 1-D crack profile is computed on
a sequence of refined meshes; its L2 error against the analytical profile and the
error of its dissipated energy against Gc are measured as functions of h/ell. This
documents (i) that the discretization converges, and (ii) the practical resolution
requirement h <~ ell/2 needed to resolve the regularized band.

Part B (ell-scaling): the homogeneous critical stress is measured over a range of
regularization lengths and compared with the closed form sigma_c = C sqrt(E Gc/ell);
a log-log fit recovers the theoretical exponent -1/2, confirming that ell is a
*material* length that sets the strength (not a vanishing numerical parameter).

Run from the repository root:
    python validation/val04_convergence.py
"""
from __future__ import annotations

import os
import numpy as np

import _pf_path  # noqa: F401
from phasefield import Material, PhaseFieldModel, analytic
from phasefield.onedim import (run_tension_test, optimal_profile_1d,
                               fracture_energy_1d)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def h_convergence(variant="AT2", ell=0.04, L=1.0):
    mat = Material(E=210e3, nu=0.3, Gc=2.7)
    mdl = PhaseFieldModel(mat, ell=ell, variant=variant)
    rows = []
    for n_elem in (12, 25, 50, 100, 200, 400, 800):
        x, d = optimal_profile_1d(mdl, L=L, n_elem=n_elem)
        dana = analytic.crack_profile(x, ell, variant)
        l2 = float(np.sqrt(np.mean((d - dana) ** 2)))
        Ef = fracture_energy_1d(x, d, mdl)
        e_err = 100.0 * abs(Ef - mat.Gc) / mat.Gc
        h = L / n_elem
        rows.append((n_elem, h, h / ell, l2, e_err))
    return rows


def ell_scaling(variant="AT2"):
    mat = Material(E=210e3, nu=0.3, Gc=2.7)
    rows = []
    ells = [0.010, 0.015, 0.020, 0.030, 0.050]
    for ell in ells:
        mdl = PhaseFieldModel(mat, ell=ell, variant=variant)
        res = run_tension_test(mdl, L=1.0, n_elem=400, n_steps=400)
        num = res["peak_stress"]
        ana = analytic.critical_stress(mat.E, mat.Gc, ell, variant)
        rows.append((ell, num, ana, 100.0 * abs(num - ana) / ana))
    ell_arr = np.array([r[0] for r in rows])
    num_arr = np.array([r[1] for r in rows])
    slope = np.polyfit(np.log(ell_arr), np.log(num_arr), 1)[0]
    return rows, slope


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=== Part A: h-convergence of the optimal profile (ell=0.04) ===")
    conv = {}
    for v in ("AT2", "AT1"):
        print(f"--- {v} ---")
        print(f"{'n_elem':>7} {'h':>8} {'h/ell':>7} {'L2_err':>11} {'E_err%':>9}")
        rows = h_convergence(v)
        conv[v] = rows
        for n, h, hl, l2, ee in rows:
            print(f"{n:7d} {h:8.4f} {hl:7.3f} {l2:11.3e} {ee:9.3f}")

    print("\n=== Part B: sigma_c ~ ell^{-1/2} scaling ===")
    scal = {}
    for v in ("AT2", "AT1"):
        rows, slope = ell_scaling(v)
        scal[v] = (rows, slope)
        print(f"--- {v}:  fitted exponent = {slope:+.4f}  (theory -0.5) ---")
        print(f"{'ell':>6} {'sigma_num':>11} {'sigma_ana':>11} {'err%':>7}")
        for ell, num, ana, err in rows:
            print(f"{ell:6.3f} {num:11.2f} {ana:11.2f} {err:7.2f}")

    _plot(conv, scal, RESULTS)


def _plot(conv, scal, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    floor = 2e-4
    for v, c in (("AT2", "C0"), ("AT1", "C3")):
        hl = np.array([r[2] for r in conv[v]])
        ee = np.array([r[4] for r in conv[v]])  # energy error (%)
        ax[0].loglog(hl, np.maximum(ee, floor), "o-", color=c, label=v)
    ax[0].axvline(0.5, color="k", ls="--", lw=0.8, alpha=0.6)
    ax[0].text(0.52, 5.0, r"$h=\ell/2$", fontsize=8)
    ax[0].set_xlabel(r"$h/\ell$")
    ax[0].set_ylabel(r"dissipated-energy error (\%)")
    ax[0].set_title("Mesh convergence (fracture energy)")
    ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)

    for v, c in (("AT2", "C0"), ("AT1", "C3")):
        rows, slope = scal[v]
        ell = np.array([r[0] for r in rows])
        num = np.array([r[1] for r in rows])
        ax[1].loglog(ell, num, "s", color=c, label=f"{v} (slope {slope:+.2f})")
        ax[1].loglog(ell, num[0] * (ell / ell[0]) ** (-0.5), "-", color=c, lw=0.8, alpha=0.6)
    ax[1].set_xlabel(r"regularization length $\ell$")
    ax[1].set_ylabel(r"critical stress $\sigma_c$")
    ax[1].set_title(r"$\sigma_c \sim \ell^{-1/2}$ scaling")
    ax[1].legend(fontsize=8); ax[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(results, "val_convergence.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
