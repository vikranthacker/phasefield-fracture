# Theory and numerical method

This note documents the model and discretization implemented in the
`phasefield` package. The formulation follows the standard variational
(Ambrosio–Tortorelli) phase-field approach to brittle fracture used by the
FEniCSx / PhaseFieldX community, so results are directly comparable.

## 1. Variational phase-field model

Brittle fracture is modelled by regularizing the Griffith energy. For a body
Ω with displacement **u** and a scalar damage (phase) field `d ∈ [0,1]`
(`d = 0` intact, `d = 1` fully broken), the total energy is

```
E(u, d) = ∫_Ω g(d) ψ⁺(ε(u)) dΩ  +  ∫_Ω ψ⁻(ε(u)) dΩ
        + (Gc/c_w) ∫_Ω [ w(d)/ℓ + ℓ |∇d|² ] dΩ
```

* `g(d) = (1−d)² + k` — stiffness **degradation function** (`k` a small
  residual stiffness).
* `ψ⁺, ψ⁻` — tensile / compressive parts of the elastic strain energy density
  (the *split*, see §3).
* `Gc` — critical energy release rate (fracture toughness), may vary in space.
* `ℓ` — regularization length controlling the width of the smeared crack.
* `w(d)` — local dissipation function with normalization
  `c_w = 4 ∫₀¹ √w(β) dβ`.

Two regularizations are implemented:

| model | `w(d)` | `c_w` | elastic stage | critical stress (1-D) |
|-------|--------|-------|---------------|-----------------------|
| **AT2** (Bourdin) | `d²` | `2` | none (damage from zero load) | `σc = (3/16)√3 · √(E·Gc/ℓ)` |
| **AT1** (Pham–Marigo) | `d` | `8/3` | yes (true threshold) | `σc = √(3 E Gc /(8ℓ))` |

As `ℓ → 0` the regularized energy Γ-converges to the Griffith functional.

## 2. Governing equations (staggered form)

Minimizing `E` alternately in **u** and `d` gives two sub-problems.

**Elasticity** (at fixed `d`): `∇·σ = 0`, with the degraded stress
`σ = g(d) ∂ψ⁺/∂ε + ∂ψ⁻/∂ε`.

**Damage** (at fixed **u**), using the history field
`H = max_τ≤t ψ⁺` to enforce irreversibility (Miehe et al. 2010):

* **AT2:**  `Gc ℓ (−Δd) + (Gc/ℓ + 2H) d = 2H`
* **AT1:**  `(3Gcℓ/4)(−Δd) + 2H d = 2H − 3Gc/(8ℓ)`,  with `d ≥ d_old`, `d ≤ 1`.

Both are solved subject to the irreversibility constraint `d ≥ d_old`.

## 3. Energy split

* **isotropic** — `ψ⁺ = ½ ε:ℂ:ε`, `ψ⁻ = 0`. Exact for the 1-D benchmarks and
  adequate for tension-dominated problems (e.g. SENT).
* **Amor** (volumetric–deviatoric, Amor et al. 2009) —
  `ψ⁺ = ½K⟨tr ε⟩₊² + μ e:e`, `ψ⁻ = ½K⟨tr ε⟩₋²`, where `e` is the deviatoric
  strain. Only `ψ⁺` is degraded, which prevents cracking and interpenetration
  under compression — needed for mixed-mode and branching problems.

## 4. Discretization

* **Mesh:** structured grid of bilinear (Q1) quadrilaterals, plane strain.
  Because all cells are identical rectangles, the reference element operators
  (B-matrices, shape values, gradient/mass matrices) are evaluated once at the
  2×2 Gauss points and reused; global assembly is a single vectorized COO
  scatter scaled by `g(d)` (elasticity) or `H` and the material coefficients
  (damage).
* **Elasticity solve:** symmetric Dirichlet elimination + sparse LU (`spsolve`).
* **Damage solve:** box-constrained quadratic program
  `min ½dᵀA d − bᵀd  s.t.  d_old ≤ d ≤ 1`.
  - A **primal–dual active-set** (semismooth Newton) method is used in the
    2-D loading loop — it converges in a few iterations from the good initial
    guess provided by the previous load step.
  - A **projected SOR** solver (globally/monotonically convergent for the
    M-matrix operator) is used for the 1-D verification profiles, where the
    damaged region must grow connectedly from a poor initial guess and a
    pointwise active-set method can stall.

## 5. Solution algorithm

```
for each load increment (adaptive size):
    repeat (staggered / alternate minimization):
        1. assemble & solve degraded elasticity  -> u
        2. update history   H = max(H, ψ⁺(ε(u)))
        3. solve box-constrained damage QP        -> d   (d ≥ d_old)
    until max|Δd| < tol
    commit (H, d); record reaction force and energies
```

The load increment is grown while steps converge quickly (elastic regime) and
shrunk when a step needs many staggered iterations (crack propagation), which
keeps the total linear-solve count low without a monolithic scheme.

## 6. Verification

Closed-form references (`phasefield/analytic.py`) used by `validation/` and
`tests/`:

* homogeneous critical stress and strain (AT1, AT2);
* optimal crack profiles `exp(−|x|/ℓ)` (AT2) and `(1−|x|/2ℓ)²` (AT1);
* Γ-convergence consistency: dissipated energy of the optimal profile `= Gc`;
* LEFM / Griffith critical stress for a single-edge-notched plate,
  `σc = √(Gc E′) / (Y(a/W) √(πa))`.

## References

1. B. Bourdin, G.A. Francfort, J-J. Marigo. *The variational approach to
   fracture.* J. Elasticity **91** (2008) 5–148.
2. C. Miehe, M. Hofacker, F. Welschinger. *A phase field model for
   rate-independent crack propagation.* CMAME **199** (2010) 2765–2778.
3. H. Amor, J-J. Marigo, C. Maurini. *Regularized formulation of the
   variational brittle fracture…* JMPS **57** (2009) 1209–1229.
4. K. Pham, H. Amor, J-J. Marigo, C. Maurini. *Gradient damage models…*
   Int. J. Damage Mech. **20** (2011) 618–652.
5. E. Tanné, T. Li, B. Bourdin, J-J. Marigo, C. Maurini. *Crack nucleation in
   variational phase-field models of brittle fracture.* JMPS **110** (2018).
6. M. Hintermüller, K. Ito, K. Kunisch. *The primal-dual active set strategy as
   a semismooth Newton method.* SIAM J. Optim. **13** (2002) 865–888.
