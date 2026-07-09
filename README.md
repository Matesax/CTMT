# CTMT — Resolution Geometry of Observable Transport

**What it is, in one sentence:** given a forward model's Jacobian and an observation covariance, CTMT is a small, rigorous geometry that says *which directions are resolvable from measurements, how the unresolvable ones still matter through correlation, and which transformations preserve that structure.*

It is an **observational / estimation geometry**, built on standard linear algebra, information geometry, and inverse-problem theory. It is **not** a theory of physics, and it makes no claim about the fundamental nature of space, time, or matter. If you are looking for a "theory of everything," this is not that, and the [Scope](#scope-and-non-claims) section says so explicitly.

---

## Core mechanism (read this before drawing conclusions)

Everything is built from one object. Fix an operating point of a system probed by admissible perturbations, and record the first-order response and its observational uncertainty.

- **Response.** Let `J = ∂(observations)/∂(perturbations)` be the local Jacobian of the forward map, on an observation space `O`.
- **Reference metric.** `O` carries a fixed inner product `g` set by the measurement protocol — units, or the noise-whitening metric `g = Σ⁻¹` when the covariance is nondegenerate. This metric is *part of the data*; it is what makes the decomposition below coordinate-independent.
- **Two sectors.**
  - `R = Im(J)` — the **resolved sector**: directions distinguishable at first order by the response.
  - `N = R^⊥_g` — the **null sector**: directions not activated at first order.
- **Covariance.** The observation covariance `Σ` decomposes against `O = R ⊕ N` into `Σ_R`, `Σ_N`, and the **resolved–null coupling** `C_RN = P_R Σ P_N`.

The single primitive object is therefore

```
  G = (O, g; R, Σ)          N := R^⊥_g   (derived)
```

The **coupling `C_RN` is the central object.** When it is zero, resolved and null uncertainties are independent. When it is nonzero, the null sector — though it carries no first-order signal on its own — is statistically tied to the
resolved sector, and that tie has consequences that run through every layer of the theory.

Two points that prevent common first-glance misreadings:

1. **This is not "just Fisher geometry."** The Fisher/response metric `J⁻ᵀ Σ⁻¹ J` appears here as a *derived diagnostic* (for a Gaussian model it is exactly the Fisher information), not as the primitive. The primitive is the full geometry `G`, of which Fisher is one shadow.
2. **This is first-order and local.** The objects are Jacobians and covariances at an operating point. Global assembly and dynamics are separate, partly-open layers (see below). Nothing here derives turbulence, scaling laws, or physical constants.

---

## What is proved (the defensible core)

The results below are established with standard mathematics (Cramér–Rao, Gauss–Markov, Schur complements, compact-group actions, Čencov, Mori–Zwanzig, Lyapunov theory) and are numerically checked. Each lives in a short paper.

| Result | Statement | Paper |
|---|---|---|
| **Decomposition** | `O = R ⊕ N` is `g`-orthogonal; invariant under the isometry group `O(g)` (and *not* under general coordinate change — the metric is required). | Foundations |
| **Covariance geometry** | Block structure `(Σ_R, Σ_N, C_RN)`; the singular-covariance case handled via the Moore–Penrose pseudo-inverse, with the finite-information criterion `R ⊆ range(Σ)`. | Covariance |
| **Fisher is derived** | For a Gaussian model, `Fisher = Jᵀ Σ⁻¹ J = J_Rᵀ (Σ/Σ_N)⁻¹ J_R` (inverse Schur complement); hence Fisher is provably *insufficient* to determine the covariance geometry. | Covariance |
| **Symmetry** | Admissible maps form a category; the automorphism group is a **coupling stabilizer** — nonzero `C_RN` rigidifies it. | Foundations / Covariance |
| **Estimation gain** | A coupling-aware estimator strictly beats a resolved-only one (Cramér–Rao), by an amount set by the canonical correlations; the confidence-volume reduction **equals** the resolved–null mutual information. Requires a *calibrated null baseline*. | Pillar 1 |
| **Classification** | Complete invariant on the generic stratum; moduli dimension `rq + r + q`; the coupling is an **irreducible** coordinate. Canonical correlations alone are *not* complete for `r, q ≥ 2`. | Pillar 3 |
| **Dynamics** | Linear-stochastic layer: covariance obeys the Lyapunov equation; coupling is *generated* by the cross-drift; eliminating the null sector gives an exact **Mori–Zwanzig memory kernel** `K(τ) = A_RN e^{A_NN τ} A_NR`. | Dynamics |
| **Globalization** | Charts glue by Čech cocycles with structure group `Aut(G)`; holonomy is the obstruction to trivialization (not a failure of gluing). | Atlas |
| **Synthesis** | All layers are functions of the one primitive, tied by exact identities; a single-knob demonstration shows every diagnostic moving in lockstep. | Synthesis |

---

## For sensor, measurement, and inverse-problem people

If you have a **forward model and a noise covariance**, this framework tells you, concretely and without physics:

- which parameter directions your instrument can and cannot resolve (`R` vs `N`);
- how **correlated-but-signal-free channels** (a calibrated null sector) can be used as noise references to sharpen your estimates — with an exact formula for the variance you gain (Pillar 1);
- when two measurement setups are **genuinely equivalent** vs merely Fisher-equivalent (Fisher agreement is necessary, not sufficient);
- a complete, checkable **invariant** for classifying observation charts up to admissible transformation (Pillar 3).

The one operational caveat worth stating up front: the null-sector estimation gain requires the null baseline to be **calibrated**. If it is a free nuisance, the gain vanishes — which is itself a testable prediction.

---

## Scope and non-claims

This section is deliberate. Please read it before citing or extending the work.

- **Not a physical theory.** CTMT is a geometry of observation and estimation. It does not claim that any physical system *must* realize its structure; that is an empirical question.
- **`C_RN` is not (yet) known to be physical.** The resolved–null coupling is a proven mathematical and estimation-theoretic object. Whether a measured nonzero coupling reflects genuine transport physics or protocol/instrument structure is open.
- **"Seepage" has a precise, limited meaning here.** It denotes the life-cycle of the coupling `C_RN` (generation → memory kernel → information → estimation → rigidity → classification → holonomy). It is a *static, second-order / correlational* notion unless the dynamical hypotheses are separately met. It is not a physical flow of substance.
- **Explicitly retired.** Earlier drafts attempted to derive turbulence / Kolmogorov scaling from Fisher-rank loss, and to "diagnose" constants such as
  π. Those claims are **withdrawn**: the turbulence arguments rested on an unproven conservation law, the Kolmogorov step was circular (and its exponent bookkeeping did not close), and constant-recovery through a flexible kernel is reparametrization, not confirmation (Bernstein's theorem). The framework is stronger without them.

If a claim is not in the [proved core](#what-is-proved-the-defensible-core), it is either an open problem or a non-claim. There is no hidden physics.

---

## Origins (historical, pre-rigorous)

The programme began from a childhood intuition — an unresolved influence reaching an observable with a delay — and passed through a phenomenological "Chronotopic Kernel" stage using the vocabulary of rhythm, modulation, rupture, and seepage across layers. That stage is preserved for intellectual history, but it is **motivation, not result**.

The one idea from that stage that survived into the rigorous theory is worth stating, because it is the honest conceptual seed:

> A perfectly isolated object cannot participate in observation, interaction, or
> transport. Operational identity therefore requires *unresolved* structure that
> couples the object to a wider compatibility structure.

In the mature theory, the role of the null sector `N` and the coupling `C_RN` is now exact.  
The original delayed‑feedback intuition has hardened into the Mori–Zwanzig **memory kernel**, and every layer that follows from it—covariance coupling, information gain, estimator tightening, rigidity of automorphisms, and holonomy—is rigorously derived from that kernel.

Everything *beyond* the kernel stage—the rhythm‑first ontology, emergent spacetime, cross‑domain unification, and the π‑coherence idea—is **explicitly marked as speculative** and **not part of any claim made here**. These early explorations remain in `index.html` only as historical context for readers interested in the developmental path.

CTMT required the broader Chronotopic Theory of Matter and Time as a creative scaffold: a place to write down twenty years of intuitions, formulas, and structural guesses. Without externalizing that long formative process, the mature CTMT machine could not have been built. That early material is now clearly separated from the rigorous core. Historic development is captured in `index.html`.

---

## Open problems (current state)

Local geometry is stabilized; the frontier is global and physical.

- **Globalization (principal frontier).** Existence/uniqueness of coherent atlases for empirically estimated transitions; classification of admissible holonomy groups.
- **Coupling physics.** Is `C_RN` a genuine transport phenomenon? The precise, falsifiable probe is a **fluctuation–dissipation cross-exponent test** relating the memory-kernel decay to an independently measured fluctuation exponent — which requires real data.
- **Dynamical buildup.** Conditions on the drift/noise `(A, Q)` under which a directed-information or Fisher-gain functional is monotone (the "buildup ladder"). Currently the *mechanism* (generation + kernel) is proved; monotone accumulation is conjectural.
- **Non-generic strata.** The complete invariant is established on the distinct-eigenvalue stratum; repeated-eigenvalue strata coarsen and are open.
- **Sector-relative covariance protocol.** Strict and conformal protocols are handled; the sector-relative option is unresolved.
- **Higher-order lifts.** How much of the first-order classification survives under Hessian / phase-Hessian transport.
- **Universality.** Domain of validity beyond the synthetic and tested-class systems (coil, diffusion, reaction–diffusion, seismic-style charts).

---

## Papers

1. **Foundations** — Resolution Geometry of Observable Transport Systems _Paper 1_
2. **The CTMT Covariance Geometry** — Covariance geometry, singular support, coupling rigidity _Paper 2_
3. **The CTMT Atlas and Null-Sector Information** — Orbit–stabilizer classification, Čech globalization, null-sector information _Paper 3_
4. **The CTMT Dynamics Skeleton** — Linear-stochastic layer and the Mori–Zwanzig seepage kernel (skeleton + open ladder)
5. **Coupling-Aware Estimation in CTMT** — Coupling-aware estimation: a Cramér–Rao efficiency gap
6. **The CTMT and the Omori Law** — CTMT and the Omori law: a consistency result and a precise limit *(documented negative/consistency result)*
7. **Complete Invariants of CTMT Covariance Resolution Geometry** — Complete invariants and the necessity of coupling
8. **The CTMT - Testament of 22 years** — CTMT as a single machine (one-knob demonstration)

---

## Status, citation, license

- **Local morphism classification:** stabilized.
- **Global transport geometry & physical interpretation:** open.

DOI: [10.5281/zenodo.18229539](https://doi.org/10.5281/zenodo.18229539)
· OSF: `10.17605/OSF.IO/RFE8N`

Author: Matěj Rada · Licensed CC BY-NC-ND 4.0.

Serious questions, counterexamples, and attempts to break the theorems are welcome — the framework is designed to be falsifiable, and a clean refutation of any proved statement is a contribution.
