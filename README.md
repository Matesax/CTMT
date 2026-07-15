# Resolution Geometry (RG) — *formerly CTMT*

**One sentence.** Given a forward model's Jacobian and an observation covariance, Resolution Geometry is a small, rigorous geometry that says *which directions are resolvable from measurements, how the unresolvable ones still matter through correlation, and which transformations preserve that structure.*

It is an **observational / estimation geometry**, built on standard linear algebra, information geometry, and inverse-problem theory. It is **not** a theory of physics, and it makes no claim about the fundamental nature of space, time, or matter. See [Scope and non-claims](#scope-and-non-claims).

> **Naming.** The mature core is **Resolution Geometry (RG)**, stripped of physics vocabulary. Files prefixed **`RG -`** are the current, CTMT-free corpus. Files prefixed **`The CTMT …` / `CTMT …`** are the original, detailed versions of the same results (still valid; older vocabulary). Everything under [Historic](#historic--pre-rigorous--retired-quarantined) is pre-rigorous or retired and is kept for intellectual history only.

---

## Repository conventions

- **Theorem package = one PDF.**
- **Battery / stress-test bundle = a `.zip` with the *exact same name* as its PDF.** If a PDF `Foo.pdf` has an accompanying `Foo.zip`, that zip is the runnable numerical falsification battery for that package. A ✓ in the tables below marks packages that ship a battery zip.
- **Every load-bearing theorem is numerically verified before it is written.** Retirements are recorded, not hidden ([honesty ledger](#the-honesty-ledger--what-was-retired)).
- **Status tags:** `[proved]` defensible core · `[semi-win]` exact result, interpretation deliberately bounded · `[open]` delimited frontier · `[retired]` withdrawn, kept for history.

---

## Core mechanism (read this before drawing conclusions)

Everything is built from one object. Fix an operating point probed by admissible perturbations; record the first-order response and its observational uncertainty.

- **Response.** `J = ∂(observations)/∂(perturbations)` — the local Jacobian on an observation space `O`.
- **Reference metric.** `O` carries a fixed inner product `g` set by the measurement protocol (units, or the whitening metric `g = Σ⁻¹`). The metric is *part of the data*; it makes the decomposition coordinate-independent.
- **Two sectors.** `R = Im(J)` — the **resolved sector**; `N = R^⊥_g` — the **null sector**.
- **Covariance.** `Σ` decomposes against `O = R ⊕ N` into `Σ_R`, `Σ_N`, and the **resolved–null coupling** `C_RN = P_R Σ P_N`.

The single primitive is

```
  G = (O, g; R, Σ)          N := R^⊥_g   (derived)
```

The **coupling `C_RN` is the central invariant.** Zero coupling → resolved and null uncertainties are independent. Nonzero coupling → the null sector, though signal-free at first order, is statistically tied to the resolved sector, with consequences that run through every layer.

Two guards against first-glance misreadings:

1. **This is not "just Fisher geometry."** The Fisher/response metric `Jᵀ Σ⁻¹ J` appears as a *derived diagnostic* (a Gaussian shadow), not the primitive. The primitive is the full geometry `G`.
2. **This is first-order and local.** Global assembly and dynamics are separate, partly-open layers. Nothing here derives turbulence, scaling laws, or physical constants.

---

## The canonical corpus (reading order)

The mature RG series, top-down. Read the **Complete Framework** first for the whole machine, then descend.

| # | Paper | Role | File | Battery |
|---|---|---|---|---|
| 0 | **Complete Framework** | Grand synthesis — the whole machine, honest impact, curvature spine | `RG - Complete Framework.pdf` | — |
| 1 | **Foundations** | Anchor: the object `G`, sectors, derived Fisher, automorphism rigidity | `Foundation.pdf` | — |
| 2 | **Reconstruction / Identifiability** | Geometry is identifiable from `(J,Σ,g)` up to gauge; Fisher is insufficient | `RG - Reconstruction Identifiability.pdf` | ✓ |
| 3 | **Atlas / Globalization** | Principal `Aut(G)`-bundle; Čech cocycle + holonomy; decidable edges | `RG - Atlas Globalization.pdf` | ✓ |
| 4 | **Global Structure** | `Aut(G)` compact Lie group; flat classification = character variety | `RG - Global Structure.pdf` | ✓ (`_proto`) |
| 5 | **Transport Invariants** | What survives transport: physical sector invariant; reactive geometric phase | `RG - Transport Invariants.pdf` | — |
| 6 | **Second-Order Moduli** | Null sector opens as the second fundamental form; normal holonomy; bounded moduli | `RG - Second Order Moduli.pdf` | — |

> **Superseded duplicate:** `Resolution Geometry - Atlas.pdf` (+ `.zip`) is an earlier Atlas draft; **#3 above supersedes it.**

---

## Pillars (detailed results)

The three numbered pillars carry the framework's sharpest standalone results. These are the CTMT-branded detailed versions behind the corpus above.

| Pillar | Statement | Status | File | Battery |
|---|---|---|---|---|
| **Pillar 1 — Estimation gap** | Coupling-aware estimation beats resolved-only; confidence-volume reduction **equals** resolved–null mutual information. Needs a *calibrated* null baseline. | `[proved]` | `Coupling-Aware Estimation in CTMT.pdf` | — |
| **Pillar 2 — Omori limit** | Consistency, not confirmation: a completely-monotone kernel hosts any power law (Bernstein). A documented **negative result**; the genuine test needs real data. | `[proved]` (negative) | `The CTMT and the Omori Law.pdf` | — |
| **Pillar 3 — Complete invariants** | Complete invariant on the generic stratum; moduli dim `rq+r+q`; the coupling is **irreducible** (canonical correlations alone incomplete for `r,q ≥ 2`). | `[proved]` | `Complete Invariants of CTMT Covariance Resolution Geometry.pdf` | — |

---

## Core layers & supporting results

Detailed papers behind individual corpus layers, plus verified supporting results and applications.

| Package | Role / layer | Status | File | Battery |
|---|---|---|---|---|
| **Covariance geometry** | Fisher derived (inverse Schur); singular case via pseudo-inverse; `R ⊆ range(Σ)` | `[proved]` | `The CTMT Covariance Geometry.pdf` | — |
| **Atlas & null-sector information** | Orbit–stabilizer classification, Čech globalization, null-sector info | `[proved]` | `The CTMT Atlas and Null-Sector Information.pdf` | — |
| **Dynamics skeleton** | Lyapunov covariance; coupling generation; Mori–Zwanzig kernel `K(τ)=A_RN e^{A_NN τ} A_NR` | `[proved]` + `[open]` ladder | `The CTMT Dynamics Skeleton.pdf` | — |
| **Dynamics II** | Physical-frontier roadmap (entropy production; Gaussian ceiling) | `[proved]` + `[open]` | `The CTMT Dynamics II.pdf` | — |
| **Reactive Lorentzian signature** | `(1, n−1)` signature via Sylvester congruence; sign **derived**, not inserted | `[semi-win]` | `A Derived Lorentzian-Type Signature.pdf` | — |
| **Synthesis (one-knob)** | All layers as functions of the one primitive; single-knob lockstep demo | `[proved]` | `The CTMT - Testament of 22 years.pdf` | — |
| **Automorphisms of resolution** | Coupling stabilizer / automorphism rigidity support | `[proved]` | `The CTMT Automorphisms of Resolution.pdf` | ✓ (+ `Automorphisms Support Battery.zip`) |
| **Compatibility lock & holonomy** | Holonomy as trivialization obstruction (atlas support) | `[proved]` | `The CTMT Compatibility Lock and Holonomy Obstruction.pdf` | — |
| **Resolved–null coupling** | The central object, standalone treatment | `[proved]` | `The CTMT Resolved–Null Covariance Coupling.pdf` | — |
| **Independent-protocol recovery** | Coupling recovered across protocols (identifiability support) | `[proved]` | `Independent-Protocol Recovery of Resolved–Null Coupling.pdf` | — |
| **Čencov–inversion compatibility** | Čencov uniqueness of the Fisher metric under inversion (foundations support) | `[proved]` | `Čencov–Inversion Compatibility for CTMT Transport.pdf` | — |
| **Morphism state** | Morphism-state diagnostics | `[proved]` | `The CTMT Morphism State.pdf` | — |
| **Morphism state — coil** | Application battery (synthetic coil) | `[proved]` | `The CTMT Morphism State on the Synthetic Coil Battery.pdf` | ✓ |
| **Seismic morphism** | Application (seismic-style charts) | `[proved]` | `The CTMT Seismic Morphism.pdf` | ✓ |
| **Transport-class rigidity** | Rigidity of transport classes | `[proved]` | `The CTMT Transport-Class Rigidity.pdf` | ✓ |
| **Trajectory-gated persistence** | Persistence diagnostics | `[proved]` | `The CTMT Trajectory-Gated Persistence.pdf` | ✓ |
| **Trajectory-resolved batteries** | Trajectory battery suite | `[proved]` | `Trajectory-Resolved CTMT Batteries.pdf` | ✓ |

---

## For sensor, measurement, and inverse-problem people

If you have a **forward model and a noise covariance**, the framework tells you, without physics:

- which parameter directions your instrument can and cannot resolve (`R` vs `N`);
- how **correlated-but-signal-free channels** (a calibrated null sector) sharpen estimates — with an exact formula for the variance gained (**Pillar 1**; gain = mutual information);
- when two setups are **genuinely equivalent** vs merely Fisher-equivalent (Fisher agreement is necessary, not sufficient);
- a complete, checkable **invariant** for classifying observation charts (**Pillar 3**).

**Operational caveat:** the null-sector gain requires the null baseline to be **calibrated**. If it is a free nuisance, the gain vanishes — itself a testable prediction.

---

## Scope and non-claims

Read before citing or extending.

- **Not a physical theory.** RG is a geometry of observation and estimation. It does not claim that any physical system *must* realize its structure.
- **`C_RN` is not (yet) known to be physical.** Whether a measured coupling reflects transport physics or protocol/instrument structure is `[open]`.
- **"Seepage" has a precise, limited meaning:** the life-cycle of `C_RN` (generation → kernel → information → estimation → rigidity → classification → holonomy). It is static/correlational unless dynamical hypotheses are separately met — not a physical flow.
- **Formal kinship ≠ physics.** Connection, curvature, holonomy, the Lorentzian signature, the geometric phase, and the second fundamental form are objects of statistics and submanifold/bundle geometry. Their resemblance to gauge theory or spacetime is formal.
- **Globalization is reduced and bounded, not enumerated.** The character variety is compact, real-algebraic, of explicit dimension, symplectic on its smooth locus — but not computed.

If a claim is not in the proved core or pillars, it is an open problem or a non-claim. **There is no hidden physics.**

---

## The honesty ledger — what was retired

The credibility of the surviving core rests on removing what could not be supported. `[retired]`, with reasons:

- **Kolmogorov turbulence from Fisher-rank loss** — rested on an unproven conservation law; the step was circular and the exponent bookkeeping did not close (`k·k^{-4/3} = k^{-1/3} ≠ k^{-5/3}`).
- **π-diagnosis / constant recovery** — a flexible kernel reproducing a constant is reparametrization, not confirmation (Bernstein).
- **Emergent spacetime, Fisher-induced Lorentz signature as gravity, nodes-of-presence, quantum/biological anchorings** — no support; removed by the framework's own falsification discipline.
- **Circular Omori validation** — rewritten as Pillar 2's documented negative result.

Retirement/correction records are preserved (e.g. `Correction and Maturation of the CTMT Redshift Claim.pdf` + `.zip`).

---

## Open problems (current frontier)

Local geometry is stabilized; the frontier is global and physical.

- **Globalization** — existence/uniqueness of coherent atlases for *estimated* transitions; **computing** the character variety / admissible holonomy groups (bounded, not enumerated).
- **Coupling physics** — is `C_RN` genuine transport? Probe: a **fluctuation–dissipation cross-exponent test** (needs real data).
- **Dynamical buildup** — drift/noise `(A,Q)` conditions for a monotone information/Fisher-gain ladder (mechanism proved; monotonicity conjectural).
- **Non-generic strata** — repeated-eigenvalue strata coarsen and are open.
- **Sector-relative covariance protocol** — strict/conformal handled; sector-relative unresolved.
- **Higher-order lifts** — second-order structure opened (second fundamental form, normal holonomy); its observable signature and full classification are open.
- **Universality** — domain of validity beyond synthetic/tested classes; requires real datasets (not establishable synthetically).

---

## Historic / pre-rigorous / retired (quarantined)

Preserved for intellectual history only. **Not part of any claim made here.** Historic developmental path is in `index.html`.

- **Chronotopic Theory of Matter and Time** — `- I`, `- II`, `- III`, `- IV`, `- CHI`, `- Causality`, `- Seepage`
- **Chronotopic Metric Theory** — `.pdf`, `- Physics`, `- Trigonometry`
- **Retired physics attempts** — `CTMT Universal Causal Transport Law for Energy`, `The CTMT boundary for Newton G` (+ `.zip`), `Constants and π–Factors in Radiative Physics`, `Emergent Time and Rank–Stabilised Causal Ordering`, `Fisher–Stabilised Coherence Geometry`, `Seepage, Fisher Rank Loss and Nodes of Presence`, `CTMT Geomagnetic Falsification Test Using IGRF Data`
- **Pre-rigorous notes** — `CTMT Axial Geometry and Visible-Band Null Transport`, `CTMT Axis and Hessian Boundary Constant α` / `α2`, `CTMT Complete Boundary Between Coherence and Physics`, `CTMT Full Elemental Computation`, `CTMT Gauge Group Uniqueness`, `CTMT Stationary Phase, Coherence Dynamics, and Admissibility`, `CTMT Trigonometry`, `The CTMT Calculus`, `Two Minimal Falsification Attempts for CTMT` (+ `.zip`)
- **Assets / misc** — `BCEI.figure.png`, `CoherenceHealthMonitor.py`, `MagnetismChaosTest.zip`, `The CTMT Covariance.pdf` (early draft, superseded by *Covariance geometry*)

---

## Status, citation, license

- **Local morphism classification:** stabilized.
- **Global transport geometry & physical interpretation:** open.

DOI: [10.5281/zenodo.18229539](https://doi.org/10.5281/zenodo.18229539) · OSF: [`10.17605/OSF.IO/RFE8N`](https://osf.io/RFE8N/)

Author: **Matěj Rada** · Licensed **CC BY-NC-ND 4.0**.

Serious questions, counterexamples, and attempts to break the theorems are welcome — the framework is designed to be falsifiable, and a clean refutation of any proved statement is a contribution.
