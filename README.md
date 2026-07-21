# Resolution Geometry (RG) — *formerly CTMT*

**One sentence.** Given a forward model's Jacobian and an observation covariance, Resolution Geometry is a small, rigorous geometry that says *which directions are resolvable from measurements, how the unresolvable ones still matter through correlation, and which transformations preserve that structure.*

> **Naming.** The mature core is **Resolution Geometry (RG)**, stripped of physics vocabulary. Files prefixed **`RG -`** are the current, CTMT-free corpus. Files prefixed **`The CTMT …` / `CTMT …`** are the original, detailed versions of the same results (still valid; older vocabulary). Everything under [Historic](#historic--pre-rigorous--retired-quarantined) is pre-rigorous or retired and is kept for intellectual history only.

**Status after elimination.** RG has been deliberately attacked and *eliminated piece by piece*. Assuming a reader who knows all of modern mathematics, every individual RG object reduces to an established one — image/cokernel, orthogonal complement, connection/groupoid, Schur complement, canonical correlation, second fundamental form, monodromy/descent. **No RG object is a new primitive**, and any claim to a novel invariant unknown to existing mathematics is refuted in writing. What survives is not any single object but the **lock** between them: the *proven* mutual agreement of several established fields (linear algebra, information geometry, submanifold geometry, gauge/globalization) that **no single field reproduces alone**. RG's status now rests on that coherence — formalized as a nonzero descent class. See [The elimination result](#the-elimination-result) and [Scope and non-claims](#scope-and-non-claims).

It is an **observational / estimation geometry**, built on standard linear algebra, information geometry, and inverse-problem theory. It is **not** a theory of physics, and it makes no claim about the fundamental nature of space, time, or matter.

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

## Real-world necessity — tested on world data, not toys

**One sentence.** Beyond the synthetic constructions, RG's two operational claims — that the coherent geometry is *necessary* (some questions cannot be answered by independent probes), and that its hole geometry is *safe* (it refuses fits the data cannot support) — were tested on large, public, real-world datasets, using permutation nulls, out-of-sample validation, cross-year replication, and an independent second model generation.

**Why these are the right tests.** RG does not replace independent measurements; it is a **certificate** on top of them. It matters when a field *(i)* infers an **unobserved** sector from an observed one (coupling `C`), *(ii)* **combines** measurements across changing conditions (frame transport), *(iii)* **compares** across setups (gauge invariants), or *(iv)* must know **when not to trust a fit** (the resolved/null "hole"). When none of these applies, RG returns ≈ 0 and certifies that plain probes suffice — the self-limiting property that makes it a diagnostic, not an ornament. See [Scope and non-claims](#scope-and-non-claims).

**What real data showed.**

*Space weather — NASA **OMNI** solar-wind → magnetosphere, 5-minute cadence, 2022–2024
(~282,000 complete samples).*
- **The coupling is real and predictive.** Out of sample, the drivers nowcast the response at
  R² ≈ 0.53 (AE), 0.47 (AL), 0.42 (SYM-H), and the coupling **strengthens** at the true
  ~30–60 min physical lag — the signature of a genuine coupling, not a fit.
- **The coupling geometry depends on external condition.** Binned by season and by the 27-day
  solar rotation, it differs from a permutation null far beyond chance (*p* = 0.002), varying
  smoothly around the cycle (adjacent-bin overlap 0.987).
- **Pre-registered Russell–McPherron test (positive, replicated).** In the (Bᵧ, B_z)_GSM plane,
  the geoeffective coupling frame rotates **47° peak-to-peak** through the year and **replicates
  across all three independent years** (*r* = 0.56; permutation *p* = 0.001), cleanest for the
  reconnection-driven AL index.

> **The honest ceiling.** The net winding of that seasonal loop is ≈ 0: the frame rotates and returns. What real data establishes is a **reproducible, condition-locked rotation of the coupling frame**, not a nonzero topological monodromy. Still decisive for necessity — the geoeffective frame is up to 47° apart between seasons, so any fixed-frame coupling analysis is wrong by tens of degrees — but the stronger (loop-fails-to-close) claim was **not** found and is not asserted.

*Seismology — USGS significant-earthquake catalogue (a hard case, kept on purpose).* The coupling was present (felt-impact recoverable from physical source, out-of-sample R² up to
0.54), but the cross-condition holonomy returned an **honest negative**: too sparse, regional structure too abrupt to form a smooth loop, and RG's permutation null **correctly declined to report a holonomy**. A framework that only ever confirms itself is not credible; the negative is kept because it shows RG does not manufacture structure.

*Geomagnetic field — **IGRF** reference-field coefficients (estimation-safety test).* This tests
the *hole*, not the pillars, on the problem where resolution analysis was born (Backus–Gilbert,
1968). Two successive model generations (IGRF-13 vs IGRF-14) act as independent estimates; their
disagreement is an empirical measure of the hole.
- **The hole is real and grows with degree.** Relative model-revision error rises ~**250×** from
  the dipole (6×10⁻⁵) to spherical-harmonic degree 13 (5×10⁻²): low degrees pinned, high degrees
  in the hole.
- **The health-bar flags a genuinely dangerous real operation.** Downward-continuing the field to
  the core–mantle boundary amplifies exactly the least-resolved degrees; the health-bar drops
  from **1.00 (surface)** to **0.84 (CMB)**, with **16% of the field power pushed into the
  unresolved hole** — reproducing, from a domain-agnostic resolved/hole power budget, the fact
  geomagnetists know independently (core-surface maps degrade toward the truncation degree).

> **Honest scope.** IGRF is a slow, non-cyclic, ~26-epoch series; it does **not** rescue the transport-holonomy pillar — there is no loop to close. Its role is different and real: a physical, historically apt confirmation that the hole-rejection health-bar recovers known resolution structure on real data.

**What this proves, and what it does not.** On real, public, physically independent data the RG phenomena are real, not artifacts of synthetic construction: the coupling exists and predicts the unobserved sector out of sample; the coupling geometry is a reproducible function of external conditions that a fixed-frame analysis cannot capture; and the covariance/hole geometry **rejects dangerous fits**, reproducing the known resolution structure of the geomagnetic field and flagging its classic dangerous operation. It does **not** prove a nonzero topological holonomy in nature (bounded to ≈ 0 net winding), and it makes **no** physical claim — no earthquake prediction, no space-weather forecast beyond a nowcasting baseline, no new geomagnetic result. These datasets are neutral test objects for an estimation geometry.

> **Reproducibility.** Full analyses, permutation nulls, bootstraps, and runnable code:
> `RG - OMNI Necessity`, `RG - Seismic Necessity`, `RG - Necessity` (synthetic argument),
> `RG - Undermine Attacks` (adversarial battery), and `RG - Hole Rejection` (with the IGRF real-data anchor). Each ships the `numpy`/`scipy` script that reproduces its numbers and states its own non-claims.

---

## CHI — the fast/cheap RG estimator for first real-world use

**One sentence.** `CHI` is the **rank-1 special case of RG** packaged as a throwaway-cheap estimator: from *one anchor measurement* it predicts a system's throughput / power / consumption across its operating range as a dimensional power law, and — unlike an ordinary quick fit — it ships a **validity certificate** that says where that single-anchor estimate can be trusted and flags the moment the system leaves that regime.

**What it is, mechanically.** Under the _RG reduction of CHI_, the coherence volume is the coordinate along the single *resolved direction* of the log-observation geometry:

```
log Y = log k + a · log θ          (a Buckingham–Pi monomial; a = resolved exponents)
```

Learn (or assume) the exponents `a`, fix the prefactor `k` from one measurement, predict everywhere. It is pure log-linear algebra — `O(n·d²)`, runs on a laptop, scales to large operating logs. `RG_CHI_estimator.py` is the reference implementation.

**Why it is the right *first* tool on a real-world system.** It costs almost nothing and answers the two questions you actually have on day one:

- *What scaling do I have, and what happens at another operating point?* → exponents + one-anchor prediction, carrying physical units, with no fitting library and no domain-specific correlation required.
- *Where can I believe this before spending on CFD/FEM/test rigs or a data campaign?* → the certificate returns the **validated coherence class** and **flags** any new point that has left it.

**How it differs from other numeric-first estimators.**

|                    | typical quick estimator            | **RG-CHI**                                   |
|--------------------|------------------------------------|----------------------------------------------|
| data needed        | many points / a fit set            | **one anchor** (+ known or once-learned `a`) |
| output             | a number (± noise band)            | scaling law + number + **validity boundary** |
| extrapolation      | silent, often confidently wrong    | **flagged**: detects the *law itself* changing |
| failure mode       | hidden until checked against truth | **surfaced**: resolved-direction drift is measured |
| interpretability   | curve-fit / black box              | explicit dimensional exponents               |

> **The one real difference.** Ordinary error bars report the scatter *inside* the fit.
> RG-CHI's certificate reports when the **model form** has changed — the resolved direction drifting or rotating (a regime break, the old "seepage") — which is exactly the silent-extrapolation failure that ruins naive estimators. It is *self-limiting*: it would rather say "you have left the coherence class" than return a confident wrong number.

**What it is not.** Not a physics theory, not a replacement for CFD/FEM/circuit simulation, and not more accurate than a full model in-domain. It is a first-order **triage** tool: cheap scaling plus an honest map of where the cheap estimate holds — the fast front end you run *before* committing to the expensive one.

> **Provenance.** The formula and its single-anchor transport predate the rigorous core (see [Historic](#historic--pre-rigorous--retired-quarantined)); what is new is the RG reading that explains *why* it transported (index-locked, flat resolved direction) and the certificate that makes its range checkable. Reduction and worked tests: `RG - CHI
> Reduction`; runnable tool: `RG_CHI_estimator.py`.

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
| 7 | **Elimination / Reduction Analysis** | Meta-level: every RG object reduced into established mathematics; the residual is the multi-field lock | `RG - Elimination.pdf` | — |

> **Superseded duplicate:** `Resolution Geometry - Atlas.pdf` (+ `.zip`) is an earlier Atlas draft; **#3 above supersedes it.**

---

<!-- Note: the paper's worked example (Θ=R³, O=R⁵) is numerically -->
<!-- verified inline (Schur floor, canonical correlations, Fisher = -->
<!-- inverse Schur to 1e-16, Z/2 phase = -1, monodromy). A matching -->
<!-- battery zip `RG - Elimination.zip` can be shipped if desired.  -->

## The elimination result

*(File: `RG - Elimination.pdf`)*

The sharpest test we could pose was not "is RG new?" but **"assume the reader knows all of modern mathematics — can every piece of RG be eliminated?"** The elimination paper runs that test under a fixed rule — the **Elimination Principle**: *replace every RG object by its canonical established equivalent; introduce no new definition until all canonical reductions are exhausted* — across twelve foundations (linear algebra, functional analysis, operator theory, information geometry, differential geometry, Lie groupoids, category theory, homological algebra, sheaves/stacks, higher categories), plus a universality test and an irreducibility test.

**Outcome — piece by piece, RG is eliminated.** Every individual object reduces:

| RG object | eliminated to |
|---|---|
| resolved / unresolved | image / cokernel |
| null sector | orthogonal complement *(requires the metric)* |
| transport | connection / Lie groupoid |
| coupling `C_RN` | off-diagonal block / `Ext¹` class / canonical correlation |
| recoverability | least squares (Gaussian conditional) |
| blind recursion | recursive Schur complement |
| second-order lift | second fundamental form (Gauss–Codazzi–Ricci) |
| globalization | monodromy / descent (`Čech H¹`, character variety) |
| strata | Luna slice quotient |
| reverse reading | transposed Gaussian conditional (same invariant) |
| reparametrization rigidity | column-space invariance |

**No RG object is a new primitive.** This is a genuine, deliberate negative result — and it is the point.

### How the lock works

What is *not* eliminated is the object **as a whole**. The twelve reductions land in **different** fields, and the paper proves there is **no single field that receives them all** — no faithful functor from RG into any one foundation. The reduction table has **no complete column**: each foundation forgets something another keeps —

- covariance/linear algebra forgets the **globalization** (no monodromy);
- information geometry forgets the **covariance blocks** (hence the Schur floor — Čencov fixes the metric, not the blocks);
- differential geometry forgets the **covariance/statistical layer**;
- category theory and homology forget the **metric order** (the Schur floor is an *inequality*, not a morphism).

The surviving content is the **lock** between the fields — a web of *proven* bridges that make the separate realizations **one object**:

- `g = Σ⁻¹`  ⟹  `g_F = Jᵀ Σ⁻¹ J`   *(covariance ↔ information)*
- `g_F = (Σ_R − C Σ_N⁻¹ Cᵀ)⁻¹`   *(Fisher = inverse Schur)*
- `II =` off-diagonal block of the transport connection   *(immersion ↔ gauge, via Gauss–Weingarten)*
- canonical correlations = complete gauge invariant on each stratum   *(covariance ↔ moduli)*

These bridges are **mutually consistent**: first-order Schur admissibility and second-order Gauss–Codazzi–Ricci constrain disjoint data and never over-determine. Formally, the residual is a **nonzero descent class** — RG is a nontrivial *bundle of foundations*, glued by the bridge cocycle, with no global trivialization into any single field. Fittingly, RG's own irreducibility turns out to be a partial-observability phenomenon: **no single foundation observes all of it.**

### Honest calibration

Coherence-across-structures is itself a **known kind** of object: **Kähler** manifolds (Riemannian + complex + symplectic) and **Frobenius** manifolds (metric + product + grading) are already coherent multi-structure objects glued by a compatibility law. So RG does **not** claim a new *kind* of mathematics. What it claims, precisely:

> a **new instance of a known kind** (a coherence object), **in a new subject** — the geometry of partial observability — **spanning an unusually wide foundational range**, with a **partly order-theoretic gluing law** (the Schur floor is an inequality, which is exactly why the categorical and homological attacks cannot reach it).

Less than a new branch of mathematics; more than a repackaging. That is the defensible statement of what RG is, and the elimination paper is what earns it.


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
