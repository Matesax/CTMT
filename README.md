# Resolution Geometry (RG)

*formerly CTMT*

**Resolution Geometry is the geometry of what a declared family of observation protocols can distinguish.**

Its primitive is the equality-of-law relation of a joint observation law. That relation defines a kernel groupoid and an observational quotient. The stabilizer of the law, taken modulo its gauge part, defines an effective symmetry group that acts on the quotient.

On regular strata, every structure computed from laws descends to the quotient and is invariant under that group. Differential, statistical, causal, temporal and metric structure are then added in a fixed order. Each addition is either forced by the preceding objects, imported from established mathematics, or conditioned on a named hypothesis. The singular loci are joined by orbit-type strata, slices and tangent cones.

The statement of record is **`Resolution Geometry.pdf`**, the Theorem Ladder. It gives the complete dependency-ordered route, rungs L0–L60, from equality of laws to calibrated physical distance, with proofs, explicit boundaries and open problems.

---

## The object

An observation system consists of a latent space $X$, protocols $F_i : X \to Y_i$ valued in spaces of observable laws, and a declared relabeling group $\Psi$. The joint law is

$$
F = (F_i)_{i \in I} : X \longrightarrow Y = \prod_i Y_i .
$$

Its kernel groupoid and quotient are

$$
\mathcal K = X \times_Y X, \qquad \pi : X \longrightarrow Q = X / \mathcal K \cong \mathrm{im}\, F .
$$

The stabilizer of the law, its gauge part, and the effective symmetry group fit into one exact sequence:

$$
1 \longrightarrow \mathrm{Gau} \longrightarrow \mathrm{Stab} \longrightarrow \Gamma \longrightarrow 1 .
$$

A **natural module** is a pullback of a relabeling-invariant structure $\mu$ on law space:

$$
M = F^{*} \mu .
$$

Observable invariants, clock readings, record counts, the Fisher tensor and law-defined cones are all natural. On a regular stratum every natural module descends and is $\Gamma$-invariant:

$$
M = \pi^{*} \bar M
$$

$$
\gamma^{*} \bar M = \bar M \quad \text{for all } \gamma \in \Gamma .
$$

Consequently the following are theorems about natural modules, not separate hypotheses:

- representative independence;
- basicness with respect to the kernel;
- projectability of law-defined transports and cones.

What remains empirical is:

- the content of $\Gamma$, tested by equality of laws;
- regularity and saturation of the chosen stratum;
- latent (non-natural) transport;
- anchors and held-out validation.

The logical order of the construction is

$$
F \longrightarrow \mathcal K \longrightarrow Q \longrightarrow \Gamma \longrightarrow g_F \longrightarrow \text{cone} \longrightarrow \text{scale} \longrightarrow \text{duration} \longrightarrow \text{physical distance} .
$$

---

## The ladder at a glance

| Layer | Rungs | Main results |
|---|---|---|
| Exact | L0–L5 | kernel groupoid; universal factorization; joint kernel is the intersection of protocol kernels; garbling enlarges the kernel; the exact layer fixes no metric, order, cone, scale or unit |
| Symmetry | L6–L11 | exact sequence with faithful action of Gamma on Q; non-splitting; count rigidity; orbit–quotient comparison; saturation hierarchy; coherent completeness |
| Differential | L12–L16 | tangent sequence on regular strata; natural descent; automatic basicness and projectability; slices and tangent cones at walls |
| Second order | L17–L24 | Fisher is natural and its radical is the observational vertical space; regular reconstruction; Fisher does not determine covariance; second-order quotient dimension p+q+pq; nuisance conventions and nuisance-rank walls; data processing; kernel order is not Blackwell order |
| Ignorance | L25–L27 | persistent null and finite attainment; sensor, precision and representation nulls; finite-sample boundary reconstruction |
| Statistical distance | L28–L32 | RG distance is representative-free, chart-free and dimensionless; monotone under garbling; no angle in rank one; finite divergences are not rulers; RG distance is experiment-relative |
| Order and cone | L33–L37 | reachability descends; projective typing; realization branches; quadratic cone gives a conformal ray; Lorentz isotropy in Gamma forces the light cone |
| Scale and transport | L38–L42 | metric representative exists iff the scale cocycle is exact; strict transport fixes the conformal factor; induced connection on regular strata; entropy cannot select a connection |
| Duration | L43–L46 | clocks with equal kernels form a transition groupoid; a stabilizer time flow makes them affine; path independence is a kernel inclusion; coherence duration and BCH memory |
| Physical distance | L47–L56 | no internal unit; counted channels remove dilation; two-point invariant; metric proportionality; SI radar anchor; radar versus rod distance; media; identification theorem |
| Assembly | L57 | kernel–stabilizer theorem and compatibility equations |
| Open | L58–L60 | canonical connection for latent transport; residual characteristic branch; non-static, quantum, non-dominated and infinite-dimensional extensions |

Each rung has exactly one status: forced, imported, hypothesis-conditioned (the hypothesis is named), $\Gamma$-conditional, no-go, or open. Every dependency points to an earlier rung.

---

## How to read the corpus

1. **`Resolution Geometry.pdf`** — the Theorem Ladder. It is self-contained and is the statement of record.
2. **`RG - Made Visible.pdf`** — the geometric picture: a Fisher base with stratified angular and conical fibres.
3. **Module papers**, listed by ladder layer below. They contain extended proofs, worked examples and hostile batteries for individual layers.
4. **Applications and real-data studies.**
5. **Historic syntheses**, only when a derivation has not been restated in current vocabulary.

**Statement-of-record rule.** Where a module paper and `Resolution Geometry.pdf` differ in scope, hypotheses or status, the ladder governs. Module papers are kept unchanged so that every correction remains auditable.

**Notation.** In the ladder, $\Gamma$ denotes the effective symmetry group $\mathrm{Stab}/\mathrm{Gau}$. The earlier characterization papers use $\Gamma$ for the admissibility protocol $(G,\tau,N,S,C,T)$. In this README that protocol is written $\Pi$.

**Citing results.** Cite the ladder rung, for example "RG, L51", rather than the module paper in which the result first appeared.

---

## Module papers by ladder layer

A same-stem ZIP next to a paper (for example `RG - Physical Distance.pdf` and `RG - Physical Distance.zip`) contains its runnable hostile battery. Each battery registers naive claims and sharpened claims, uses fixed seeds, fails on non-finite evidence, and includes a mutation (canary) mode where available. Batteries are falsification witnesses, not premises of any proof.

| Layer | Rungs | Papers |
|---|---|---|
| Foundation | all | `Resolution Geometry.pdf` |
| Exact root | L1–L5 | `RG - Kernel pair root and limits.pdf`; `RG - Quotients.pdf`; `RG - Observation Laws.pdf`; `RG - Observation Laws - Influence.pdf` |
| Symmetry and assembly | L6–L11, L57 | `Assembly of Resolution Geometry.pdf`; `RG - Assembly at the Kernel.pdf`; `RG - Functorial Resolution Geometry.pdf`; `RG - Saturation.pdf` |
| Differential and strata | L12–L16 | `RG - Made Visible.pdf`; `RG - Atlas Globalization.pdf`; `RG - Gluing Test.pdf` |
| Second order and characterization | L17–L24 | `RG - Axioms.pdf`; `RG - Fibrewise Characterization.pdf`; `RG - Fibrewise Characterization - QM.pdf`; `RG - Admissibility Protocol Characterization.pdf`; `RG - Reconstruction Identifiability.pdf`; `RG - Covariance Geometry.pdf`; `RG - Blind Sector.pdf` |
| Ignorance and observation limit | L25–L27 | `RG - Ignorance.pdf`; `RG - Observation Limit.pdf`; `RG - Null Resolution Depth.pdf`; `RG - Entropy.pdf` |
| Statistical distance | L28–L32 | `RG - Distance and Angles.pdf` |
| Order and cone | L33–L37 | `RG - Cone.pdf`; `RG - Cone - Closure.pdf`; `RG - GR Signature Emergence.pdf` |
| Scale and transport | L38–L42 | `RG - Atlas Globalization.pdf`; `RG - Transport Invariants.pdf`; `RG - Canonical Connection.pdf` |
| Duration | L43–L46 | `RG - Time.pdf`; `RG - Operational Duration.pdf`; `RG - Operational Duration Boundary.pdf`; `RG - GR Time.pdf` |
| Physical distance | L47–L56 | `RG - Physical Distance.pdf`; `RG - Wall-Tap Delay.pdf` |

---

## Characterization modules

Once an admissibility protocol $\Pi$ is declared, the regular finite classical RG object is unique up to protocol-preserving natural isomorphism and metric normalization. This is the fibrewise characterization of `RG - Fibrewise Characterization.pdf` and `RG - Axioms.pdf`. Its modules are:

**Quotient.** Extensionality forces unique factorization through $Q$.

**Metric.** Fisher–Rao is selected by Markov invariance on regular finite models, up to a positive constant. Quotient logic alone does not select a metric.

**Resolved selector.** The unique hard projector satisfying metric self-adjointness, idempotence, information compatibility and threshold consistency is

$$
P_\tau = \mathbf{1}_{(\tau,\infty)}\left(G^{-1} F\right).
$$

**Admissible sector.** It is the maximal gate-admissible resolved subobject:

$$
W_{\mathrm{obs}|\mathrm{adm}} = \max \mathrm{Adm}_{\Pi}(R_\tau).
$$

**Transport.** On a regular stratum with constant-rank admissible tangent sector and projector $P$:

$$
\nabla^{W} = P \, \nabla^{\mathrm{LC}} .
$$

**Entropy.** For a deterministic quotient,

$$
H(X) = H(Q) + H(X \mid Q) .
$$

The equality is invariant under fibre automorphisms, so entropy cannot select a connection.

**Quantum.** In finite-dimensional quantum experiments, the quotient, projector, sector and transport layers carry over unchanged once a Petz monotone metric is declared as the metric module. The kernel–stabilizer assembly and the physical-distance identification are not yet extended to quantum protocols (L60).

The experiment determines what can be distinguished. The protocol determines what counts as admissible. RG does not derive $\Pi$ from the experiment.

---

## Physics interface

**Physical distance.** RG distance is dimensionless and experiment-relative; no function of it alone is a physical distance. Suppose the following are established on a regular static slice:

- Euclidean motions are in $\Gamma$, with irreducible isotropy;
- a counted two-way timing channel with exact SI constants $c$ and $\Delta\nu_{\mathrm{Cs}}$ is part of the joint law;
- the medium is resolved;
- a held-out anchor is predicted.

Then every natural metric satisfies

$$
\bar g_{\mathcal E} = \lambda_{\mathcal E}^{2}\, h, \qquad d_{\mathrm{phys}} = \frac{d_{RG}^{\mathcal E}}{\lambda_{\mathcal E}}, \qquad d_{\mathrm{rad}} = \frac{c}{2\,\Delta\nu_{\mathrm{Cs}}}\, n_{\mathrm{Cs}} ,
$$

with $\lambda_{\mathcal E}$ constant, and $h$ the unique invariant metric whose distance equals the radar distance. The unit enters only through the counted channel and the SI definition; it is a declared anchor, not a derived quantity. (L47–L56)

**Duration.** Clock readouts with equal kernels are charts of one duration object with forced cocycle closure. A time flow in the stabilizer makes the transitions affine. Path-independent accumulation holds exactly when the kernel of the declared increment is contained in the kernel of the clock. (L43–L46)

**Cone.** A regular quadratic cone determines a conformal ray, not a metric. If $\Gamma$ contains the standard Lorentz group in dimension at least 2+1, the quadratic branch is forced. Otherwise the Lorentz–Finsler, multicone, stratified and residual branches remain. (L33–L37)

**GR placement and automation.** Observable sectors are placed against GR-style data without identifying RG with spacetime. The bounded automation packages are listed below. They are constructive bridges, not a general solver.

| Layer | File |
|---|---|
| Placement | `RG - GR Placement Bridge.pdf` |
| Fisher layer | `RG - GR Placement Bridge - F-layer.pdf` |
| Gauge-aware geometry | `RG - GR Placement Bridge - Gauge-Aware Fisher Geometry.pdf` |
| Gauge-aware observation | `RG - GR Placement Bridge - Gauge-Aware Wobs.pdf` |
| Real data | `RG - GR Placement Bridge - Gauge-Aware Wobs H1-L1.pdf`; `RG - GR Placement Bridge - Gauge-Aware Wobs ECG.pdf` |
| Physical direction | `RG - GR Placement Bridge - Physics Direction.pdf` |
| Automation | `RG - GR Placement Bridge - Automation.pdf` |
| Signature | `RG - GR Signature Emergence.pdf` |
| Blind scalar sector | `Automation of General Relativity - Blind Scalar Sector.pdf` |
| Fisher holes | `Automation of General Relativity - Fisher Holes.pdf` |
| Source-side action | `Automation of General Relativity - Source-Side Action.pdf` |

---

## Real-data anchors

These studies test individual layers. They are not proofs of universality.

- **OMNI space weather:** predictive resolved–null coupling, lag dependence, condition-dependent frame rotation; no nonzero net winding claimed.
- **USGS seismic catalogue:** coupling signal; honest negative for smooth-loop holonomy under the tested protocol.
- **IGRF geomagnetic models:** resolution-hole diagnostics recover the expected instability toward poorly resolved harmonic degrees.
- **H1–L1 gravitational-wave data:** gauge-aware observable-sector and degeneracy placement.
- **ECG data:** gauge-aware sector construction in a distinct signal domain.
- **Optical measurement systems:** admissible observable sectors; see `RG - Admissible Observable Sectors in Optical Measurement Systems.pdf`.

---

## Novelty calibration

Every RG component reduces to established mathematics:

- equality-of-law quotients and kernel groupoids;
- Fisher–Rao geometry and Čencov uniqueness;
- spectral projectors and Schur complements;
- canonical correlations and PSD cones;
- Blackwell and Le Cam comparison;
- stratified orbit spaces of proper groupoids;
- Cauchy–Hölder ratio scales and Schur's lemma;
- Beckman–Quarles rigidity;
- Malament–Hawking conformal reconstruction;
- the SI definitions.

RG claims no new primitive and no new branch of mathematics. Its contribution is threefold:

- a protocol-explicit, dependency-typed assembly of these components into one geometry of partial observability;
- the kernel–stabilizer mechanism, which makes the modules share one symmetry group;
- explicit failure conditions for every step, backed by hostile batteries.

See `RG - Elimination.pdf` and `RG - Elimination - Lock Conclusion.pdf`.

---

## Scope and non-claims

1. **No geometry of latent reality.** RG describes distinctions supported by declared protocols. Latent representatives exist; observation does not select them.
2. **No protocol-free uniqueness.** The quotient is universal. The admissibility protocol, relabeling group and nuisance conventions are declared.
3. **No unrestricted Fisher claim.** Fisher–Rao is selected in the regular finite classical domain, up to normalization. Singular, infinite-dimensional, non-dominated and quantum experiments need separate modules.
4. **No Fisher = spacetime identity.** On a certified static slice the Fisher metric is *proportional* to the physical spatial metric, with an experiment-dependent constant. This is an identification theorem under named hypotheses, not an identity.
5. **No internal unit.** Statistics, flatness and thermodynamic length supply no metre. The unit is the SI anchor entering through a counted channel.
6. **No automatic Lorentzian structure.** A cone is Lorentzian only under the quadratic gate or sufficient isotropy in $\Gamma$.
7. **No field equations.** No Einstein equation, matter dynamics or quantum measurement law is derived.
8. **No physical coupling by default.** Resolved–null correlation may come from dynamics, preparation, nuisance or instrumentation. Attribution requires intervention.
9. **No theorem from batteries.** Theorem status comes from stated hypotheses and proofs.

---

## Problem status

### Closed

| Problem | Resolution | Rungs |
|---|---|---|
| Observable domain | equality-of-law quotient with universal factorization | L1–L3 |
| Coupling of protocols | joint kernel is the intersection of protocol kernels | L3 |
| Source of shared symmetry | exact sequence Gau → Stab → Γ; faithful action on Q | L6–L7 |
| Basicness and projectability of law-defined modules | automatic by natural descent | L13–L14 |
| Exact versus infinitesimal ignorance | kernel groupoid versus radical of Fisher | L17, L25–L26 |
| Processing and ignorance | garbling enlarges the kernel and contracts Fisher | L4, L23, L29 |
| Completeness of second-order coordinates | quotient dimension p+q+pq; spectra and canonical correlations are insufficient | L20 |
| Hard resolved projector | unique threshold spectral projector for fixed (F,G,τ) | fibrewise characterization |
| Final admissible sector | maximal gate-admissible resolved subobject | fibrewise characterization |
| Regular induced connection | induced from Levi-Civita by the sector projector | L40 |
| Entropy as connection selector | negative: entropy is invariant under fibre automorphisms | L41 |
| Quadratic cone | forced by Lorentz isotropy in Γ in dimension at least 2+1 | L37 |
| Metric representative | exists iff the scale cocycle is exact | L38–L39 |
| Operational duration | clock transition groupoid; affine under a stabilizer time flow | L43–L45 |
| Noiseless finite-history dynamics | regular away from kneading walls; topological entropy recovered from kernel-count growth | L46.1–L46.3 |
| Dynamical certification | typed certificates combining module, escalation window and exact-return tests | L46.4 |
| Chaotic characteristic transport | globally conic under uniform hyperbolicity; otherwise typed only per certified window | L46.5, L59.10 |
| Unit | no internal unit; counted channel plus SI anchor | L47–L48, L53 |
| Physical distance | identification theorem under named hypotheses | L56 |
| Cooperation of modules | kernel–stabilizer theorem and compatibility equations | L57 |
| Specialization across protocol and latent walls | kernel upper semicontinuity; unique specialization under kernel inclusion; functorial composition | L16.1–L16.5 |
| Stratified observable category | regular transitions and specialization arrows form a typed category; holonomy is defined within strata | L16.5 |
| Symmetry across walls | regular symmetries descend exactly through the normalizer of the wall kernel | L16.3 |
| Distinguishability across walls | Hellinger distance, Fisher modules, Bayes risks and the Cramér–Rao bound specialize under the declared convergence hypotheses | L24.1 |
| Finite-data rank-wall inference | no finite sample certifies an exact rank wall; only full-rank or upper-bound certificates are admissible | L27.1 |
| Residual characteristic branch | reduced to tame, hyperbolically locked and wild classes with explicit resolution invariants and window certificates | L59.1–L59.10 |
| Tame residual geometry | finite Whitney-stratified type with finitely many sheets, integer dimensions, signatures, incidences and sheet monodromy | L59.4 |
| Limits of residual classification | no finite linear classification even for tame cones; wild sets admit no finite combinatorial classification | L59.5, L59.9 |
| Lorentz–Finsler admissibility | smoothness and strict convexity must hold simultaneously for the primal and dual cones | L59.6 |
| Hyperbolic characteristic lock | forward cone is convex; sheets are real, ordered and nested; only contact and Z₂ polarization holonomy remain | L59.7 |
| Residual ill-posedness | failure of hyperbolicity forces complex characteristic roots and unbounded high-frequency amplification | L59.8 |
| Quantum metric family gate | CPTP monotonicity does not select a unique quantum metric; monotonicity is a family gate, not a selector | Q1 |
| Quantum task-relative metric selection | declared operational tasks select SLD, BKM, WY, etc.; metric selection is protocol-relative | Q2–Q5 |
| Quantum wall re-discrimination | wall geometry is reconstructed by re-discrimination with outcomes LOCKED, DIVERGENT, VACUOUS, UNDERDETERMINED or INCONSISTENT | Q5 |
| No law-natural connection | laws, Fisher geometry, entropy, specialization and metric selection do not determine a connection | T1 |
| Canonical latent transport under declared covariance | with declared covariance/memory kernel, canonical transport is uniquely determined and coherent across walls | T2–T5 |
| Observable global atlas closure | transport within strata is the transition groupoid; across walls, maps under kernel inclusion and correspondences otherwise | L16.5, T6 |
| Metric continuation versus state transport | wall continuation is obtained by specialization and re-discrimination of the observation law, not by transporting metric components | Q5, T6 |
| Canonical wall object | fibre-product correspondence / specialization relation, not a globally defined wall transport map | T6 |

### Open

1. **Transport from observation alone.** The observable atlas is closed, but determine when a declared observation protocol is sufficient to reduce the wall correspondence to a unique transport law. Without additional transport data, latent transport remains observationally underdetermined.
2. **Enumeration of tame characteristic types.** Enumerate the tame types of degree-\(d\) characteristic cones in dimension \(1+n\). The residual branch already supplies the correct finite type, but complete enumeration remains an external problem in real algebraic geometry.
3. **Extensions** (L60):
   - non-static slices where radar and rod distance differ;
   - contextual quantum protocols beyond a fixed metric module;
   - non-dominated, infinite-dimensional, field-valued and path-space experiments.
4. **Non-generic strata.** Reconstruction at repeated eigenvalues, enlarged stabilizers and rank jumps; complete stabilizer classification and compatibility with specialization.
5. **Protocol characterization.** Nuisance closure, threshold selection, escalation rules, stability margins and quantum task declarations are conditionally characterized, but not derived internally.
6. **Observation to physical tensor.** Construct a sector map to an observable stress tensor with uniqueness, conservation, gauge-independence and boundary conditions while preserving the distinction between *not identifiable* and *identified as zero*.
7. **Physical versus protocol coupling.** Separate physical coupling from coupling introduced by preprocessing, nuisance structure, protocol design or task selection.
8. **Higher-order geometry.** Observable curvature, normal holonomy and second-fundamental-form behaviour near rank, threshold, isotropy and characteristic discriminants.
9. **Flat classification.** Explicit enumeration of flat observable classes by character data and specialization behaviour.
10. **Entropy beyond discrete fibres.** Continuous, relative, path-space and non-equilibrium versions of kernel-count and coherence entropy.
11. **Nonuniform dynamical transport.** Replace window-typed characteristic certificates by a global object, or prove that no such object exists, for nonuniformly hyperbolic and chaotic systems.
12. **Wild-class asymptotics.** Determine which resolution invariants beyond count exponent, box dimension and Cantor–Bendixson rank are operationally stable and composable.
13. **Benchmarks.** Maintain a preregistered fixed-gate benchmark suite with held-out datasets, deliberate failures, escalation canaries and cross-instrument replication.
14. **Operational transport characterization.** Find necessary and sufficient observational conditions under which the wall correspondence collapses to a unique transport, or prove that correspondence-valued transport is the maximal observable object.

---

## Honesty ledger

The surviving framework depends on keeping negative results visible.

- **Kolmogorov turbulence from Fisher-rank loss:** retired. The conservation step and exponent closure failed.
- **Constants or π-factors from flexible kernels:** retired as reparametrization.
- **Emergent spacetime, gravity from Fisher geometry, nodes of presence, quantum or biological identifications:** not supported; not part of RG.
- **Circular Omori validation:** retained only as a bounded negative result.
- **Universal nonzero holonomy:** not established; several real-data tests correctly reject loop structure.
- **RG distance as physical distance:** refuted (L31–L32). Physical distance requires the identification theorem (L56).
- **Independent module gates:** superseded. Basicness and projectability of law-defined modules are theorems (L13–L14).
- **Earlier synthesis papers as foundation:** superseded by `Resolution Geometry.pdf`.

Correction records remain in the repository, including `Correction and Maturation of the CTMT Redshift Claim.pdf` (+ ZIP).

---

## Repository policy

The repository preserves the full development record: foundations, module papers, numerical attacks, corrections, superseded formulations and retired claims. Older files are not deleted or silently rewritten. Their presence makes corrections auditable; it does not make historical statements current claims.

| Label | Meaning |
|---|---|
| `[foundation]` | statement of record: `Resolution Geometry.pdf` |
| `[module]` | extended proofs and batteries for one ladder layer; governed by the foundation |
| `[supported]` | constructive bridge, implementation, or real-data demonstration |
| `[historic]` | superseded presentation retained as development record |
| `[retired]` | withdrawn claim retained so the correction is visible |

**Historic syntheses** `[historic]`:

- `RG - Theorem Ladder.pdf` (the first ladder);
- `RG - Complete Framework.pdf`, `RG - Synthesis.pdf`, `RG - Fundamental Theorem.pdf`, `RG - Locked Foundation.pdf`, `RG - Atlas.pdf`, `Resolution Geometry of Observation Systems.pdf`;
- `The CTMT - Testament of 22 years.pdf`.

**Falsification and necessity studies** `[supported]`:

- `RG - Necessity.pdf`, `RG - OMNI Necessity.pdf`, `RG - Seismic Necessity.pdf`;
- `RG - Hole Rejection.pdf`, `RG - Undermine Attacks.pdf`, `RG - Undermine Attacks Improved.pdf`, `RG - Final Chaotic Test.pdf`;
- `RG - CHI Reduction.pdf`, `RG - Elemental Characterization.pdf`.

**CTMT-era results still used as support**:

- `Independent-Protocol Recovery of Resolved Null Coupling.pdf`;
- `The CTMT Compatibility Lock and Holonomy Obstruction.pdf`;
- `The CTMT Resolved Null Covariance Coupling.pdf`;
- `CTMT Full Elemental Computation.pdf`.

**Origins and manifest:** `RG - Origins.pdf`, `RG - Manifest.pdf`.

### Historic / pre-rigorous / retired (quarantined)

Preserved for intellectual history; not part of current claims.

- **Chronotopic Theory of Matter and Time:** I, II, III, IV, CHI, Causality, Seepage.
- **Chronotopic Metric Theory:** original overview, physics and trigonometry papers.
- **Retired physics attempts:** universal causal energy transport, Newton-G boundary, radiative constants, emergent time and signature interpretations, nodes of presence, early geomagnetic physical claims.
- **Pre-rigorous notes:** axial geometry, Hessian boundary constants, visible-band null transport, elemental computation, early gauge uniqueness, stationary phase, calculus, minimal falsification attempts.
- **Assets and utilities:** site files, fonts, images, scripts, JSON outputs, standalone battery archives.

---

## Origins

RG began as a coherence project: an attempt to force structure on whatever holds physical description together. CTMT was the first forced model. CTMT-Metric was the second, obtained by falsifying CTMT with Fisher geometry. RG is the third: the part that survived.

---

## Citation and license

DOI: [10.5281/zenodo.21297385](https://doi.org/10.5281/zenodo.21297385)
Author: **Matěj Rada**
License: **CC BY-NC-ND 4.0**

Historic CTMT — DOI: [10.5281/zenodo.18229539](https://doi.org/10.5281/zenodo.18229539) · OSF: [10.17605/OSF.IO/RFE8N](https://osf.io/RFE8N/)

Counterexamples and attempts to break the theorems are welcome. A clean failure under the stated hypotheses is a contribution.
