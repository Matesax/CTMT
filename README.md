## Resolution Geometry (RG) — *formerly CTMT*

**Resolution Geometry is the protocol-relative geometry of what a declared experiment can distinguish.** Starting from a complete statistical experiment and an explicit admissibility protocol, it forms the universal observational quotient, equips its regular finite classical part with Fisher–Rao distinguishability, selects resolved directions by a characterized spectral projector, and glues admissible local sectors by natural transport.

The mature framework is **Resolution Geometry (RG)**. Files prefixed **`RG -`** are the current corpus. CTMT-prefixed papers preserve earlier derivations and applications, but use older vocabulary. Material listed under [Historic / retired](#historic--pre-rigorous--retired-quarantined) is not part of the current claim.

### What is now characterized

For a declared admissible experiment

$$
\mathcal E=(\Theta,\mathcal O,\{P_\theta\}_{\theta\in\Theta}),
\qquad
\Gamma=(G,\tau,N,S,C,T),
$$

RG has the logical order

$$
\text{declared experiment}
\longrightarrow Q_{\mathcal E}
\longrightarrow g_F
\longrightarrow P_\tau
\longrightarrow W_{\mathrm{obs}\mid\mathrm{adm}}
\longrightarrow \{T_{ij}\}.
$$

Here:

- $Q_{\mathcal E}=\Theta/{\sim_{\mathcal E}}$, where $\theta\sim_{\mathcal E}\theta'$ iff $P_\theta=P_{\theta'}$, is the **universal observational quotient**;
- $g_F$ is the **Fisher–Rao metric**, selected—up to normalization—by the classical sufficient-Markov information-geometric module on regular finite statistical models;
- $P_\tau=\mathbf 1_{(\tau,\infty)}(G^{-1}F)$ is the unique hard resolved projector satisfying metric self-adjointness, idempotence, information compatibility, and threshold consistency;
- $W_{\rm obs|adm}\subseteq {\rm Im}(P_\tau)$ is the final sector after nuisance, stability, conditioning, coarse-graining, and transport gates;
- $T_{ij}$ are natural local transports satisfying identity and cocycle laws.

The result is a **modular characterization relative to the declared experiment and protocol**, not a protocol-free derivation of physics. The experiment determines what can be observed; the protocol states what counts as admissible.

### What the final geometry looks like

RG is not globally one smooth manifold. Its natural home is a **Whitney-stratified orbit-space bundle**:

- the smooth resolved/base directions carry Fisher–Rao path geometry, transport, holonomy, and monodromy;
- resolved–null coupling is angular, described by principal angles/canonical correlations and frame-alignment invariants;
- the null fibre is conical: residual covariance and related quadratic forms live in positive-semidefinite cones stratified by rank;
- eigenvalue collisions, threshold crossings, nuisance-rank changes, and conditioning failures form the discriminant walls;
- the Schur admissibility condition cuts out the allowed part of the fibre.

So Fisher–Rao geometry does not disappear, and it is not the whole object. It becomes the smooth horizontal/base geometry of distinguishability, while the full fibre contains closed eigenvalue chambers, angular data, and PSD cones. At degeneracy walls the fibre is singular by construction. This is the geometry made explicit in **`RG - Made Visible.pdf`**.

### Scope in one paragraph

RG is an observational and estimation geometry assembled from established mathematics: statistical experiments and quotients, Fisher information, generalized spectral projectors, Schur complements, canonical correlations, PSD cones, naturality, cocycle gluing, and stratified orbit spaces. No individual ingredient is claimed as a new primitive. The contribution is the explicit, protocol-aware characterization and compatible assembly of these ingredients into one geometry of partial observability. RG does **not** identify the observational quotient with latent reality, derive the admissibility protocol from nothing, or reduce physics to information.

---

## Repository status and preservation policy

This repository preserves the full RG/CTMT development record: current foundations, supporting derivations, numerical attacks, corrections, superseded formulations, failed physical interpretations, and retired claims. Older files are not deleted or silently rewritten. Their presence documents how the mature framework was reached and makes its corrections auditable; it does **not** make every historical statement a current RG claim.

The mature RG corpus now carries the load-bearing work. New readers should begin with **`RG - Axioms.pdf`**, then **`RG - Made Visible.pdf`**, and only then follow the specialized reconstruction, atlas, transport, GR-placement, and automation papers. CTMT-era papers remain useful when a detailed derivation or historical application has not been restated in the newer vocabulary.

Status labels used below mean:

- **`[foundation]`** — current axiomatic or structural entry point.
- **`[proved]`** — mathematical result established within its stated hypotheses.
- **`[supported]`** — constructive bridge, implementation, or numerical/real-data demonstration; not a proof of a broader universal claim.
- **`[proved/open]`** — a proved core with an explicitly unfinished extension or classification problem.
- **`[historic]`** — superseded presentation retained as a supporting synthesis or development record.
- **`[retired]`** — withdrawn claim retained so that the correction is visible.

A same-stem ZIP, where present, contains the associated runnable battery or reproducibility bundle. Batteries test implementations, assumptions, and stronger conjectures; they do not replace proofs.

---

## Canonical reading order

| # | Paper | Status | Role | File | Battery |
|---:|---|---|---|---|:---:|
| 0 | **Axioms / Characterization** | **`[foundation]`** | Universal observational quotient; Fisher module; unique spectral resolved projector; natural transport, gluing, stability, and entropy decomposition | `RG - Axioms.pdf` | ✓ |
| 1 | **Made Visible** | **`[foundation]`** | Final geometric home: Fisher base with stratified angular/conical orbit-space fibres | `RG - Made Visible.pdf` | — |
| 2 | **Foundations** | **`[historic]`** | Earlier local object, sectors, covariance structure, and automorphism rigidity; retained as supporting groundwork | `Foundation.pdf` | — |
| 3 | **Reconstruction / Identifiability** | **`[proved]`** | Reconstruction from observational data up to gauge; Fisher alone is generally insufficient | `RG - Reconstruction Identifiability.pdf` | ✓ |
| 4 | **Atlas / Globalization** | **`[proved]`** | Observable charts, principal automorphism bundle, Čech cocycle, and holonomy | `RG - Atlas Globalization.pdf` | ✓ |
| 5 | **Global Structure** | **`[proved/open]`** | Compact automorphism structure and flat classification by character data; explicit enumeration remains open | `RG - Global Structure.pdf` | proto |
| 6 | **Transport Invariants** | **`[proved]`** | Gauge-invariant content surviving admissible transport | `RG - Transport Invariants.pdf` | — |
| 7 | **Second-Order Moduli** | **`[proved/open]`** | Second fundamental form, normal directions, and bounded higher-order moduli; full observable classification remains open | `RG - Second Order Moduli.pdf` | — |
| 8 | **Elimination / Reduction** | **`[proved]`** | Reduction of individual RG components to established mathematics and calibration of the residual assembly claim | `RG - Elimination.pdf` | — |

### Supporting syntheses and legacy entry points

- **`RG - Complete Framework.pdf`** — **`[historic]`** the previous grand synthesis. It remains useful as a broad map of the pre-axiomatic corpus, curvature spine, and historical integration, but it is no longer the foundation or preferred first paper.
- **`RG - Synthesis.pdf`**, **`RG - What Holds the Machine.pdf`**, and **`The CTMT - Testament of 22 years.pdf`** — **`[historic]`** supporting syntheses in progressively older vocabulary.
- **`Resolution Geometry - Atlas.pdf`** — **`[historic]`** earlier atlas draft; superseded by **`RG - Atlas Globalization.pdf`**.

---

## Axiomatic core

The paper **`RG - Axioms.pdf`** replaces the former synthesis as pillar #0 because it states what is primitive, what is forced, what remains protocol-relative, and what would falsify the construction.

### Operational axioms

1. **Observational extensionality:** experiment-internal conclusions are constant on equality-of-law classes.
2. **Naturality:** observable constructions commute with isomorphisms of complete experiments and covariant protocol transport.
3. **Information order:** parameter-independent garbling cannot increase distinguishability.
4. **Regularity off the discriminant:** continuous structures vary continuously and discrete invariants remain locally constant away from declared transition sets.
5. **Composition and gluing:** local comparison maps preserve identities and satisfy cocycle closure.

### Characterized modules

- **Quotient:** extensionality forces unique factorization through $Q_{\mathcal E}$.
- **Metric:** a separate classical information-geometric module selects Fisher–Rao in its stated regular finite domain. Quotient logic alone does not select a metric.
- **Resolved selector:** the four hard-selector axioms uniquely give the threshold spectral projector.
- **Transport:** naturality gives covariant sector transport; identity and cocycle closure characterize a consistent observable atlas.
- **Entropy:** for a deterministic quotient $Q=\pi(X)$,

  $$
  H(X)=H(Q)+H(X\mid Q).
  $$

  Under conditional equiprobability, the fibre term is the expected Boltzmann multiplicity entropy. Increased resolution redistributes entropy from unresolved fibres to observable classes; it does not create total entropy.

### Exact limitations of the characterization

The axioms do not derive $G$, $\tau$, nuisance conventions, stability classes, a preferred loss, or a physical interpretation without further assumptions. They characterize RG **once the complete experiment and protocol are declared**. A latent point is not an experiment-internal observable object unless the experiment separates it; this does not deny that latent representatives exist.

---

## General-relativity placement and automation

The recent GR sequence materially changes the frontier. The question is no longer whether RG can be placed next to differential geometry in principle; explicit gauge-aware placement and restricted automation bridges now exist.

### Placement ladder

| Layer | Result | File |
|---|---|---|
| Placement | Observable sectors placed against GR-style field/geometric data without identifying RG with spacetime | `RG - GR Placement Bridge.pdf` |
| Fisher layer | Local information/Fisher structure isolated from the physical field geometry | `RG - GR Placement Bridge - F-layer.pdf` |
| Gauge-aware observation | Quotient and nuisance handling incorporated into $W_{\mathrm{obs}}$ | `RG - GR Placement Bridge - Gauge-Aware Wobs.pdf` |
| Real-data demonstrations | Gauge-aware construction exercised on H1–L1 gravitational-wave data and ECG data | `RG - GR Placement Bridge - Gauge-Aware Wobs H1-L1.pdf`; `RG - GR Placement Bridge - Gauge-Aware Wobs ECG.pdf` |
| Physical-direction bridge | Separates identifiable physical directions from gauge and observational degeneracies | `RG - GR Placement Bridge - Physics Direction.pdf` |
| Automation | End-to-end restricted placement/sector automation | `RG - GR Placement Bridge - Automation.pdf` |
| Signature | Conditions under which a Lorentzian-type signature emerges in the derived construction | `RG - GR Signature Emergence.pdf`; `A Derived Lorentzian-Type Signature.pdf` |

### Constructive automation packages

| Package | What is automated | File | Battery |
|---|---|---|:---:|
| **Blind scalar sector** | Restricted scalar-sector selection and admissibility | `Automation of General Relativity - Blind Scalar Sector.pdf` | ✓ |
| **Fisher holes** | Detection and handling of rank loss / non-identifiable directions | `Automation of General Relativity - Fisher Holes.pdf` | ✓ |
| **Source-side action** | Source/action-side path toward observable stress-energy content under declared assumptions | `Automation of General Relativity - Source-Side Action.pdf` | ✓ |

These papers establish **constructive bridges and bounded automation**, not a theorem that arbitrary GR models can be generated or solved from observations. They also do not turn Fisher information into the spacetime metric. Gauge closure, admissibility, model class, and physical interpretation remain explicit inputs or gates.

---

## Elimination result and novelty calibration

The elimination programme asks whether any RG component survives reduction to established mathematics. Individually, none does:

| RG component | Established reduction |
|---|---|
| observational equivalence | equality-of-law quotient of a statistical experiment |
| resolved / unresolved sectors | image, kernel/cokernel, metric orthogonal complement, or spectral subspaces according to protocol |
| local distinguishability | Fisher–Rao geometry in the characterized classical domain |
| resolved selector | generalized spectral projector |
| coupling | off-diagonal covariance block, canonical correlation, and frame-alignment data |
| recoverability | Gaussian conditioning / least squares where those assumptions apply |
| blind recursion | Schur complement |
| second-order lift | second fundamental form and Gauss–Codazzi–Ricci structure |
| globalization | connection, groupoid, Čech descent, monodromy, and character data |
| singular sectors | stratified orbit spaces, Weyl chambers, PSD cones, and slice models |

Accordingly, RG claims neither a new primitive nor a new branch of mathematics. Its defensible contribution is a **new protocol-explicit assembly and characterization for partial observability**, together with theorem/battery separation, gauge-aware automation, and explicit failure conditions.

The older claim that the residual must be described as a uniquely nonzero “descent class” should be read as supporting synthesis language, not as the axiomatic foundation. The current foundation is the modular characterization theorem in **`RG - Axioms.pdf`**.

---

## Real-data and application anchors

The framework has been exercised beyond synthetic examples. These studies test different layers and should not be conflated with proof of universality.

- **OMNI space-weather data:** predictive resolved–null coupling, lag dependence, and condition-dependent frame rotation; no claimed nonzero net topological winding.
- **USGS seismic catalogue:** coupling signal but an honest negative for smooth-loop holonomy under the tested protocol.
- **IGRF geomagnetic models:** resolution-hole diagnostics recover the expected growth of instability toward poorly resolved harmonic degrees.
- **H1–L1 gravitational-wave data:** gauge-aware observable-sector and degeneracy placement demonstrations.
- **ECG data:** gauge-aware $W_{\mathrm{obs}}$ construction in a distinct signal domain.
- **Optical measurement systems:** admissible observable-sector analysis in `RG - Admissible Observable Sectors in Optical Measurement Systems.pdf`.

These are demonstrations of observational geometry and automation. They do not establish new domain physics or prove that one fixed protocol is universal across instruments.

---

## Selected supporting corpus

### Structure, necessity, and falsification

- `RG - Necessity.pdf` (+ ZIP)
- `RG - OMNI Necessity.pdf` (+ `RG - OMNI battery.zip`)
- `RG - Seismic Necessity.pdf` (+ ZIP)
- `RG - Hole Rejection.pdf` (+ ZIP)
- `RG - Undermine Attacks.pdf` / `RG - Undermine Attacks Improved.pdf` (+ ZIP)
- `RG - Final Chaotic Test.pdf`
- `RG - Elimination - Lock Conclusion.pdf` (+ ZIP)
- `RG - Saturation.pdf`
- `RG - Stratified Null.pdf`
- `RG - Blind Sector.pdf`
- `RG - Canonical Connection.pdf`
- `RG - Functorial Resolution Geometry.pdf` (+ ZIP)
- `RG - Fundamental Theorem.pdf`

### Applications and reductions

- `RG - CHI Reduction.pdf` (+ ZIP)
- `RG - Admissible Observable Sectors in Optical Measurement Systems.pdf` (+ ZIP)
- `RG - Physics Path Draft.pdf`
- `RG - Origins.pdf`
- `RG - Manifest.pdf`

### Detailed CTMT-era results still used as support

- `Complete Invariants of CTMT Covariance Resolution Geometry.pdf`
- `Coupling-Aware Estimation in CTMT.pdf`
- `Independent-Protocol Recovery of Resolved–Null Coupling.pdf`
- `Čencov–Inversion Compatibility for CTMT Transport.pdf`
- `The CTMT Compatibility Lock and Holonomy Obstruction.pdf`
- `The CTMT Resolved–Null Covariance Coupling.pdf`
- `The CTMT Dynamics Skeleton.pdf`
- `The CTMT Dynamics II.pdf`
- `The CTMT Transport-Class Rigidity .pdf` (+ ZIP)
- `The CTMT Trajectory-Gated Persistence.pdf` (+ ZIP)
- `Trajectory-Resolved CTMT Batteries.pdf` (+ ZIP)

---

## Scope and non-claims

1. **No geometry of latent reality.** RG describes distinctions supported by a declared experiment; it does not prove that latent reality is exhausted by observational equivalence classes.
2. **No protocol-free uniqueness.** The quotient is universal, but $G$, $\tau$, nuisance equivalence, stability rules, and physical semantics require declaration or an additional characterization theorem.
3. **No unrestricted Fisher claim.** Fisher–Rao is selected within the regular finite classical module and up to normalization. Singular, quantum, infinite-dimensional, non-dominated, and strongly nonregular experiments need separate treatment.
4. **No Fisher = spacetime metric claim.** The Fisher geometry is the geometry of local distinguishability. GR placement preserves that distinction.
5. **No universal information–physics identity.** The entropy decomposition is Shannon’s chain rule on the observational quotient. Energy, temperature, equilibrium, and $k_B$ are not selected by quotient logic.
6. **No automatic full GR solver.** Current automation is sector- and assumption-bounded. It does not derive arbitrary field equations, sources, gauges, or boundary conditions from raw data.
7. **No physical interpretation of coupling by default.** Resolved–null correlation may arise from dynamics, preparation, nuisance structure, or instrumentation. Physical attribution requires an independent intervention or model test.
8. **No theorem from batteries alone.** Numerical attacks test implementations and stronger conjectures; theorem status comes from stated hypotheses and proofs.

---

## Open problems — current frontier

The axioms and GR bridges close several older questions: the observational domain is now fixed by a universal quotient; the regular local metric module is explicitly separated; the hard resolved projector is characterized; natural transport and cocycle gluing are stated; gauge-aware sector construction has restricted real-data demonstrations; and the final stratified geometry is visible. The remaining frontier is sharper:

### 1. Characterize more of the admissibility protocol

The largest foundational gap is no longer the quotient or hard selector, but the declared protocol $\Gamma$. Seek decision-theoretic, resource-theoretic, or physical characterizations of:

- the comparison metric $G$;
- threshold or soft-resolution rule $\tau$;
- nuisance/gauge closure;
- stability margin and conditioning bounds;
- admissible coarse-graining and transport class.

A successful result must derive these from explicit operational assumptions, not hide them in a preferred coordinate system or regularizer.

### 2. Extend the characterization beyond regular finite classical experiments

Develop precise analogues for:

- singular and rank-changing statistical models;
- infinite-dimensional inverse problems and field-valued observations;
- non-dominated and path-space experiments;
- quantum statistical experiments;
- nonlinear sectors where a single local Fisher operator is insufficient.

The goal is not to assume Fisher–Rao survives unchanged, but to state the correct module and its domain.

### 3. Global stratified atlas and discriminant control

Local cocycle closure is characterized; full constructive globalization across rank changes is not. Open tasks include:

- existence and uniqueness of atlases across changing strata;
- transport through or around threshold/eigenvalue collisions;
- computable admissible holonomy/monodromy sets;
- slice-compatible numerical continuation;
- intersection/cohomological invariants that add operational content rather than terminology.

### 4. Completeness and reconstruction on non-generic strata

Generic-stratum invariants are substantially understood, and canonical correlations alone are known to be incomplete in higher rank. A full classification is still needed when eigenvalues repeat, ranks jump, stabilizers enlarge, or frame-alignment data become non-unique.

### 5. Full automation from observation to admissible physical tensor content

The GR papers establish restricted bridges for scalar sectors, Fisher holes, gauge-aware physical directions, and source-side action. The next hard step is a theorem-level pipeline

$$
(\mathcal E,\Gamma)
\longrightarrow W_{\mathrm{obs}\mid\mathrm{adm}}
\longrightarrow \text{sector map}
\longrightarrow T^{\mathrm{obs}}_{\mu\nu}
$$

with explicit hypotheses for uniqueness, conservation compatibility, gauge independence, boundary data, and failure. This must distinguish “not identifiable” from “identified as zero.”

### 6. Intervention-level separation of physical coupling from protocol coupling

Observational covariance alone cannot decide whether coupling is dynamical. Needed are controlled perturbations, independent sensor protocols, or causal/interventional data that can separate system transport from preparation and instrument effects.

### 7. Higher-order observable geometry

Second-order moduli and the second fundamental form are opened but not fully characterized operationally. Open questions include observable estimators, uncertainty bounds, higher-order nuisance closure, normal-holonomy reconstruction, and compatibility with the stratified fibre at rank changes.

### 8. Real-data universality and benchmark suite

Cross-domain demonstrations now exist, so “use any real dataset” is no longer the right frontier. What is needed is a preregistered benchmark suite with fixed gates, held-out protocols, failure cases, and cross-instrument replications. Universality should mean transportable axioms and diagnostics—not one universal threshold, metric, or physical interpretation.

### 9. Entropy beyond the deterministic discrete quotient

The discrete quotient identity is closed. Extensions should address continuous variables, sufficient sigma-algebras, coarse-grained relative entropy, non-equilibrium path measures, and singular fibres without dropping conditional/fibre terms.

---

## Honesty ledger — retired or bounded claims

The surviving framework depends on preserving negative results.

- **Kolmogorov turbulence from Fisher-rank loss:** retired; the proposed conservation step and exponent closure failed.
- **Recovery of constants or $\pi$-factors from flexible kernels:** retired as reparametrization rather than confirmation.
- **Emergent spacetime, gravity from Fisher geometry, nodes of presence, and quantum/biological identifications:** not supported by the observational geometry and not part of RG.
- **Circular Omori validation:** retained only as a bounded consistency/negative result.
- **Universal or nonzero natural holonomy:** not established; some real-data tests return rotation with approximately zero net winding, and others correctly reject loop structure.
- **Complete-Framework primacy:** superseded. `RG - Complete Framework.pdf` remains a useful legacy synthesis, but `RG - Axioms.pdf` is now the foundation.

Correction and retirement records remain in the repository, including `Correction and Maturation of the CTMT Redshift Claim.pdf` (+ ZIP).

---

## Historic / pre-rigorous / retired (quarantined)

Preserved for intellectual history; not part of the current theorem claims.

- **Chronotopic Theory of Matter and Time:** `- I`, `- II`, `- III`, `- IV`, `- CHI`, `- Causality`, `- Seepage`.
- **Chronotopic Metric Theory:** original overview, physics, and trigonometry papers.
- **Retired physics attempts:** universal causal energy transport, Newton-$G$ boundary, radiative constants, emergent time/signature interpretations, nodes of presence, and early geomagnetic physical claims.
- **Pre-rigorous notes:** axial geometry, Hessian boundary constants, visible-band null transport, elemental computation, early gauge uniqueness, stationary phase, calculus, and minimal falsification attempts.
- **Assets and utilities:** site files, fonts, images, scripts, JSON outputs, and standalone battery archives.

---

## Status, citation, and license

- **Axiomatic observational core:** characterized in the stated regular finite classical domain.
- **Hard resolved projector:** characterized relative to $(F,G,\tau)$ away from the discriminant.
- **Final geometric home:** stratified orbit-space bundle with Fisher/base and angular/conical fibre structure.
- **GR placement:** explicit and gauge-aware, with bounded automation and real-data demonstrations.
- **Protocol selection, singular/global extension, and general physical automation:** open.

DOI: [10.5281/zenodo.18229539](https://doi.org/10.5281/zenodo.18229539)  
OSF: [10.17605/OSF.IO/RFE8N](https://osf.io/RFE8N/)  
Author: **Matěj Rada**  
License: **CC BY-NC-ND 4.0**

Serious questions, counterexamples, and attempts to break the theorems are welcome. A clean failure under the stated hypotheses is a contribution.
