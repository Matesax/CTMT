# Resolution Geometry (RG)

*formerly CTMT*

**Resolution Geometry is the geometry of what a declared family of observation protocols can distinguish, and of every structure that this distinguishability forces.**

The primitive is the equality-of-law relation of a joint observation law. That relation is a groupoid, and its orbit space is the observable quotient. The stabilizer of the law, taken modulo its gauge part, is an effective symmetry group that acts faithfully on the quotient.

Every further structure is added in a fixed order. Each addition is either forced by the preceding objects, imported from an established theorem, or conditioned on a named and testable hypothesis. The structures, in order, are:

- differential and stratified structure;
- statistical, quantum and probabilistic structure;
- causal and characteristic structure;
- scale and transport;
- duration and dynamics;
- calibrated physical distance.

The statement of record is **`Resolution Geometry.pdf`**, the print edition of October 2026 (31 pages). It is a ladder of **116 rungs, L0–L115**, with backward-only dependencies, proofs or precise citations for every step, a computational-complexity section, and **two** remaining open problems.

---

## The object

An observation system consists of three parts:

- a latent space $X$;
- protocols $F_i : X \to Y_i$, valued in spaces of observable laws;
- a declared relabeling group $\Psi$.

The joint law is

$$
F = (F_i)_{i \in I} : X \longrightarrow Y = \prod_i Y_i .
$$

Its kernel groupoid and quotient are

$$
\mathcal K = X \times_Y X, \qquad \pi : X \longrightarrow Q = X / \mathcal K \cong \mathrm{im}\, F .
$$

The stabilizer of the law, its gauge part (the bisections of $\mathcal K$), and the effective symmetry group fit into one exact sequence:

$$
1 \longrightarrow \mathrm{Gau} \longrightarrow \mathrm{Stab} \longrightarrow \Gamma \longrightarrow 1 .
$$

A **natural module** is a pullback $M = F^{*}\mu$ of a relabeling-invariant structure on law space. Examples are observable invariants, clock readings, record counts, the Fisher tensor and law-defined cones. On a regular stratum every natural module descends and is $\Gamma$-invariant:

$$
M = \pi^{\ast} \bar M, \qquad \gamma^{\ast} \bar M = \bar M \quad \text{for all } \gamma \in \Gamma .
$$

Representative independence, basicness and projectability of law-defined modules are therefore theorems, not hypotheses. What remains is either declared or empirical:

- the protocols, the task and the admissible class;
- the content of $\Gamma$, tested by equality of laws;
- regularity of the stratum;
- latent (non-natural) data;
- anchors and held-out validation.

---

## The calculus: how RG computes

RG is used as a short sequence of operations. Each operation has a precise criterion and a typed failure mode.

| Step | Operation | Criterion | Rung |
|---|---|---|---|
| 1 | **Declare** protocols, relabelings, task, admissible class, hypotheses | the declarations are fixed before the output is seen | L0, H-adm |
| 2 | **Quotient** by equality of laws | a task $T$ is law-determined iff $\mathcal K_F \subseteq \mathcal K_T$; then $T = \bar T \circ \pi$ | L1–L2 |
| 3 | **Combine** protocols | $\mathcal K_{(F_i)} = \bigcap_i \mathcal K_{F_i}$; under model uncertainty, take the equivalence closure of the union | L3, L14 |
| 4 | **Select** panels | the complete antichain of inclusion-minimal sufficient panels; ties retained | L12 |
| 5 | **Descend** natural modules | basic and $\Gamma$-invariant automatically; Fisher radical $= \ker DF$ | L19–L20, L28 |
| 6 | **Reduce** nuisance | efficient metric by Schur complement | L33 |
| 7 | **Stratify and specialize** at walls | specialization map iff $\mathcal K_\varepsilon \subseteq \mathcal K_0$; correspondence otherwise | L21–L27 |
| 8 | **Certify** in declared precision units | rank (double escalation), Hellinger, forgetting (operator bound over a class), frontier | L46, L36, L55, L88 |
| 9 | **Anchor** | a counted channel removes the dilation fibre; SI radar anchor; held-out validation | L92–L107 |
| 10 | **Dispatch** | exactly one status: LOCKED, ESCALATE, CORRESPONDENCE, VACUOUS, INCONSISTENT, STRUCTURAL WALL[ε] or OUTSIDE DOMAIN | L110–L113 |

Five formulas carry most of the weight:

$$
\text{descent:}\quad T = \bar T\circ\pi \iff \mathcal K_F \subseteq \mathcal K_T
\qquad\qquad
\text{efficient metric:}\quad g_{\mathrm{eff}} = I_{\theta\theta} - I_{\theta\eta} I_{\eta\eta}^{-1} I_{\eta\theta}
$$

$$
\text{forecast frontier:}\quad n^\ast(\sigma_0) = \frac{\log(\varepsilon/\sigma_0)}{\lambda} + c_P
\qquad\qquad
\text{non-circularity:}\quad \sup_{\mu,\nu \in \mathcal A} d\big(\Phi_{t\#}\mu,\Phi_{t\#}\nu\big) \le \tau
$$

$$
\text{soundness:}\quad \Pr(\text{false LOCKED}) \le \sum_{g} \alpha_g .
$$

A result is **LOCKED** only when every gate of the requested module is certified. Otherwise RG returns the maximal observable correspondence or a typed abstention. **No admitted request ever requires choosing between alternatives the protocol cannot tell apart.**

---

## The ladder at a glance

| Stratum | Rungs | Main results |
|---|---|---|
| Exact | L0–L5 | kernel groupoid; universal factorization; joint kernel is the intersection of protocol kernels; garbling enlarges the kernel; the exact layer fixes no metric, order, cone, scale or unit |
| Symmetry | L6–L11 | exact sequence with faithful $\Gamma$; non-splitting; count rigidity; orbit–quotient comparison; saturation hierarchy; coherent completeness |
| Observable sector | L12–L17 | complete minimal-panel selector; no task-free isotropic selector; more protocols intersect, more models close; common model-blind decoder; a coupled law refines its marginals; confidence-set bridge; tolerance is not a groupoid |
| Differential and strata | L18–L27 | tangent sequence; natural descent; slices and tangent cones; kernel upper semicontinuity; specialization iff kernel inclusion; symmetry descent through the normalizer; stratified quotient category |
| Second order and quantum | L28–L42 | Fisher is natural; covariance is not determined by Fisher; quotient dimension $p+q+pq$; efficient metric; data processing; Hellinger specialization; Petz family gate; task lock (SLD, BKM, WY); nonparametric Petz identification; quantum walls; trace rule; contextual protocols (Fine) |
| Ignorance | L43–L46 | persistent null; sensor, precision and representation nulls; finite-sample reconstruction; rank walls are never certified from finite data |
| Uncertainty and forgetting | L47–L56 | uncertainty splits along the quotient; de Finetti; LAN and branch selection; TV rigidity; no uniform forgetting without a class; sharp BV mixing lock; Navier–Stokes contraction gate; Dobrushin blocks; samples refute and operators lock; non-circularity certificate; non-dominated and infinite-dimensional strata |
| Statistical distance | L57–L61 | RG distance is representative-free, chart-free and dimensionless; monotone under garbling; no angle in rank one; finite divergences are not rulers; experiment-relative |
| Order, cone and residual | L62–L71 | reachability; projective typing; realization branches; conformal ray; Lorentz forcing; tame reduction; no finite linear classification; duality; hyperbolic lock; wild class |
| Scale and transport | L72–L78 | scale cocycle; strict transport; induced connection; entropy selects no connection; no law-natural connection; canonical connection under declared covariance |
| Duration and dynamics | L79–L87 | clock transition groupoid; affine clocks; path independence; coherence duration; noiseless finite histories; kernel-count entropy; Ruelle and Pesin locks; typed dynamical certificates; cone-field criterion |
| Chaos frontier | L88–L91 | frontier slope $1/\lambda$; duality $2\lambda$; quotient growth $h$; information cost of $\lambda/\ln 2$ bits per certified step; clock distortion classes |
| Physical distance | L92–L107 | no internal unit; counted channels; count loss; ratio scale; two-point invariant; metric proportionality; anchor budget; unit-distance rigidity; radar and rod distance; stationary slices and Sagnac holonomy; non-stationary no-go; media; ergodic transfer; identification theorem |
| Assembly | L108 | kernel–stabilizer theorem and compatibility equations |
| Global transport and closure | L109–L113 | global transport; dispatcher totality; composite soundness; window-indexed walls; maximal closure |
| Open | L114–L115 | enumeration of tame characteristic types (external); uniform forgetting for forced deterministic turbulence |

Each rung has exactly one status: **F** forced, **I** imported, **H** hypothesis-conditioned (the hypothesis is named), **Γ** conditional on the effective symmetry group, **N** no-go, or **O** open.

**Named hypotheses:**
- regularity and nuisance: `H-reg`, `H-nuis`, `H-sens`;
- declared groupoids: `H-sound`, `H-compl`, `H-coh`;
- cone, scale and anchors: `H-top`, `H-int`, `H-SI`, `H-med`, `H-val`;
- walls: `H-unif`, `H-hell`;
- task and latent data: `H-task`, `H-cov`;
- probability and uncertainty: `H-stat`, `H-exch`, `H-adm`;
- symmetry: `H-Γ`.

Each is stated in the paper together with the test that falsifies it.

---

## Frontier examples

These are worked results from the module batteries. Each is reproducible from its ZIP with fixed seeds and mutation canaries.

**Chaos is not hidden (L88–L91).** The boundary between LOCKED and ESCALATE forecasts is a straight line of slope $1/\lambda$ in the log of the declared precision. A protocol moves only its intercept. Recovered across five seeds:

| System | true λ | λ recovered from RG's own frontier |
|---|---|---|
| logistic, r = 4 | 0.693 | 0.674 – 0.696 |
| logistic, r = 3.9 (median frontier) | 0.496 | 0.496 – 0.509 |
| Hénon | 0.420 | 0.414 – 0.421 |
| Lorenz flow | 0.906 | 0.87 – 0.93 |

Other measured quantities:
- **Protocol invariance:** three sensor protocols give slopes within 1–7% of each other, with intercepts about 4 steps apart.
- **Past and quotient:** Fisher information about the past grows at $2\lambda$ (1.015 against 0.993), and observable quotients grow at $h$ (0.6931 at 2, 4 and 8 bins).
- **Information cost:** each certified step costs $\lambda/\ln 2$ bits of initial precision (Brudno, Pesin).
- **Hiding chaos:** the only way is to change the task, for example by re-observing the state, which is a declaration and is caught.

See `RG - Chaos Is Not Hidden.pdf` and `RG - Chaos Frontier Status Map.png`.

**The wall–tap experiment (origin, L0–L113 on one system).** The experiment is declared as five cumulative acoustic protocols.
- **Saturation:** unresolved directions fall 5, 4, 3, 2, 2 by kernel intersection, and close only with length and gain anchors.
- **Hidden fold:** the rhythm channel hides an exact $\mathbb Z_2$ fold, $b \leftrightarrow 2d(x)-b$, on 56% of the domain. It is a wall inside the regular stratum, removed only by a new signed channel.
- **Matched wall:** specialization holds under Hellinger continuity, with Fisher rank 3 → 1 and Cramér–Rao bound $O(\varepsilon^{-2})$.
- **Medium:** wind turns echo distance into the acoustic Landau–Lifshitz metric.
- **Closure:** 3000 random requests per seed gave 0 false LOCKED; seven sabotaged certificates each produced false locks.

See `RG - Wall-Tap Experiment.pdf`.

**Distance versus angle (L91).** Perceived time can be distorted in three ways, and each preserves a different amount:
- **offset:** preserves differences;
- **rate:** preserves ratios and angles, not distances;
- **warp** (total rupture): preserves only order and counts.

Under memory loss, a remembered distance calibration degrades like $\sqrt T$ (measured exponent 0.494–0.510), while angles from delays measured together stay exact. See `RG - From the Wall to the Kernel.pdf`.

**Quantum task changes holonomy (L37–L42).** On the same qubit slice the SLD metric (estimation) has curvature +1, while the BKM metric (Stein testing) is negatively curved. Metric selection is task-relative, and so is observable transport. No noncontextual hidden values exist: 0 of 512 assignments satisfy the Peres–Mermin square, and Fine's criterion agrees with linear programming on 60/60 random boxes.

**Uncertainty is not a primitive (L47–L56).** Two priors merge on the quotient as $1/n$ and stay apart along the kernel (0.312 at every sample size). On a declared class of bounded variation, mixing locks with a sharp constant:

$$
\mathrm{TV}(T^n_{\#}\mu_f, T^n_{\#}\mu_g) \le 2^{-(n+2)}\,\mathrm{Var}(f-g),
$$

attained exactly at every $n$. Over a finite list of declarations nothing is certified. See `RG - Uncertainty Geometry.pdf` and `RG - Uncertainty Geometry - Forgetting Ladder.pdf`.

**Latent transport is declared (L77–L78).** Vertical shears preserve the law and map any connection to any other. Under a declared covariance, the Kalman/RTS connection is canonical. Its latent holonomy around a closed observed loop scales as the loop's area, and it depends on the declared covariance.

---

## Computational complexity

| Task | Cost |
|---|---|
| kernel of a finite law table | linear hashing, union–find |
| minimal sufficient panels | output can be $\binom{m}{\lfloor m/2\rfloor}$ (Shamir threshold system); a panel of size ≤ k is NP-complete (Minimum Test Collection) |
| stratification of a semialgebraic law | doubly exponential in dimension (cylindrical decomposition) |
| rank certificate | $O(d^3)$ per escalation level |
| certified chaotic forecast to horizon $n$ | at least $n\lambda/\ln 2$ bits of initial data |
| TV certificate without a declared class | impossible from finite samples (Le Cam) |
| Monte Carlo on a FORCED output | $O(\tau^{-2})$ draws |

Exactness is cheap at the exact layer. It is expensive in exactly three places, each reported where it arises:
- selection;
- stratification;
- chaos.

---

## How to read the corpus

1. **`Resolution Geometry.pdf`**: the statement of record (print edition).
2. **`RG - The Groupoid of Resolution.pdf`**: the theory as one groupoid climbing the ladder, bottom-up and top-down, showing what each hypothesis does to it.
3. **`RG - From the Wall to the Kernel.pdf`**: the origin and progress account, including the Terror experiment made exact.
4. **Module papers**, listed below, each with a same-stem ZIP containing its hostile battery.
5. **Historic syntheses**: development record only.

**Statement-of-record rule.** Where a module paper and `Resolution Geometry.pdf` differ in scope, hypotheses or status, the ladder governs. **Citing:** cite the rung (for example "RG, L88"), not the module paper.

**Notation.** In the ladder, $\Gamma$ is the effective symmetry group $\mathrm{Stab}/\mathrm{Gau}$. The early characterization papers use $\Gamma$ for the admissibility protocol, which this README writes $\Pi$.

---

## Module papers by stratum

Every battery registers naive and sharpened claims, uses fixed seeds, fails on non-finite evidence, and carries **mutation canaries**: a check that cannot fail is not counted. Batteries are falsification witnesses, not premises of proofs.

| Stratum | Rungs | Papers (same-stem ZIP = battery) |
|---|---|---|
| Foundation | all | `Resolution Geometry.pdf`; `RG - Final Locks.zip` |
| Groupoid and assembly | L0–L11, L108 | `RG - The Groupoid of Resolution.pdf`; `RG - Assembly at the Kernel.pdf`; `Assembly of Resolution Geometry.pdf`; `RG - Functorial Resolution Geometry.pdf`; `RG - Saturation.pdf`; `RG - Kernel pair root and limits.pdf`; `RG - Quotients.pdf`; `RG - Observation Laws.pdf` |
| Observable sector | L12–L17 | `RG - Observable Sector.pdf`; `RG - Source Automation.pdf` |
| Differential, strata, walls | L18–L27 | `RG - Stratified Specialization.pdf`; `RG - Made Visible.pdf`; `RG - Atlas Globalization.pdf`; `RG - Gluing Test.pdf` |
| Second order and quantum | L28–L42 | `RG - Quantum Metric Selection.pdf`; `RG - Fibrewise Characterization.pdf`; `RG - Fibrewise Characterization - QM.pdf`; `RG - Axioms.pdf`; `RG - Covariance Geometry.pdf`; `RG - Reconstruction Identifiability.pdf`; `RG - Blind Sector.pdf` |
| Ignorance | L43–L46 | `RG - Ignorance.pdf`; `RG - Observation Limit.pdf`; `RG - Null Resolution Depth.pdf`; `RG - Entropy.pdf` |
| Uncertainty and forgetting | L47–L56 | `RG - Uncertainty Geometry.pdf`; `RG - Uncertainty Geometry - Forgetting Ladder.pdf` |
| Statistical distance | L57–L61 | `RG - Distance and Angles.pdf` |
| Order, cone, residual | L62–L71 | `RG - Residual Characteristic Geometry.pdf`; `RG - Cone.pdf`; `RG - Cone - Closure.pdf`; `RG - GR Signature Emergence.pdf` |
| Scale and transport | L72–L78 | `RG - Canonical Transport.pdf`; `RG - Atlas Globalization.pdf` |
| Duration, dynamics, chaos | L79–L91 | `RG - Chaos Is Not Hidden.pdf`; `RG - Chaotic Experiments.pdf`; `RG - Time.pdf`; `RG - Operational Duration.pdf`; `RG - Operational Duration Boundary.pdf`; `RG - GR Time.pdf` |
| Physical distance | L92–L107 | `RG - Physical Distance.pdf` (battery: `RG_Physical_Distance_Attack_Bundle.zip`); `RG - Wall-Tap Delay.pdf` |
| Global transport and closure | L109–L113 | `RG - Maximal Closure.pdf`; `RG - Canonical Transport.pdf` |
| Origin experiment | all | `RG - Wall-Tap Experiment.pdf`; `RG - From the Wall to the Kernel.pdf` |

---

## Physics interface

**Physical distance (L92–L107).** RG distance is dimensionless and experiment-relative, and no function of it alone is a physical distance. Suppose the following hold on a regular static slice:
- Euclidean motions are in $\Gamma$, with irreducible isotropy;
- a counted two-way timing channel with exact SI constants is part of the joint law;
- the medium is resolved;
- a held-out anchor is predicted.

Then

$$
\bar g_{\mathcal E} = \lambda_{\mathcal E}^{2}\, h, \qquad d_{\mathrm{phys}} = \frac{d_{RG}^{\mathcal E}}{\lambda_{\mathcal E}}, \qquad d_{\mathrm{rad}} = \frac{c}{2\,\Delta\nu_{\mathrm{Cs}}}\, n_{\mathrm{Cs}} .
$$

Stationary slices carry the Landau–Lifshitz radar metric, and loop synchronization there has Sagnac holonomy. Non-stationary slices carry no time-independent radar metric (no-go). The unit enters only through the counted channel.

**Duration (L79–L82).** Clock readouts with equal kernels form one transition groupoid. A stabilizer time flow makes the transitions affine. Path independence holds iff the kernel of the increment is contained in the kernel of the clock.

**Cone (L62–L71).** A regular quadratic cone gives a conformal ray. Lorentz isotropy in $\Gamma$ (dimension at least 2+1) forces the light cone. Otherwise the characteristic set falls into one of three classes:
- **tame-typed**, with finite stratified data;
- **hyperbolically locked**, which is what well-posed propagation forces;
- **wild**, with window certificates only.

**GR placement and automation.** These are bounded constructive bridges, not a general solver: `RG - GR Placement Bridge*.pdf`, `Automation of General Relativity - *.pdf`, `RG - Source Automation.pdf`. A fail-closed result (for example, the H1–L1 records failing the declared injection-veto gate) is an intended output.

---

## Problem status

### Closed

| Problem | Resolution | Rungs |
|---|---|---|
| Observable domain | equality-of-law quotient, universal factorization | L1–L2 |
| Coupling of protocols | joint kernel is the intersection | L3 |
| Shared symmetry | Gau → Stab → Γ; faithful action on Q | L6–L7 |
| Observable-sector selection | complete minimal-panel antichain; ties retained; no task-free isotropic selector | L12–L13 |
| Robust observability | protocols intersect, models close their union; common decoder via cross-model collisions | L14 |
| Coupled versus marginal detector laws | coupled law refines marginals; pairing is not coupling | L15 |
| Set-valued inference | confidence-set bridge; point answers need a certified radius | L16 |
| Basicness and projectability | automatic by natural descent | L19–L20 |
| Walls | kernel upper semicontinuity; specialization iff kernel inclusion; normalizer descent; stratified category | L21–L27 |
| Distinguishability across walls | Hellinger (Le Cam) specialization; weak convergence insufficient | L36 |
| Fisher layer | natural; radical = observational vertical; does not determine covariance; quotient dimension p+q+pq | L28–L31 |
| Quantum metric | Petz family gate; task lock; nonparametric identification; wall specialization iff f(0) > 0 | L37–L40 |
| Contextual quantum protocols | trace rule; no noncontextual hidden values; product law over contexts (Fine) | L41–L42 |
| Finite-data rank walls | never certified; full-rank or upper-bound certificates only | L46 |
| Primitive of uncertainty | indistinguishability; uncertainty splits along the quotient; kernel prior never forgotten | L47 |
| Probability, LAN, Monte Carlo | de Finetti; Gaussian propagation = LAN limit (fails across folds); Monte Carlo objective only on FORCED outputs | L48–L49, L55 |
| Forgetting of declarations | TV rigidity under bijections; no uniform forgetting without a class; sharp BV lock; NS contraction gate; Dobrushin; samples refute, operators lock | L50–L54 |
| Non-circularity | certificate over a declared admissible class | L55 |
| Non-dominated, infinite-dimensional, path-space | pairwise domination; Feldman–Hájek / Cameron–Martin geometry | L56 |
| Residual characteristic branch | tame reduction; no finite linear classification; duality; hyperbolic lock; wild class | L67–L71 |
| Latent connection | not law-natural; canonical under declared covariance (Kalman/RTS); curvature = latent holonomy; wall coherence | L77–L78 |
| Operational duration | clock groupoid; affine under time flow; path independence | L79–L81 |
| Noiseless chaos | stratified-regular finite histories; entropy from kernel counts; Ruelle and Pesin locks; typed certificates | L83–L86 |
| Nonuniform dynamical transport | global cone field iff uniform hyperbolicity (no-go otherwise); window-typed certificates | L87 |
| Chaos versus declarations | frontier law; duality; quotient growth; information cost | L88–L90 |
| Delay, distance, angle | offset, rate and warp classes; angles survive rate change, counts survive warp | L91 |
| Unit and physical distance | no internal unit; counted channel; anchors; identification theorem | L92–L107 |
| Non-static slices | stationary: Landau–Lifshitz plus Sagnac; non-stationary: no-go | L102–L104 |
| Assembly | kernel–stabilizer theorem | L108 |
| Global transport | observable atlas closed; latent transport canonical under declared data | L109 |
| Closure | dispatcher totality; composite soundness ≤ Σαg; window-indexed walls; maximal closure | L110–L113 |

### Open problems of the ladder

1. **Tame type enumeration (L114, external).** Enumerate the tame types of degree-d characteristic cones. For plane curves this is the first part of Hilbert's sixteenth problem (known for d ≤ 7). It is a problem of real algebraic geometry, not of RG.
2. **Forced deterministic turbulence (L115).** Above the Galerkin gate $G_N \ge 1$, no rung certifies uniform forgetting. TV is rigid, the contraction gate fails, and there is no noise and no expanding non-invertible transfer operator. With stochastic forcing, ergodicity is known only qualitatively (Hairer–Mattingly). An explicit uniform constant over a declared class is open.

### Declarations by design (not open problems)

- **Protocol, task and admissible class.** RG does not derive which question an observer must ask. This is a theorem, not a gap: there is no task-free equivariant selector (L13). Given a task, selection is exact (L12). Given a declared class, non-circularity is certified (L55).
- **Latent data.** A connection, prior or covariance is declared or reported as undetermined (L77–L78).
- **Anchors.** The unit enters only through counted channels and SI definitions (L92–L100).

### Research directions beyond the ladder

These are extensions with new content. They are not gaps in the ladder's proofs.

- **Observable stress tensor:** a sector map with uniqueness, conservation and gauge independence, keeping *not identifiable* distinct from *identified as zero*.
- **Physical versus protocol coupling:** attribution by intervention and independent instrumentation.
- **Higher-order geometry near discriminants:** curvature, normal holonomy, second fundamental form.
- **Classifications:** complete stabilizer classification on non-generic strata, and enumeration of flat observable classes by character data.
- **Continuous entropy:** continuous differential-entropy decompositions beyond kernel counts and finite-resolution quotient growth.
- **Wild-class invariants:** operationally stable invariants beyond count exponent, box dimension and Cantor–Bendixson rank.
- **Benchmarks:** a preregistered fixed-gate benchmark suite with held-out and deliberately failing datasets.

---

## Scope and non-claims

1. **No geometry of latent reality.** RG describes the distinctions supported by declared protocols. Latent representatives exist, but observation does not select them.
2. **No protocol-free uniqueness.** The quotient is universal. Protocols, relabelings, tasks, nuisance and admissible classes are declared.
3. **Domains of the metric layer.** The metric module depends on the domain:
   - **Classical, regular, finite:** Fisher–Rao is selected (Čencov).
   - **Quantum:** the Petz member is selected by the declared task.
   - **Non-dominated or infinite-dimensional:** the Fisher layer is absent, or reduces to Cameron–Martin geometry.
4. **No Fisher = spacetime identity.** On a certified static slice, Fisher is *proportional* to the physical spatial metric, under named hypotheses.
5. **No internal unit.** Statistics, flatness and thermodynamic length supply no metre.
6. **No automatic Lorentzian structure.** Lorentzian structure requires the quadratic gate or sufficient isotropy in Γ.
7. **No field equations.** No Einstein equation, matter dynamics or quantum measurement law is derived.
8. **No physical coupling by default.** Resolved–null correlation is a protocol-metric property; attribution needs intervention.
9. **No theorem from batteries.** Theorem status comes from proofs and cited theorems; batteries are falsification witnesses.
10. **No prediction beyond the frontier.** RG does not forecast chaos beyond its horizon. It certifies where the horizon lies and what extending it costs.

---

## Novelty calibration

Every RG component reduces to established mathematics:

- **Quotients and symmetry:** equality-of-law quotients and Lie groupoids; Schur's lemma.
- **Statistics:** Fisher–Rao geometry and Čencov uniqueness; Le Cam, Blackwell and LAN; Bernstein–von Mises; de Finetti.
- **Quantum:** Petz, Gleason, Busch and Fine.
- **Ergodic theory and dynamics:** Oseledets, Pesin, Ruelle and Brudno; Lasota–Yorke and Dobrushin; Feldman–Hájek.
- **Geometry and characteristics:** o-minimal and semialgebraic stratification (Hardt, Collins); Gårding hyperbolicity.
- **Physics and measurement:** Malament–Hawking reconstruction; Beckman–Quarles; Cauchy–Hölder ratio scales; Landau–Lifshitz radar metric; the SI definitions.
- **Complexity:** Garey–Johnson; Shamir.

RG claims no new primitive and no new branch of mathematics. Its contribution has three parts:

- a protocol-explicit, dependency-typed assembly of these components into one geometry of partial observability;
- the kernel–stabilizer mechanism, which makes all modules share one symmetry group;
- explicit failure conditions for every step, each backed by a hostile battery with mutation canaries.

---

## Honesty ledger

The surviving framework depends on keeping negative results visible.

- **Kolmogorov turbulence from Fisher-rank loss:** retired.
- **Constants or π-factors from flexible kernels:** retired as reparametrization.
- **Emergent spacetime, gravity from Fisher geometry, nodes of presence, quantum or biological identifications:** not part of RG.
- **RG distance as physical distance:** refuted. Physical distance needs the identification theorem (L107).
- **Independent module gates:** superseded by natural descent (L19–L20).
- **Resolved–null coupling as the primary law invariant:** cut. It vanishes identically under the whitening (Čencov) metric. It is the relative position of a declared protocol metric, and its efficiency theorem is data processing.
- **Principal Aut(G)-bundle over a varying operating point:** cut. It exists only on level sets of varying invariants; the law-natural structure group is O(r)×O(q).
- **Local automorphisms as symmetries:** cut. Jet isotropy need not integrate to a law symmetry.
- **"Delay is the invariant":** corrected. Delays scale under rate change; ratios, angles and counts are the invariants.
- **CHI coherence volume:** audited.
  - Its bottleneck inequality is reversed: the harmonic mean is at least the minimum.
  - Its Fisher-rank seal is blind to misspecification.
  - Density cancels in the wind example.
  - The condition-number hazard is retained as a rank-wall indicator.
- **Monte Carlo as the source of uncertainty:** superseded. Monte Carlo is the integration engine for FORCED outputs, not the lock.
- **Forgetting over finite lists of declarations:** corrected. Locks are suprema over declared classes (L51–L55).
- **Earlier wall–tap regular quotient (x, r, b):** corrected to (x, r, |b − d|), because of the Z₂ fold.
- **Earlier batteries:** several checks could not fail and are replaced by computed, canaried checks.
  - Literal `True` checks; class indices used as features; hand-set dispatcher booleans; vacuous invertible-law inclusion tests.
  - The chaos/coil attack, which passes on a non-chaotic surrogate.
- **Earlier synthesis papers as foundation:** superseded by `Resolution Geometry.pdf`.

Correction records remain in the repository.

---

## Repository policy

The repository preserves the full development record: foundations, module papers, batteries, corrections, superseded formulations and retired claims. Older files are not deleted or silently rewritten.

| Label | Meaning |
|---|---|
| `[foundation]` | statement of record: `Resolution Geometry.pdf` |
| `[module]` | extended proofs and batteries for one stratum; governed by the foundation |
| `[supported]` | constructive bridge, implementation, or real-data demonstration |
| `[historic]` | superseded presentation retained as development record |
| `[retired]` | withdrawn claim retained so the correction is visible |

**Historic syntheses** `[historic]`:
- `RG - Theorem Ladder.pdf`, `RG_-_Theorem_Ladder_Final.pdf` (the earlier ladders, L0–L60 and L0–L92);
- `RG - Complete Framework.pdf`, `RG - Synthesis.pdf`, `RG - Fundamental Theorem.pdf`, `RG - Locked Foundation.pdf`, `RG - Atlas.pdf`;
- `Resolution Geometry of Observation Systems.pdf`, `The CTMT - Automorphisms of Resolution.pdf`;
- `The CTMT - Testament of 22 years.pdf`.

**Falsification and necessity studies** `[supported]`:
- `RG - Necessity.pdf`, `RG - OMNI Necessity.pdf`, `RG - Seismic Necessity.pdf`;
- `RG - Hole Rejection.pdf`, `RG - Undermine Attacks.pdf`, `RG - Undermine Attacks Improved.pdf`, `RG - Final Chaotic Test.pdf`;
- `RG - CHI Reduction.pdf`, `RG - Elemental Characterization.pdf`.

**Real-data studies** `[supported]`:
- OMNI space weather;
- USGS seismic catalogue;
- IGRF geomagnetic models;
- H1–L1 gravitational-wave strain (fail-closed injection gate);
- ECG;
- optical measurement systems.

These test individual strata. They are not proofs of universality.

**Historic, pre-rigorous and retired (quarantined):**
- Chronotopic Theory of Matter and Time (I–IV, CHI, Causality, Seepage);
- original Chronotopic Metric Theory papers;
- retired physics attempts;
- pre-rigorous notes.

---

## Origins

RG began as a coherence project: an attempt to force structure on whatever holds physical description together. CTMT was the first forced model. CTMT-Metric was the second, obtained by falsifying CTMT with Fisher geometry. RG is the third: the part that survived.

The childhood wall–tap experiment, which asked whether a delay lives in the observer or in the world, is now worked through the whole ladder, and its answers are exact:
- latency cancels;
- a common rate is undecidable without a second clock;
- angles survive a change of rate;
- counts survive total rupture.

See `RG - From the Wall to the Kernel.pdf` and `RG - Wall-Tap Experiment.pdf`.

---

## Citation and license

DOI: [10.5281/zenodo.23152797](https://doi.org/10.5281/zenodo.23152797)
Author: **Matěj Rada**
License: **CC BY-NC-ND 4.0**

Historic CTMT: DOI [10.5281/zenodo.18229539](https://doi.org/10.5281/zenodo.18229539) · OSF [10.17605/OSF.IO/RFE8N](https://osf.io/RFE8N/)

Counterexamples and attempts to break the theorems are welcome. A clean failure under the stated hypotheses is a contribution.
