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

What remains after every gate is certified is **not a new primitive**. On second-order strata of constant canonical-correlation multiplicity, the final remainder is global structure of a classical kind. It consists of local canonical-correlation moduli together with a tuple of Euclidean block bundles. This pair is the initial complete natural invariant, and it is classified by $\prod_j B\mathrm O(m_j)\times B\mathrm O(p-r)\times B\mathrm O(q-r)$.

The statement of record is **`Resolution Geometry.pdf`**, the closed print edition of October 2026 (45 pages). It is a ladder of **134 rungs, L0–L133**, with backward-only dependencies, proofs or precise citations for every step, a computational-complexity section, and **two** remaining open problems, both outside RG proper. The second-order remainder across walls is closed (L129–L131): hidden-room descent where chamber stabilizers are discrete, an obstruction tower in general, and a no-go for any constructible exit-path classification. The ladder is closed: later results are companion papers that attach to rungs and add none. Directly after the ladder, **the atlas** (Section 22, with the companion `RG - Realization Atlas.pdf`) charts concrete systems through nine layers:
- A1: exact gluing;
- A2: structured descent;
- A3: metric charts;
- A4: walls;
- A5: equivariant charts;
- A6: global classes;
- A7: stability;
- A8: physical anchors;
- A9: a registry of certified realizations.

Pointwise, the second-order data are classified completely (L124–L128):
- Two triples (R, N, C) are equivalent under the declared-metric gauge O(p)×O(q) iff all ordered word traces of an associated symmetric tuple agree. This holds on every stratum.
- On simple-spectrum strata, the classification reduces to spectra, coupling magnitudes and cycle signs in H¹(G; F₂). These loop values are real hidden values, not gauge.
- In Lorentzian geometry, scalar invariants do not separate curvature, so a declared observer is required.
- No classification selects a dynamics.

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
| 6 | **Reduce** nuisance | efficient metric by Schur complement | L34 |
| 7 | **Stratify and specialize** at walls | specialization map iff $\mathcal K_\varepsilon \subseteq \mathcal K_0$; correspondence otherwise | L21–L27 |
| 8 | **Certify** in declared precision units | rank (double escalation), Hellinger, forgetting (operator bound over a class), frontier | L47, L37, L57, L90 |
| 9 | **Anchor** | a counted channel removes the dilation fibre; SI radar anchor; held-out validation | L94–L109 |
| 10 | **Dispatch** | exactly one status: LOCKED, ESCALATE, CORRESPONDENCE, VACUOUS, INCONSISTENT, STRUCTURAL WALL[ε] or OUTSIDE DOMAIN | L112–L115 |
| 11 | **Report the remainder** | multiplicity type, local moduli and block-bundle classes (e.g. $w_1$ on loops), not local invariants alone; linking monodromy near walls | L116–L123 |

Five formulas carry most of the weight:

$$
\text{descent:}\quad T = \bar T\circ\pi \iff \mathcal K_F \subseteq \mathcal K_T
\qquad\qquad
\text{efficient metric:}\quad g_{\mathrm{eff}} = I_{\theta\theta} - I_{\theta\eta} I_{\eta\eta}^{-1} I_{\eta\theta}
$$

$$
\text{forecast frontier:}\quad n^\ast(\sigma_0) = \frac{\log(\varepsilon/\sigma_0)}{\lambda} + c_P
\qquad\qquad
\text{non-circularity:}\quad \sup_{\mu,\nu \in \mathcal A} d\big(\Phi_{t\sharp}\mu,\Phi_{t\sharp}\nu\big) \le \tau
$$

$$
\text{soundness:}\quad \Pr(\text{false LOCKED}) \le \sum_{g} \alpha_g .
$$

A result is **LOCKED** only when every gate of the requested module is certified. Otherwise RG returns either the exact observable correspondence or a typed abstention. The exact correspondence is the least sound set of jointly attainable outputs, with joint witnesses retained; "maximal" means maximally informative, not largest. **No admitted request ever requires choosing between alternatives the protocol cannot tell apart.**

---

## The ladder at a glance

| Stratum | Rungs | Main results |
|---|---|---|
| Exact | L0–L5 | kernel groupoid; universal factorization; joint kernel is the intersection of protocol kernels; garbling enlarges the kernel; the exact layer fixes no metric, order, cone, scale or unit |
| Symmetry | L6–L11 | exact sequence with faithful $\Gamma$; non-splitting; count rigidity; orbit–quotient comparison; saturation hierarchy; coherent completeness |
| Observable sector | L12–L17 | complete minimal-panel selector; no task-free isotropic selector; more protocols intersect, more models close; common model-blind decoder; a coupled law refines its marginals; confidence-set bridge; tolerance is not a groupoid |
| Differential and strata | L18–L27 | tangent sequence; natural descent; slices and tangent cones; kernel upper semicontinuity; specialization iff kernel inclusion; symmetry descent through the normalizer; stratified quotient category |
| Second order and quantum | L28–L43 | Fisher is natural; descent does not select, Fisher selected under H-mono (Čencov); covariance is not determined by Fisher; quotient dimension $p+q+pq$ under the declared-metric contract; efficient metric; data processing; Hellinger specialization; Petz family gate; task lock (SLD, BKM, WY); nonparametric Petz identification; quantum walls; trace rule; contextual protocols (Fine) |
| Ignorance | L44–L48 | persistent null; sensor, precision and representation nulls; finite-sample reconstruction; rank walls are never certified from finite data; exact law equality is not uniformly certifiable from finite samples |
| Uncertainty and forgetting | L49–L58 | uncertainty splits along the quotient; de Finetti; LAN and branch selection; TV rigidity; no uniform forgetting without a class; sharp BV mixing lock; Navier–Stokes contraction gate; Dobrushin blocks; samples refute and operators lock; non-circularity certificate; non-dominated and infinite-dimensional strata |
| Statistical distance | L59–L63 | RG distance is representative-free, chart-free and dimensionless; monotone under garbling; no angle in rank one; finite divergences are not rulers; experiment-relative |
| Order, cone and residual | L64–L73 | reachability; projective typing; realization branches; conformal ray; Lorentz forcing; tame reduction; no finite linear classification; duality; hyperbolic lock; wild class |
| Scale and transport | L74–L80 | scale cocycle; strict transport; induced connection; entropy selects no connection; no law-natural connection; canonical connection under declared covariance |
| Duration and dynamics | L81–L89 | clock transition groupoid; affine clocks; path independence; coherence duration; noiseless finite histories; kernel-count entropy; Ruelle and Pesin locks; typed dynamical certificates; cone-field criterion |
| Chaos frontier | L90–L93 | frontier slope $1/\lambda$; duality $2\lambda$; quotient growth $h$; information cost of $\lambda/\ln 2$ bits per certified step; clock distortion classes |
| Physical distance | L94–L109 | no internal unit; counted channels; count loss; ratio scale; two-point invariant; metric proportionality; anchor budget; unit-distance rigidity; radar and rod distance; stationary slices and Sagnac holonomy; non-stationary no-go; media; ergodic transfer; identification theorem |
| Assembly | L110 | kernel–stabilizer theorem and compatibility equations |
| Global transport and closure | L111–L115 | global transport; dispatcher totality; composite soundness; window-indexed walls; maximal closure |
| Final remainder | L116–L123 | task residual and terminal sufficient refinement; the coupling is complement-relative; coupling stabilizer $\prod\mathrm O(m_j)\times\mathrm O(p-r)\times\mathrm O(q-r)$ (multiplicities, not rank, control symmetry); Global Structure Theorem (stack, twisted forms $\check H^1$, block-bundle equivalence, initial invariant, classifying space); consequences ($2^\kappa$ classes over $S^1$, Fisher remainder $[K_N]$); gluing class in $\check H^2$; forced wall monodromy; no new primitive |
| Local classification | L124–L128 | complete word-trace classifier of (R,N,C) modulo O(p)×O(q) on every stratum; simple-spectrum sign chart $H^1(G;\mathbb F_2)$; shorter descriptors and joint tasks; Lorentzian scalar invariants fail (vacuum plane wave), so curvature classification is observer-relative; classification selects no dynamics |
| Remainder across walls | L129–L131 | hidden-room descent (discrete chamber stabilizers: lifted wall rooms, link functions modulo a constant, holonomies); no constructible exit-path classification (continuous crossing moduli, functional link moduli, pointwise rooms insufficient); obstruction tower (transporter section; germ steps plus cell classes in $[S^{r-1},H_\tau]$; 3-cells unobstructed; base dimension ≤ 3 needs only π₀ and π₁) |
| Open | L132–L133 | enumeration of tame characteristic types (external); uniform forgetting for forced deterministic turbulence |

Each rung has exactly one status: **F** forced, **I** imported, **H** hypothesis-conditioned (the hypothesis is named), **Γ** conditional on the effective symmetry group, **N** no-go, or **O** open.

**Named hypotheses:**
- regularity and nuisance: `H-reg`, `H-nuis`, `H-sens`;
- declared groupoids: `H-sound`, `H-compl`, `H-coh`;
- cone, scale and anchors: `H-top`, `H-int`, `H-SI`, `H-med`, `H-val`;
- walls: `H-unif`, `H-hell`;
- task and latent data: `H-task`, `H-cov`;
- physical response: `H-resp` (the system moves by the minimum-P⁻¹ lift: overdamped bodies with P = mobility, free bodies with zero conserved momentum with P = inverse inertia);
- probability and uncertainty: `H-stat`, `H-exch`, `H-adm`;
- selection principle: `H-mono` (metrics may not increase under Markov morphisms; selects Fisher by Čencov);
- symmetry: `H-Γ`;
- final remainder:
  - `H-split`: the resolved/null splitting is declared, as $N=\ker J^{*}$;
  - `H-type`: the coupling rank and the canonical-correlation multiplicities are locally constant;
  - the $\check H^2$ gluing class additionally assumes an abelian automorphism band.

Each is stated in the paper together with the test that falsifies it.

---

## Frontier examples

These are worked results from the module batteries. Each is reproducible from its ZIP with fixed seeds and mutation canaries.

**Chaos is not hidden (L90–L93).** The boundary between LOCKED and ESCALATE forecasts is a straight line of slope $1/\lambda$ in the log of the declared precision. A protocol moves only its intercept. Recovered across five seeds:

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

**The final remainder (L116–L123) and its local classifier (L124–L128).** The question was whether what survives all local repairs is a new primitive or global structure. On constant-type strata it is global structure. The battery runs 18 attacks and 4 canaries with master seed 20261022; every canary is rejected, and fresh-process reruns give identical results.
- **Rigidity refuted:** $C=\tfrac12 I_3$ has rank 3 and $\dim\mathrm{Aut}=3$, more than $\dim\mathrm{Aut}=2$ at rank 1.
- **Dimension zero is not trivial:** distinct full-rank correlations give $\mathrm{Aut}\cong\mathbb Z_2^3$, which has dimension 0 but 8 elements.
- **Random sweeps miss strata:** 5000 Gaussian couplings find no counterexample, because the bad stratum has measure zero.
- **Local invariants are incomplete:** a Möbius RG family over $S^1$ has constant spectrum, constant canonical correlation and constant Aut, yet is globally nontrivial.
- **Contractible bases are trivial:** brute-force gauge orbits for $S_3$ give 1, 3 and 11 classes on a tree, a circle and a figure eight, as predicted.
- **A $-I$ triple defect is not by itself a gluing failure:**
  - on a filled triangle it can always be removed by re-choosing a transition;
  - on $\mathbb RP^2$ it cannot: the $\mathbb Z_4\to\mathbb Z_2$ lifting defect survives all $2^{15}$ lifts.
- **Forced wall monodromy:** around a codimension-2 collision of canonical correlations, the singular line bundle returns with sign $-1$; around a loop that does not enclose the wall it returns with $+1$.

See `RG - Final Remainder.pdf` and `RG_final_remainder_bundle.zip`.

**The wall–tap experiment (origin, L0–L115 on one system).** The experiment is declared as five cumulative acoustic protocols.
- **Saturation:** unresolved directions fall 5, 4, 3, 2, 2 by kernel intersection, and close only with length and gain anchors.
- **Hidden fold:** the rhythm channel hides an exact $\mathbb Z_2$ fold, $b \leftrightarrow 2d(x)-b$, on 56% of the domain. It is a wall inside the regular stratum, removed only by a new signed channel.
- **Matched wall:** specialization holds under Hellinger continuity, with Fisher rank 3 → 1 and Cramér–Rao bound $O(\varepsilon^{-2})$.
- **Medium:** wind turns echo distance into the acoustic Landau–Lifshitz metric.
- **Closure:** 3000 random requests per seed gave 0 false LOCKED; seven sabotaged certificates each produced false locks.

See `RG - Wall-Tap Experiment.pdf`.

**Distance versus angle (L93).** Perceived time can be distorted in three ways, and each preserves a different amount:
- **offset:** preserves differences;
- **rate:** preserves ratios and angles, not distances;
- **warp** (total rupture): preserves only order and counts.

Under memory loss, a remembered distance calibration degrades like $\sqrt T$ (measured exponent 0.494–0.510), while angles from delays measured together stay exact. See `RG - From the Wall to the Kernel.pdf`.

**Quantum task changes holonomy (L38–L43).** On the same qubit slice the SLD metric (estimation) has curvature +1, while the BKM metric (Stein testing) is negatively curved. Metric selection is task-relative, and so is observable transport. No noncontextual hidden values exist: 0 of 512 assignments satisfy the Peres–Mermin square, and Fine's criterion agrees with linear programming on 60/60 random boxes.

**Uncertainty is not a primitive (L49–L58).** Two priors merge on the quotient as $1/n$ and stay apart along the kernel (0.312 at every sample size). On a declared class of bounded variation, mixing locks with a sharp constant:

$$
\mathrm{TV}(T^n_\sharp\mu_f, T^n_\sharp\mu_g) \le 2^{-(n+2)}\,\mathrm{Var}(f-g),
$$

attained exactly at every $n$. Over a finite list of declarations nothing is certified. See `RG - Uncertainty Geometry.pdf` and `RG - Uncertainty Geometry - Forgetting Ladder.pdf`.

**Latent transport is declared (L79–L80).** Vertical shears preserve the law and map any connection to any other. Under a declared covariance, the Kalman/RTS connection is canonical. Its latent holonomy around a closed observed loop scales as the loop's area, and it depends on the declared covariance.

---

## Crossing into physics (real data)

RG does not claim assumption-free physical values; no framework can (Duhem–Quine; Le Cam, L56). It certifies a
**physical value conditional on a computed residual assumption set**, by agreement of independent routes.

- **`RG - Real Measurements.pdf`:** real GEONET dual-frequency GPS data.
  - One calibration is made on station A and frozen. It then serves station B, the baseline, uncertainty coverage and the swap.
  - Every RG layer is ablated: removing the Sagnac term gives 27.6 m, the relativistic clock term 8.7 m, frequency diversity 5.9 m, and the clock nuisance more than 10⁵ m.
- **`RG - Crossing into Physics.pdf`:** the consilience engine.
  - **Routes:** four routes to one baseline split into two contradicting clusters, so the status is CORRESPONDENCE.
  - **Tie-break:** it is resolved only by pre-registered tests of each cluster's distinctive assumption. The phase cluster's integer ambiguities pass (chance about 10⁻⁴); the code cluster's whiteness is falsified by constant per-satellite biases (z up to 9).
  - **Result:** 3335.3913 ± 0.0034 m, LOCKED conditional on five listed assumptions.
  - **Curvature:** physical-curvature effects are typed by resolution. Sagnac and the relativistic clock term are detected; the Shapiro double difference is a STRUCTURAL WALL at 3.5 mm.

- **`RG - Stress Tensor.pdf`:** the stress tensor as an RG target.
  - **Two routes:** a geometric route (the Einstein tensor of a metric resolved to second order) and a source route (field stresses, or the kinetic moments ∫ pᵃpᵇ f dΠ). Their agreement is the field equation, read as a consilience test.
  - **Exact results:**
    - the Schwarzschild exterior has curvature but T = 0;
    - Reissner–Nordström geometry equals the Maxwell stress, and a wrong charge is INCONSISTENT;
    - on FLRW, the radiation and dust routes agree with the geometric route;
    - a massless scalar and a stiff fluid give the same background (a source CORRESPONDENCE);
    - the radiation stress given only its energy is the set {S ⪰ 0, tr S = ε};
    - ℓ ≤ 2 angular harmonics are the minimal sufficient panel;
    - metric agreement does not control stress, so a C² certificate is required;
    - stress anisotropy is forgotten exactly as (1−s)ⁿ.
  - **Not claimed:** that GR does not need spacetime. The metric can be an RG output, but the field equation is imported (Lovelock).

- **`RG - Quantum Bridge.pdf` (print edition, supersedes the first edition):** the passage from quantum observations to geometry as a conditional, task-level construction.
  - **Exact certificate:** the final task is constant on the complete admissible fibre. Identifying every intermediate link is sufficient, not necessary; the first edition's weakest-link rule is demoted to a conservative sufficient rule.
  - **Repairs:**
    - raw second moments replace centred variances;
    - the quadrature proxy is separated from physical stress, with an explicit conserved cavity-mode stress;
    - the Gaussian width is ill-conditioned, not a structural wall;
    - the micrometre superposition is CONDITIONAL, not FORCED.
  - **Walks:**
    - Gaussian, forward and back;
    - backreaction (contraction theorem; Schrödinger–Poisson fixed point);
    - a fourth-moment source task;
    - a finite-shot, dephasing-aware entanglement witness with a negative certified bound (synthetic).
  - **Checks:** 24/24 checks, 8/8 mutations, a replay verifier, and an independent 10/10 audit.
  - **Open:** coupling selection, and full curved-background semiclassical closure.

- **`RG - Physical Curvature.pdf` (final edition):** RG doing physics with its own geometry, with no GR or QM imported.
  - **The object:** the canonical connection B = PJᵀ(JPJᵀ)⁻¹ (L80). It is a law of motion under H-resp, which holds exactly for overdamped bodies (P = mobility) and for free bodies with zero conserved momentum (P = inverse inertia).
  - **Geometry:**
    - holonomy is area times curvature, to leading order;
    - an exact apparatus gives 9.9995 µm;
    - three-sphere swimmer curvature (a/3)[L₁⁻² + L₂⁻² − (L₁+L₂)⁻²];
    - a falling-cat body reorients by −0.2748 rad;
    - reciprocal strokes cancel exactly (Purcell), while zero signed area does not imply zero holonomy.
  - **Duration:** T_min = ℓ/√W and E_min = ℓ²/T. Unloaded displacement is rate-independent, and the connection alone supplies no absolute clock.
  - **Load:** the response is affine, v = Bq̇ + (I−BJ)Pf_ext, giving a linear load law with a stall force. Nonzero momentum adds a drift.
  - **Evidence:**
    - original battery: 10 physical checks plus 1 status check;
    - passive-fluctuation calibration predicts active swimming within 0.15–6.4% over ten runs;
    - frozen holdout on 24 strokes: displacement 0.47%, duration 0.066%, stall residual 2.3×10⁻⁶;
    - 18/18 checks pass, canaries 10/10, and the independent audit passes 10/10.
  - **Not claimed:** spacetime curvature, laboratory validation, or priority over the gauge theory of deformable bodies.

- **`RG - Gravitation.pdf`:** gravity computed on RG's terms, with Einstein's equation selected rather than postulated.
  - **The stack:**
    - null data give the conformal class (cone stratum), and anchored clocks give its scale;
    - conservation gives the motion of matter (Geroch–Jang);
    - consilience under locality, metric-only, second order, identical divergence-freedom and D = 4 gives G + Λg = κT (Lovelock);
    - spherical vacuum is Schwarzschild (Birkhoff), and the rr equation excludes the clock-invisible γ-family.
  - **Countermodels:** the Ricci-only law is inconsistent; f(R) is fourth order; Gauss–Bonnet survives only in D ≥ 5; a pressure-blind source law is refuted by clocks.
  - **Calibrations:** κ = 8πG is provably only a unit anchor (TOV homology; exterior tasks need just m = GM/c²), and Λ is a window.
  - **Predictions:** calibrated from Kepler orbits alone, the stack predicts Mercury's perihelion advance (42.98″/cy), the solar deflection (1.751″), the GPS clock frequency (10.22999999543 MHz) and the Pound–Rebka shift (2.46×10⁻¹⁵).
  - **Checks:** the retained backend passes 25/25 (byte-identical rerun), and the stack battery passes 16/16 with canaries 4/4.

## Computational complexity

| Task | Cost |
|---|---|
| kernel of a finite law table | linear hashing, union–find |
| minimal sufficient panels | output can be $\binom{m}{\lfloor m/2\rfloor}$ (Shamir threshold system); a panel of size ≤ k is NP-complete (Minimum Test Collection) |
| stratification of a semialgebraic law | doubly exponential in dimension (cylindrical decomposition) |
| rank certificate | $O(d^3)$ per escalation level |
| coupling stabilizer and multiplicity type | $O((p+q)^3)$ (SVD) |
| Čech gluing on a finite atlas | $O(\sharp\text{triples}\cdot d^3)$; the obstruction is the cohomology class, not the raw defect |
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
3. **`RG - Final Remainder.pdf`**: what is left after every local repair, why it is global structure and not a primitive, and where that answer stops.
4. **`RG - From the Wall to the Kernel.pdf`**: the origin and progress account, including the Terror experiment made exact.
5. **Module papers**, listed below, each with a same-stem ZIP containing its hostile battery.
6. **Historic syntheses**: development record only.

**Statement-of-record rule.** Where a module paper and `Resolution Geometry.pdf` differ in scope, hypotheses or status, the ladder governs. **Citing:** cite the rung (for example "RG, L90"), not the module paper.

**Notation.** In the ladder, $\Gamma$ is the effective symmetry group $\mathrm{Stab}/\mathrm{Gau}$. The early characterization papers use $\Gamma$ for the admissibility protocol, which this README writes $\Pi$.

---

## The atlas (after the ladder)

The ladder states what RG proves. The atlas states how a concrete system is **charted**.

A layer fixes three things:
- the structure carried by charts;
- the compatibility condition on overlaps;
- the query it answers.

A certificate is the layer's answer for one realization. It is one of: CERTIFIED, OBSTRUCTED, SINGULAR, TWISTED, CONDITIONED, CORRESPONDENCE, or NOT TESTED. Certificates answer different queries and are never merged into one verdict.

| Layer | Structure | Content | Rungs |
|---|---|---|---|
| A1 | sets, equality of laws | effective gluing; descent iff every fibre graph is connected; conversion iff kernel inclusion (5,651 contracts) | L0–L5, L9, L12–L15 |
| A2 | submersion, tensors | tensors descend iff basic (horizontal and invariant) | L18–L20 |
| A3 | Fisher metric | radical = ker DF; tensorial chart change; chart-free distance | L28–L35, L59–L63 |
| A4 | strata, walls | specialization iff kernel inclusion; Cramér–Rao ∝ ε⁻²; forced monodromy | L21–L27, L37, L122 |
| A5 | symmetry | gauge trivial on Q; Γ-equivariant charts; slices at cone points | L6–L11, L25 |
| A6 | global classes | twisted forms H¹; Leray covers; H² gluing class | L111, L119–L122 |
| A7 | error budgets | chain drift ≤ kδ; certified holonomy; Lipschitz constant 1/ε near walls; finite-sample charts as confidence sets | L16, L44–L48, L114 |
| A8 | anchors | dilation fibre; counted channel fixes the metre | L94–L109 |
| A9 | registry | ten realizations, from a finite law map to the full computation loop | all |

## Companion papers (attached to the closed ladder)

| Companion | Attaches at | Content and status |
|---|---|---|
| `RG - Physical Path.pdf` | L28–L32, L119 | stress symbol, kinetic cone, energy conditions forced, defect charge conservation; source-remainder no-go for gravity |
| `RG - Gravity Duality.pdf` | L80, L102–L104 | Jacobi pencil (h, N); Newton fails exactly when structure is resolved |
| `RG - Light as Emission.pdf` | L102–L105 | projection formula; the mirror image as a quotient point |
| `RG - Cornering.pdf` | L13, L43, L49–L57, L94 | floating box as a resolution window; luck typed: relative, gated physical, ontological |
| `RG - Undermining Spacetime.pdf` | L2, L27, L119 | spacetime as a selected representative; permanent fibres |
| `RG - GR - Both Sides.pdf` | L19, L117–L119 | hidden versus null; Einstein operator selected; second order by H-min plus data |
| `RG - GR - Full Computation.pdf` | L12, L16, L47, L124 | tensors from probe histories; coupling tested on held-out data; first nonlinear term |
| `RG - Ladder Organization.pdf` | all | dependency maps, walls, gates, anti-ladders; audits A1–A3 proposed, not applied |
| `RG - Realization Atlas.pdf` | all (Section 22 of the print) | layers A1–A9: contract, theorem, witness and boundary per layer; registry of ten realizations with per-layer certificates; battery `rg_atlas_full.py`, which reruns layer 1 |
| `RG - Information Loss.pdf` | L0–L16, L129–L131 | contract-relative classification of information loss; star rooms K_v\∏K_v/S_i^∞ for compact gauges; uniform-limit criterion at isolated points |
| `RG - Physical Shortcut.pdf` | L64–L68, L102–L109 | anchored arrival advantage; transit sign law v_c = 1/√(2+γ); gravity train; no advance of light under NEC |
| `RG - Mesh Spacetime.pdf` | L111, L119, L131 | curvature as gluing defect of local systems; mesh closure under coverage, obstruction tower as completion |
| `RG - GR Without Spacetime.pdf` | L2, L19, L117–L119 | every GR prediction descends to local data, a selected law and relational readouts; spacetime is a representative |
| `RG - Spectral Identifiability.pdf` | L122, L129–L131 | the first complete instance of RG's information-loss rule: sharp eigenline moduli; exactly n−1 extra analyzers; exact noisy holonomy threshold max(π/N, π/2−2α) for every estimator; rank-m orientation holonomy; the continuous holonomy is not finitely identifiable; statistical rate; simulated coupled-oscillator readout |

GR does **not need spacetime** as an independent object: every prediction descends to local data, a selected law and relational readouts, and spacetime is a representative unique up to isometry (`RG - GR Without Spacetime.pdf`). Full GR is still **not** derived from RG. The field equation is selected (Lovelock, H-min) or measured, and its nonlinear completion is open beyond its first term.

## Module papers by stratum

Every battery registers naive and sharpened claims, uses fixed seeds, fails on non-finite evidence, and carries **mutation canaries**: a check that cannot fail is not counted. Batteries are falsification witnesses, not premises of proofs.

| Stratum | Rungs | Papers (same-stem ZIP = battery) |
|---|---|---|
| Foundation | all | `Resolution Geometry.pdf`; `RG - Final Locks.zip` |
| Groupoid and assembly | L0–L11, L110 | `RG - The Groupoid of Resolution.pdf`; `RG - Assembly at the Kernel.pdf`; `Assembly of Resolution Geometry.pdf`; `RG - Functorial Resolution Geometry.pdf`; `RG - Saturation.pdf`; `RG - Kernel pair root and limits.pdf`; `RG - Quotients.pdf`; `RG - Observation Laws.pdf` |
| Observable sector | L12–L17 | `RG - Observable Sector.pdf`; `RG - Source Automation.pdf` |
| Differential, strata, walls | L18–L27 | `RG - Stratified Specialization.pdf`; `RG - Made Visible.pdf`; `RG - Atlas Globalization.pdf`; `RG - Gluing Test.pdf` |
| Second order and quantum | L28–L43 | `RG - Quantum Metric Selection.pdf`; `RG - Fibrewise Characterization.pdf`; `RG - Fibrewise Characterization - QM.pdf`; `RG - Axioms.pdf`; `RG - Covariance Geometry.pdf`; `RG - Reconstruction Identifiability.pdf`; `RG - Blind Sector.pdf` |
| Ignorance | L44–L47 | `RG - Ignorance.pdf`; `RG - Observation Limit.pdf`; `RG - Null Resolution Depth.pdf`; `RG - Entropy.pdf` |
| Uncertainty and forgetting | L49–L58 | `RG - Uncertainty Geometry.pdf`; `RG - Uncertainty Geometry - Forgetting Ladder.pdf` |
| Statistical distance | L59–L63 | `RG - Distance and Angles.pdf` |
| Order, cone, residual | L64–L73 | `RG - Residual Characteristic Geometry.pdf`; `RG - Cone.pdf`; `RG - Cone - Closure.pdf`; `RG - GR Signature Emergence.pdf` |
| Scale and transport | L74–L80 | `RG - Canonical Transport.pdf`; `RG - Atlas Globalization.pdf` |
| Duration, dynamics, chaos | L81–L93 | `RG - Chaos Is Not Hidden.pdf`; `RG - Chaotic Experiments.pdf`; `RG - Time.pdf`; `RG - Operational Duration.pdf`; `RG - Operational Duration Boundary.pdf`; `RG - GR Time.pdf` |
| Physical distance | L94–L109 | `RG - Physical Distance.pdf`; `RG - Wall-Tap Delay.pdf` |
| Global transport and closure | L111–L115 | `RG - Maximal Closure.pdf`; `RG - Canonical Transport.pdf` |
| Final remainder, local classification, walls | L116–L131 | `RG - Final Remainder.pdf` (walls: `RG - Obstruction Tower.zip`) |
| Origin experiment | all | `RG - Wall-Tap Experiment.pdf`; `RG - From the Wall to the Kernel.pdf` |

---

## Physics interface

**Physical distance (L94–L109).** RG distance is dimensionless and experiment-relative, and no function of it alone is a physical distance. Suppose the following hold on a regular static slice:
- Euclidean motions are in $\Gamma$, with irreducible isotropy;
- a counted two-way timing channel with exact SI constants is part of the joint law;
- the medium is resolved;
- a held-out anchor is predicted.

Then

$$
\bar g_{\mathcal E} = \lambda_{\mathcal E}^{2}\, h, \qquad d_{\mathrm{phys}} = \frac{d_{RG}^{\mathcal E}}{\lambda_{\mathcal E}}, \qquad d_{\mathrm{rad}} = \frac{c}{2\,\Delta\nu_{\mathrm{Cs}}}\, n_{\mathrm{Cs}} .
$$

Stationary slices carry the Landau–Lifshitz radar metric, and loop synchronization there has Sagnac holonomy. Non-stationary slices carry no time-independent radar metric (no-go). The unit enters only through the counted channel.

**Duration (L81–L84).** Clock readouts with equal kernels form one transition groupoid. A stabilizer time flow makes the transitions affine. Path independence holds iff the kernel of the increment is contained in the kernel of the clock.

**Cone (L64–L73).** A regular quadratic cone gives a conformal ray. Lorentz isotropy in $\Gamma$ (dimension at least 2+1) forces the light cone. Otherwise the characteristic set falls into one of three classes:
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
| Basicness and projectability | automatic by natural descent of a fixed law-space structure | L19–L20 |
| Metric selection | descent does not select; Fisher selected under H-mono (Čencov) | L29 |
| Establishment of exact laws | exact law equality is not uniformly certifiable from finite samples; data enter through confidence correspondences | L48 |
| Walls | kernel upper semicontinuity; specialization iff kernel inclusion; normalizer descent; stratified category | L21–L27 |
| Distinguishability across walls | Hellinger (Le Cam) specialization; weak convergence insufficient | L37 |
| Fisher layer | natural; radical = observational vertical; does not determine covariance (relative to the declared complement); quotient dimension p+q+pq under the declared-metric contract; globally, the Fisher-relative remainder is the null-kernel bundle class $[K_N]$ | L28–L32, L120 |
| Quantum metric | Petz family gate; task lock; nonparametric identification; wall specialization iff f(0) > 0 | L38–L41 |
| Contextual quantum protocols | trace rule; no noncontextual hidden values; product law over contexts (Fine) | L42–L43 |
| Finite-data rank walls | never certified; full-rank or upper-bound certificates only | L47 |
| Primitive of uncertainty | indistinguishability; uncertainty splits along the quotient; kernel prior never forgotten | L49 |
| Probability, LAN, Monte Carlo | de Finetti; Gaussian propagation = LAN limit (fails across folds); Monte Carlo objective only on FORCED outputs | L50–L51, L57 |
| Forgetting of declarations | TV rigidity under bijections; no uniform forgetting without a class; sharp BV lock; NS contraction gate; Dobrushin; samples refute, operators lock | L52–L56 |
| Non-circularity | certificate over a declared admissible class | L57 |
| Non-dominated, infinite-dimensional, path-space | pairwise domination; Feldman–Hájek / Cameron–Martin geometry | L58 |
| Residual characteristic branch | tame reduction; no finite linear classification; duality; hyperbolic lock; wild class | L69–L73 |
| Latent connection | not law-natural; canonical under declared covariance (Kalman/RTS); curvature = latent holonomy; wall coherence | L79–L80 |
| Operational duration | clock groupoid; affine under time flow; path independence | L81–L83 |
| Noiseless chaos | stratified-regular finite histories; entropy from kernel counts; Ruelle and Pesin locks; typed certificates | L85–L88 |
| Nonuniform dynamical transport | global cone field iff uniform hyperbolicity (no-go otherwise); window-typed certificates | L89 |
| Chaos versus declarations | frontier law; duality; quotient growth; information cost | L90–L92 |
| Delay, distance, angle | offset, rate and warp classes; angles survive rate change, counts survive warp | L93 |
| Unit and physical distance | no internal unit; counted channel; anchors; identification theorem | L94–L109 |
| Non-static slices | stationary: Landau–Lifshitz plus Sagnac; non-stationary: no-go | L104–L106 |
| Assembly | kernel–stabilizer theorem | L110 |
| Global transport | observable atlas closed; latent transport canonical under declared data; closed is not trivial, since the atlas still has an isomorphism class (L119) | L111 |
| Closure | dispatcher totality; composite soundness ≤ Σαg; window-indexed walls; maximal closure with the exact correspondence | L112–L115 |
| Task residual | descent iff every fibre's output set is a singleton; terminal sufficient refinement, unique and natural; joint witnesses required | L116 |
| Coupling symmetry | stabilizer $\prod\mathrm O(m_j)\times\mathrm O(p-r)\times\mathrm O(q-r)$ under the covariance-only contract; multiplicities, not rank, control it | L117–L118 |
| Final remainder on constant-type strata | local moduli plus Euclidean block bundles; initial complete natural invariant; classifying space; $\check H^2$ gluing class; no new primitive | L119–L123 |
| Local classification of second-order data | word traces complete under O(p)×O(q) on every stratum; sign chart $H^1(G;\mathbb F_2)$ on simple strata; observer-relative classification for Lorentzian curvature | L124–L128 |
| Second-order remainder across walls | hidden-room descent (discrete stabilizers); no constructible classification; obstruction tower in general | L129–L131 |

### Open problems of the ladder

1. **Tame type enumeration (L132, external).** Enumerate the tame types of degree-d characteristic cones. For plane curves this is the first part of Hilbert's sixteenth problem (known for d ≤ 7). It is a problem of real algebraic geometry, not of RG.
2. **Forced deterministic turbulence (L133).** Above the Galerkin gate $G_N \ge 1$, no rung certifies uniform forgetting. TV is rigid, the contraction gate fails, and there is no noise and no expanding non-invertible transfer operator. With stochastic forcing, ergodicity is known only qualitatively (Hairer–Mattingly). An explicit uniform constant over a declared class is open.

The former third problem, the **second-order remainder across walls**, is closed by L129–L131. The conjectured constructible stack over exit paths does not exist: walls carry continuous crossing moduli and functional link moduli. The classification is an obstruction tower:
- **Germ steps:** lifted wall rooms and link functions.
- **Cell steps:** classes in homotopy groups of the block stabilizers.

What remains is classical homotopy computation, not RG.

### Declarations by design (not open problems)

- **Protocol, task and admissible class.** RG does not derive which question an observer must ask. This is a theorem, not a gap: there is no task-free equivariant selector (L13). Given a task, selection is exact (L12). Given a declared class, non-circularity is certified (L57).
- **Latent data.** A connection, prior or covariance is declared or reported as undetermined (L79–L80).
- **Anchors.** The unit enters only through counted channels and SI definitions (L94–L102).
- **Second-order contract.** Declaring only the covariance tensors and declaring the observation metric as well are different contracts, with different remainders. The resolved/null complement is declared (`H-split`).

### Research directions beyond the ladder

These are extensions with new content. They are not gaps in the ladder's proofs.

- **Observable stress tensor:** a sector map with uniqueness, conservation and gauge independence, keeping *not identifiable* distinct from *identified as zero*.
- **Physical versus protocol coupling:** attribution by intervention and independent instrumentation.
- **Higher-order geometry near discriminants:** curvature, normal holonomy, second fundamental form.
- **Classifications:** the pointwise orbit space under the declared-metric contract, $O(p)\times O(q)$, is now closed (L124). Still open:
  - its global bundle form;
  - minimal coordinates on every degenerate stratum;
  - the enumeration of flat observable classes by character data.
- **Nonlinear gravity:** the cubic bootstrap. Complete invariants give the ansatz basis, and conservation plus descent constrain it order by order.
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
11. **No intrinsic global class.** A nonzero global class belongs to a family over a topologically nontrivial base, not to RG. Over a contractible base it vanishes (L120).

---

## Novelty calibration

Every RG component reduces to established mathematics:

- **Quotients and symmetry:** equality-of-law quotients and Lie groupoids; Schur's lemma.
- **Statistics:** Fisher–Rao geometry and Čencov uniqueness; Le Cam, Blackwell and LAN; Bernstein–von Mises; de Finetti; Lehmann–Scheffé minimal sufficiency.
- **Quantum:** Petz, Gleason, Busch and Fine.
- **Ergodic theory and dynamics:** Oseledets, Pesin, Ruelle and Brudno; Lasota–Yorke and Dobrushin; Feldman–Hájek.
- **Geometry and characteristics:** o-minimal and semialgebraic stratification (Hardt, Collins); Gårding hyperbolicity.
- **Descent and global structure:**
  - stacks and nonabelian cohomology (Giraud), and twisted forms (Serre);
  - classifying spaces (Steenrod, Milnor);
  - canonical correlations (Hotelling);
  - Riesz spectral projections (Kato);
  - the von Neumann–Wigner crossing rule.
- **Physics and measurement:** Malament–Hawking reconstruction; Beckman–Quarles; Cauchy–Hölder ratio scales; Landau–Lifshitz radar metric; the SI definitions.
- **Complexity:** Garey–Johnson; Shamir.

RG claims no new primitive and no new branch of mathematics. On constant-type strata this is now a theorem (L123, under `H-split` and `H-type`), not only a remark. Its content is identification: every component of the remainder is a named classical object with a universal property. Across walls it is answered by an obstruction tower (L129–L131). Every ingredient is classical: one-sided limits of subspace configurations, and homotopy groups of orthogonal groups. RG's contribution has three parts:

- a protocol-explicit, dependency-typed assembly of these components into one geometry of partial observability;
- the kernel–stabilizer mechanism, which makes all modules share one symmetry group;
- explicit failure conditions for every step, each backed by a hostile battery with mutation canaries.

---

## Honesty ledger

The surviving framework depends on keeping negative results visible.

- **Kolmogorov turbulence from Fisher-rank loss:** retired.
- **Constants or π-factors from flexible kernels:** retired as reparametrization.
- **Emergent spacetime, gravity from Fisher geometry, nodes of presence, quantum or biological identifications:** not part of RG.
- **RG distance as physical distance:** refuted. Physical distance needs the identification theorem (L109).
- **Independent module gates:** superseded by natural descent (L19–L20).
- **"Natural, hence forced" (first rigidity paper):** withdrawn after an external critique. Descent fixes a structure once its law-space definition is fixed; selection needs a characterization (H-mono for Fisher) or a declaration. Also repaired:
  - the radical theorem now requires nondegeneracy on the image tangent (indefinite counterexample);
  - minimal panels need a finite index set;
  - Γ is the liftable, presentation-relative symmetry;
  - countermodels show non-omissibility, not necessity;
  - H-unif is split into a theorem part and a certification part.
  
  See `RG - Rigidity.pdf` (Conditional Rigidity).
- **"Presentation or thresholds control the output":** answered in `RG - Presentation and Thresholds.pdf`.
  - Observable results and the observable symmetry Γ_obs are presentation-invariant, while the liftable Γ is presentation-relative.
  - Thresholds are unavoidable (Le Cam) and act as a monotone filtration, reported as critical margins.
- **Real-data registrations:** several expectations failed on contact with real data, and all are kept on record in `RG - Real Measurements.pdf` and `RG - Crossing into Physics.pdf`.
  - a coverage gate was mis-specified;
  - calibration was expected to age within the hour, and did not;
  - a medium sign was expected positive, and came out negative (instrument biases);
  - all four routes were claimed to agree, and do not;
  - the "right" cluster was hard-coded, which is smuggling, and the engine caught it;
  - a tie-break test was used that had no power against a constant bias.
- **Resolved–null coupling as the primary law invariant:** cut. It vanishes identically under the whitening (Čencov) metric. It is the relative position of a declared protocol metric, and its efficiency theorem is data processing. The coupling coordinate also moves affinely under a change of complement, $L\mapsto L+K$ (L117).
- **Principal Aut(G)-bundle over a varying operating point:** cut. It exists only on level sets of varying invariants; the law-natural structure group is O(r)×O(q). Across a type change no single-model description exists on any neighbourhood (L122).
- **Local automorphisms as symmetries:** cut. Jet isotropy need not integrate to a law symmetry.
- **"Coupling rank controls symmetry dimension; full-rank coupling rigidifies to the trivial automorphism" (Lock Conclusion):** refuted.
  - Witness: $C=\tfrac12 I_3$ has rank 3 and $\dim\mathrm{Aut}=3$.
  - Distinct full-rank correlations give $\mathrm{Aut}\cong\mathbb Z_2^3$, not the trivial group.
  - The supporting 5000-sample sweep could not see the measure-zero stratum.
  - Replaced by the multiplicity formula (L118).
- **"RG admits no faithful functor into any single foundation":** withdrawn. Forgetful functors are faithful.
- **"The residual is a nonzero descent class; RG is a nontrivial bundle of foundations":** withdrawn. Over a contractible base the class vanishes. The remainder is classified instead by local moduli plus block bundles (L119–L120).
- **"Admissible edges with Čech defect −I signal a gluing failure":** sharpened.
  - When transitions are choices, the defect is an obstruction only through its $\check H^2$ class.
  - The class is removable on a filled triangle and not removable on $\mathbb RP^2$ (L121).
- **Holonomy norms as certificates:** sharpened. Matrix norms depend on the frame; the invariants are conjugacy classes and Σ-orthonormal norms (L121).
- **Naive nerves as gluing models:** sharpened. Čech computations need Leray covers; a non-Leray cover of $S^1$ gives the wrong $H^1$ (L121).
- **"Delay is the invariant":** corrected. Delays scale under rate change; ratios, angles and counts are the invariants.
- **CHI coherence volume:** audited.
  - Its bottleneck inequality is reversed: the harmonic mean is at least the minimum.
  - Its Fisher-rank seal is blind to misspecification.
  - Density cancels in the wind example.
  - The condition-number hazard is retained as a rank-wall indicator.
- **Monte Carlo as the source of uncertainty:** superseded. Monte Carlo is the integration engine for FORCED outputs, not the lock.
- **Forgetting over finite lists of declarations:** corrected. Locks are suprema over declared classes (L53–L57).
- **"Maximal correspondence":** corrected to the exact correspondence. It is the least sound set of jointly attainable outputs, with joint witnesses retained, because separate marginals lose correlation (L116).
- **Earlier wall–tap regular quotient (x, r, b):** corrected to (x, r, |b − d|), because of the Z₂ fold.
- **Earlier batteries:** several checks could not fail and are replaced by computed, canaried checks.
  - Literal `True` checks; class indices used as features; hand-set dispatcher booleans; vacuous invertible-law inclusion tests.
  - The chaos/coil attack, which passes on a non-chaotic surrogate.
- **"Certified luck expansion 3.5×" (luck v2):** retracted. It was an i.i.d. projection with an entropy-coded seed estimate. The final luck battery issues **no** physical certificate: 0 bits, with 8 named gates open.
- **Hidden-dynamics observer:** a sign error on sine modes, plus a Hankel matrix that flattened the output channels. Both were found by audit and corrected in `RG - Both Sides.pdf`. "Null" was reworded to "operationally null for the protocol".
- **Single joint learner in the full computation loop:** it would pass the coupling test trivially, because linear invariants of the training data propagate. It was replaced by separate curvature-only and stress-only learners.
- **Entrywise loop products at degenerate eigenvalues:** not gauge-invariant. Replaced by word traces (L124).
- **Minimal hidden-state recovery (blind leap):** not established. Validation-selected ranks were 8–10 for a 6-dimensional generator.
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
- `RG - Theorem Ladder.pdf`, `RG_-_Theorem_Ladder_Final.pdf` (the earlier ladders, L0–L62 and L0–L94);
- `RG - Theorem Ladder v2.pdf` (working ladder; its Stratum 7, R30–R44, is L116–L123 and L131 of the print edition; its wall conjecture is refuted by L130);
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

DOI: [10.5281/zenodo.23623363](https://doi.org/10.5281/zenodo.23623363)
Author: **Matěj Rada**
License: **CC BY-NC-ND 4.0**

Historic CTMT: DOI [10.5281/zenodo.18229539](https://doi.org/10.5281/zenodo.18229539) · OSF [10.17605/OSF.IO/RFE8N](https://osf.io/RFE8N/)

Counterexamples and attempts to break the theorems are welcome. A clean failure under the stated hypotheses is a contribution.
