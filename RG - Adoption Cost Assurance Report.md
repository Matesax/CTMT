# RG Adoption Cost and Assurance Report

## How to use a small, testable part of Resolution Geometry inside an ordinary laboratory, engineering or funding pipeline

**Version:** 1.0 (first public version)  
**Date:** 21 October 2026  
**Corpus:** Resolution Geometry (RG), DOI 10.5281/zenodo.23152797  
**Reading order:** `README.md`  
**Companion bundle:** `RG - Adoption Cost Assurance Report.zip`  
**Bundle contents:** hostile battery, cost tool, claims register  
**Random seeds:** 20261021, 1, 42, 777, 2026  
**Author:** Matěj Rada  
**Email:** MatejRada@email.cz  
**Licence:** CC BY-NC-ND 4.0

**Status of this document.** This is design guidance backed by a falsification battery. It is not an empirical study of RG adoption, it reports no measured savings, and it does not claim acceptance by any regulator, accreditation body or customer. Every claim the report relies on is listed in the claims register and attacked by at least one test in the bundle. Where an attack found a counterexample, the claim in this report is the corrected one.

---

## Reading guide

| Reader | Read | Time |
|---|---|---|
| Director, sponsor, funding officer, procurement | Sections 1, 2, 7 and 9 | about 20 minutes |
| Laboratory head, metrologist, model owner | Sections 3 to 8 | about 1 hour |
| Reviewer who wants to break the report | Section 11, then run the bundle | as long as it takes |

---

## 1. Summary for decision-makers

### 1.1 What RG offers, without jargon

Before a laboratory buys better precision, more repeats or a new instrument, it is worth asking a sharper question than "how large is our uncertainty?":

> Which differences can this measurement setup tell apart at all, which ones can it never tell apart however often it is repeated, and which ones are merely hidden by our current precision or our current model?

Resolution Geometry is a mathematical framework that answers exactly that question once the experiment and its rules are written down. Its core thesis, stated in the corpus, is that the experiment determines what can be distinguished, while the declared protocol determines what counts as admissible; given both, the resulting observable structure is fixed (`RG - Fibrewise Characterization.pdf`, `RG - Axioms.pdf`). RG does not claim new primitive mathematics. Each component reduces to established tools such as Fisher information, Schur complements, canonical correlations and Čech gluing (`RG - Elimination.pdf`). What it adds is a disciplined assembly of those tools with explicit failure conditions.

In practice, an RG check ends in one of seven explicit outcomes, including outcomes most pipelines are not built to give, such as "several answers fit the data equally well; here they all are" or "this cannot be decided with these data at the agreed budget". That is the core value for an adopter: **it stops money being spent on measurements that cannot change the answer, and it stops confident reports that the data do not support.**

### 1.2 What to adopt first

Adopt one small step, inserted into an existing method-validation or model-review pipeline, for one named decision:

1. Write down the measurement model with every influence quantity (most laboratories already have this for their uncertainty budget).
2. Run an **identifiability check**: search for situations that produce the same data, and classify every unresolved quantity as *structural* (no amount of data fixes it), *precision-limited* (more or better data fixes it) or *representation-limited* (a better model fixes it).
3. Check information about the target **in the declared units of the required tolerance**, after accounting for nuisances.
4. Report one of the seven outcomes with its validity window, then validate against an independent reference that was not used in development.

This is packages P1 and P2 in Section 9. It does not replace instruments, quality systems or models. It adds a pre-step and a stricter reporting rule.

### 1.3 Readiness and cost in one paragraph

The mathematical ingredients are established and in routine professional use. The RG assembly of them is assessed in Section 6 at **Technology Readiness Level 3** on the NASA ESTO software scale (proof of concept by analysis, synthetic batteries and archived real-data demonstrations with research code), with the status dispatcher a TRL 4 candidate in a synthetic laboratory environment. This is a self-assessment; an independent assessment is required before procurement. There is no historical record of RG adoption costs, so any cost estimate today is at best an AACE **Class 5** screening estimate, and public-sector appraisal guidance says to start from the upper-bound optimism-bias uplift until local evidence exists (Section 7). The bundle includes a cost tool that does this correctly.

### 1.4 When not to adopt

The corpus states its own boundary first: when a field probes one quantity, in one condition, and never combines measurements, independent probes suffice and RG adds nothing (`RG - Necessity.pdf`). RG becomes useful when one of three things is wanted: saying something about a quantity the instrument does not probe directly, combining measurements across conditions, sites or frames, or reproducing a result in another laboratory that uses a different valid frame. If none applies, keep the baseline (package P0) and record that decision.

### 1.5 What this report does not claim

- No measured cost saving, return on investment or accuracy improvement over any real baseline.
- No regulatory, accreditation, banking or medical-device acceptance of RG outputs.
- No claim that a P1 or P2 package is an "RG certification" of a laboratory or product.
- No claim that RG replaces the GUM, ISO/IEC 17025, ASME V&V 40, NASA-STD-7009B or model-risk guidance. Section 3 shows where it plugs into them.

---

## 2. RG in the vocabulary you already use

| RG term | Plain meaning | Where it already lives in your pipeline |
|---|---|---|
| Declared experiment and protocol | The measurement model, its noise model, and the rules about what counts as an acceptable result | Measurement model (JCGM GUM-6:2020); method description and validation plan (ISO/IEC 17025) |
| Observational quotient, kernel | The set of different situations that produce exactly the same distribution of data, so the data cannot tell them apart | Identifiability analysis; "confounded" parameters |
| Nuisance | An unknown you do not care about but must account for (gain, offset, temperature, path length) | Influence quantity (VIM, GUM) |
| Efficient information | How tightly the data constrain the target after the nuisances have been accounted for, in the target's units | Sensitivity coefficients and the Cramér–Rao bound; profile likelihood |
| Structural null (wall) | A direction no data, precision or model can resolve with this setup | "Not identifiable with this protocol" |
| Precision null | Information is present but below current precision or sample size | "Needs more data" |
| Representation null | Information is present in the data but the current model cannot express it | "Wrong model class" |
| Anchor | An independent reference that fixes a scale or a nuisance | Metrological traceability; reference standard; control sample |
| Held-out anchor | Validation against a reference not used in development | Method validation against an independent reference; comparator in ASME V&V 40 |
| Loop closure (cocycle) | Whether chained cross-calibrations come back to where they started | Interlaboratory comparison; round-robin closure |
| Status | The one outcome a request ends in | An extended decision-rule outcome (ISO/IEC 17025 decision rules, ILAC G8:09/2019, JCGM 106:2012) |

### 2.1 The seven outcomes

The seven statuses are defined in `Resolution Geometry.pdf`, Section 16. Their plain meaning:

| Status | Plain meaning | Usual report wording equivalent |
|---|---|---|
| LOCKED | Every required check passed in declared units; usable for the declared task inside its window | Conformity or result statement issued |
| CORRESPONDENCE | Several answers fit the data equally; all of them are returned | Set-valued result; conformity cannot be stated uniquely |
| ESCALATE | Undecided, but a specified next step remains within budget | Further testing required (named test) |
| STRUCTURAL WALL [ε] | Still undecided after the agreed budget, at the resolution ε reached | Cannot be decided with the available method |
| VACUOUS | No meaningful task survives the declared constraints | Request withdrawn or redefined |
| INCONSISTENT | A check is contradicted by the data | Nonconforming method or data |
| OUTSIDE DOMAIN | The request is outside the declared scope | Not within the scope of the method |

Whether an accreditation body or a customer accepts these labels must be agreed in advance. The practical route is to keep the laboratory's existing statement categories and record the RG status as the reason behind each statement.

---

## 3. The benchmarks this report uses

RG enters an organization through processes that already have trusted references. The report uses the following, and each row says what the benchmark contributes and where RG plugs in. These are external documents; their requirements are paraphrased, and readers must check the edition in force for their own jurisdiction.

| Benchmark | What it contributes here | Where RG plugs in |
|---|---|---|
| GAO Cost Estimating and Assessment Guide, GAO-20-195G (March 2020) | A reliable estimate is comprehensive, well documented, accurate and credible; a 12-step process; risk and uncertainty analysis; cross-checks | Section 7 uses its structure; the cost tool implements its risk-analysis step |
| AACE International RP 18R-97, cost estimate classification | Estimate classes by maturity of definition; Class 5 (concept screening) typically about −20% to −50% low and +30% to +100% high | Any RG adoption estimate today is Class 5; the tool warns when a simulated range is narrower than the class band |
| HM Treasury Green Book (2026 edition) and Supplementary Green Book Guidance: Optimism Bias | A clearly defined business-as-usual baseline; proportionate appraisal; no mechanistic ranking by one ratio; optimism-bias uplift for equipment and development projects with upper bounds of 200% (capital expenditure) and 54% (works duration), reduced only with evidence | Baseline P0; the cost tool's ceiling check; Section 7.6 on scoring |
| GAO Technology Readiness Assessment Guide, GAO-20-48G (January 2020); NASA ESTO TRL definitions with software descriptions | Nine readiness levels with exit criteria; independent assessment | Section 6 self-assessment and exit criteria |
| JCGM GUM suite: JCGM 100:2008, JCGM 101:2008, JCGM GUM-1:2023, JCGM GUM-6:2020, JCGM 106:2012; VIM JCGM 200:2012 | Measurement models, uncertainty propagation, conformity assessment, vocabulary | The identifiability check sits between model building (GUM-6) and uncertainty evaluation (JCGM 100/101) |
| ISO/IEC 17025:2017; ILAC G8:09/2019 | Method validation; documented decision rules that account for uncertainty when stating conformity | The seven outcomes become reasons behind decision-rule statements |
| ASME V&V 40-2018; NASA-STD-7009B (2024) | Credibility of computational models scaled to model risk and context of use | Package depth is chosen by decision consequence in the same spirit |
| Revised interagency Guidance on Model Risk Management (US banking agencies, 17 April 2026), which rescinded SR 11-7 | Model risk management proportionate to model materiality | Financial and funding profiles in Section 9; readers outside the US apply their own supervisory expectations |
| Decision theory: Le Cam bounds; value of information (Raiffa–Schlaifer); minimum test collection (NP-complete) | Floors on decision error; the most a measurement can be worth; hardness of minimal panel selection | Sections 7.2 and 7.5, tests T15, T21, T23 |

---

## 4. The holding structure: where each adoption claim stands on the Theorem Ladder

`RG - Theorem Ladder.pdf` orders the whole framework into thirty rungs (R0–R29) and types each one. The types matter to an adopter because they say how much local checking a claim needs:

- **forced**: follows from the declared observation map alone (only R1–R4 are forced: what can be told apart, and the structure that follows);
- **imported**: established mathematics reused (for example Fisher information as an inverse Schur complement);
- **gated**: valid only under a named hypothesis that must be checked locally (regularity, completeness, compatibility, integrability);
- **no-go**: a proved negative (something cannot be done);
- **open**: not established.

This report adds a sixth label, **external**, for claims that come from decision or cost practice rather than from RG. A business case leans on all six kinds; only the forced ones are free of local checking.

| Claim | Adoption claim (as corrected by the battery) | Ladder placement and type | Corpus source | Test |
|---|---|---|---|---|
| C01 | Repeating the same protocol never separates an exact ambiguity; only a different channel or an independent reference can | R1–R3 forced; R12–R13 gated | `RG - Observation Limit.pdf`, `RG - Wall-Tap Delay - Global.pdf` | T01, T02, T16 |
| C02 | Classify every unresolved quantity as structural, precision-limited or representation-limited before spending | R12–R13 forced typing | `RG - Null Resolution Depth.pdf` | T02, T07 |
| C03 | Silently fixing a nuisance narrows intervals and creates false confidence; certify with nuisance-efficient information | R8 imported; R12 gated | `RG - Wall-Tap Delay - Global.pdf` | T01, T08 |
| C04 | A locally well-conditioned fit can still have exact partner solutions; report a set when they exist | R10 gated; R2 forced | `RG - Wall-Tap Delay - Global.pdf` | T03, T04 |
| C05 | A register of known ambiguities is sound but can be incomplete; search for more | R5–R6 gated | `RG - The Groupoid of Resolution.pdf` | T04 |
| C06 | Exact grouping is cheap for exact model tables; with measured data it is a statistical test | R1–R3 forced; finite-data boundary | `Resolution Geometry.pdf` | T05 |
| C07 | A new channel never reduces efficient information on an existing target, but can add unresolved nuisances | R12 gated; R8 imported | `RG - Wall-Tap Delay - Global.pdf` | T06 |
| C08 | Agreement between channels sharing a calibration is not accuracy | R12–R13 | `RG - Observation Limit.pdf` | T07 |
| C09 | Rank and conditioning checks count only per target and in declared units | R11 gated; walls window-indexed | `RG - Hole Rejection.pdf` | T08 |
| C10 | Parameter information cannot validate the noise model | R8 imported; R9 no-go | `RG - Reconstruction Identifiability.pdf`, `RG - Covariance Geometry.pdf` | T09 |
| C11 | "Can measure it" does not rank instruments | R14 no-go | `RG - Theorem Ladder.pdf` | T10 |
| C12 | Passing pairwise cross-calibrations does not make a chain consistent; close every cycle | R23 gated; R25 imported | `RG - Necessity.pdf`, `RG - Atlas Globalization.pdf` | T11 |
| C13 | A holdout inspected during development is development evidence | external; procedure step 14 | `Resolution Geometry.pdf` | T12 |
| C14 | The false-acceptance budget is at most the sum of gate error rates, only for all-gates-required acceptance and as-used rates | union bound, `Resolution Geometry.pdf` Section 16 | `Resolution Geometry.pdf` | T13 |
| C15 | Tightening thresholds only removes accepted results when the data rule is frozen | `Resolution Geometry.pdf` Section 16 | `Resolution Geometry.pdf` | T14 |
| C16 | No analysis beats the data's distinguishability; sample cost scales as (noise/difference)² | imported (Le Cam) | `RG - Uncertainty Geometry.pdf` | T15 |
| C17 | Along a blind direction the result equals the prior assumption at any sample size | R2–R3 forced | `RG - Uncertainty Geometry.pdf`, `RG - Ignorance.pdf` | T16 |
| C18 | Never add most-likely costs; report P50 and P80; break-even is a distribution | external | — | T17, T18 |
| C19 | Evaluate packages in combination | external on R12 | `RG - Wall-Tap Delay - Global.pdf` | T19 |
| C20 | Apply floors, then show the nondominated set; no single weighted score | external | — | T20 |
| C21 | Do not buy a measurement worth less than its value of information | external on R12–R13 | `Resolution Geometry.pdf` | T21 |
| C22 | Each extra step of reliable forecast costs a fixed number of bits of initial precision | `Resolution Geometry.pdf` Sections 13 and 17 | `RG - Chaos Is Not Hidden.pdf` | T22 |
| C23 | Exact minimal panel selection is NP-complete | `Resolution Geometry.pdf` Section 17 | `Resolution Geometry.pdf` | T23 |

Three structural facts from the ladder are worth carrying into any adoption discussion. First, the forced root is universal for exact information and for nothing else; every higher structure needs its own gate. Second, the ladder names its open rungs (R14 Blackwell realizability, R21 global stratified compatibility, R27 the canonical connection) and the physical scale anchor as external; none of this report's recommendations depends on them. Third, absolute scale is outside the ladder by design: relative quantities can be forced, physical units always need an anchor.

---

## 5. Where RG enters a standard acceptance pipeline

The table follows the usual life of a laboratory method or a decision model. The RG additions in the middle column are the "small portion" worth adopting first; specialist modules come later and only on demand.

| Stage | Existing practice | RG addition | Deliverable | Backed by |
|---|---|---|---|---|
| 1. Purpose and context of use | Purpose, measurand and tolerance (ISO/IEC 17025); context of use and model risk (ASME V&V 40) | Name the single decision the result feeds and its consequence | Task contract | — |
| 2. Measurement model | Model and influence quantities (JCGM GUM-6) | Declare all nuisances explicitly, including those currently "fixed" | Influence-quantity register | T01, T06 |
| 3. Identifiability check (new) | Usually absent or informal | Equality-of-law search over the declared domain; classify every null as structural, precision or representation | Ambiguity map with null types | T02, T03, T04, T05 |
| 4. Uncertainty evaluation | JCGM 100 and 101 | Efficient information per target in declared units; check the full residual covariance; label any output that rests on a prior along a blind direction | Uncertainty budget plus wall report | T08, T09, T16 |
| 5. Validation | Validation against an independent reference; comparator (V&V 40) | Frozen held-out anchor; count the candidates that touched the holdout; close every calibration cycle | Validation record | T11, T12 |
| 6. Decision and statement | Documented decision rule (ISO/IEC 17025, ILAC G8, JCGM 106) | One of seven outcomes; all-gates-required acceptance; gate attempts logged | Status with window | T13, T14 |
| 7. Release, monitoring, change | Change control, drift monitoring | Invalidation triggers tied to the declared model, nuisances and window | Change record | T06, T11 |

### 5.1 The fourteen-step RG procedure, split into "adopt now" and "on demand"

`Resolution Geometry.pdf`, Section 19, defines a fourteen-step procedure. For a first adoption:

- **Adopt now:** step 1 (declare protocols and joint law), step 2 (kernel by equality-of-law tests), step 4 (regular strata and walls), step 5 (natural modules, here only the efficient information on the target), step 12 (uncertainty with declared admissible class), step 13 (dispatch with window), step 14 (held-out anchor).
- **On demand, with a specialist:** step 3 (symmetry generators), step 6 (conditional geometric structure), step 7 (counted timing and media), step 8 (specialization at walls), step 9 (dynamics), step 10 (quantum task), step 11 (latent transport).

### 5.2 The two-way cycle

Forward: task, alternatives, observation design, analysis, frozen validation, scoped result. Backward: a failed validation or an operational anomaly is attributed to one named cause and only that part is redesigned. The corpus lists the admissible causes: kernel misspecification, a wall, incomplete saturation, a smaller symmetry, a non-natural module, a failed anchor, medium or scale hypothesis, statistical rejection, or latent transport used without declared second-order data (`Resolution Geometry.pdf`, Section 19). A value loop runs beside it: excessive cost, low reuse or a changed business task returns to package selection. The value loop may change the task or the budget. It may not relabel failed evidence or relax a frozen threshold after the fact.

Organizational rollback (going back a stage) is allowed. It does not mean that information lost at a mathematical wall can be recovered.

---

## 6. Readiness assessment

The assessment below uses the NASA ESTO software descriptions of the TRL scale, which ask, for example, at TRL 3 for limited functionality validating critical properties with non-integrated components, at TRL 4 for integrated and functionally validated key components with a documented relevant environment, and at TRL 5 for end-to-end software interfaced with existing systems and tested in a relevant environment. It is a self-assessment. GAO-20-48G expects assessments by an independent team; this table is the input to such an assessment, not a substitute.

| Element | Evidence in the corpus | Self-assessed level | What moves it up |
|---|---|---|---|
| Imported ingredients (Fisher information and Cramér–Rao bound, Schur complements, GUM propagation, decision rules, Le Cam bounds, value of information) | Established mathematics and standards | In routine professional use; not assessed as RG technology | — |
| Equality-of-law search and kernel intersection across channels | Analytical results; synthetic batteries (`RG - Wall-Tap Delay - Global.pdf`, WT01–WT02); this bundle T04, T05 | TRL 3 | Package as one versioned component; define the relevant environment for a named laboratory task |
| Nuisance-efficient information and wall check in declared units | Synthetic batteries with sabotage canaries (WT08, WT16); archived real data: geomagnetic reference fields (`RG - Hole Rejection.pdf`), undergraduate optics (`RG - Admissible Observable Sectors in Optical Measurement Systems.pdf`), space weather and seismic catalogues (`RG - OMNI Necessity.pdf`, `RG - Seismic Necessity.pdf`), gravitational-wave and ECG records (`RG - GR Placement Bridge - Gauge-Aware Wobs H1-L1.pdf`, `RG - GR Placement Bridge - Gauge-Aware Wobs ECG.pdf`) | TRL 3, TRL 4 candidate | Documented agreement with predictions on a laboratory's own archived records |
| Seven-status dispatcher wired to real certifiers | WT16: 3000 random requests per seed over five seeds, 0 false LOCKED, all seven statuses reached; seven sabotaged certificates each produced false LOCKs and were caught | TRL 4 candidate in a synthetic laboratory environment | Relevant environment defined for a customer domain; interface to existing records |
| Loop-closure and atlas checks | `Global RG.md`; `RG - Necessity.pdf`; `RG - Atlas Globalization.pdf`; this bundle T11 | TRL 3 | Apply to a real multisite calibration chain |
| Specialist modules (quantum metric band, cone realization, transport, gravitation stack, chaotic forecast certificates) | Research papers and batteries, for example `RG - Gravitation.pdf`, `RG - Chaos Is Not Hidden.pdf` | TRL 2–3, research | Domain-specific pilots only |
| This adoption process as a whole | This report and bundle | TRL 2 (formulated, not piloted) | One completed pilot with recorded costs and outcomes |

**Exit criteria for the first pilot.** TRL 4: the "adopt now" steps of Section 5.1 run as one versioned tool on a laboratory's archived data for one named task, with the relevant environment written down and test performance agreeing with the predicted information and status. TRL 5: the same tool interfaced with existing records (calibration database or laboratory information system), run end-to-end on a frozen held-out set, meeting the predicted performance.

---

## 7. Cost estimation

### 7.1 What is known and what is not

There is no historical record of what RG adoption costs. GAO-20-195G treats estimates without historical data as resting on analogy or expert judgement, which must be documented, tested for sensitivity and cross-checked. AACE classifies estimates at this maturity as Class 5. The Green Book's supplementary guidance tells appraisers without robust evidence to start from the upper-bound optimism-bias uplift for equipment and development projects (200% on capital expenditure, 54% on duration) and reduce it only with documented mitigation. This report follows all three: it gives a method and a tool, not a price.

### 7.2 Where the cost actually sits

**Computation is cheap where most adopters will work, and expensive in three named places.** `Resolution Geometry.pdf`, Section 17, gives the complexity of each layer. Grouping an exact law table by hashing is near-linear; in this bundle a plain Python implementation grouped 500,000 rows in about half a second, with a measured scaling exponent of 1.26–1.32 across seeds (T05). A rank certificate costs O(d³) per escalation level; a loop-closure check is one product per loop. The three expensive places are: choosing a minimal set of measurements (NP-complete; T23 shows greedy selection needing 4 tests where 3 suffice), global stratification (doubly exponential in dimension), and chaotic forecasting (T22: each additional reliable forecast step costs one bit of initial precision for the logistic map, so ten more steps need about a thousand times better initial data). These costs sit where the theory reports them, not inside a declaration.

**One cost trap at the cheap layer.** The exact layer is cheap only for exact model tables. With measured data, "same distribution" becomes a statistical test, and tolerance-based grouping is not transitive (T05: A close to B and B close to C, yet A not close to C). Budget it as a test, not as bookkeeping.

**Labour, not computation, dominates.** In a P1–P2 pilot the work is task definition, retrieval of calibration and lineage records, validation and independent review.

**Acquisition cost follows a law.** No analysis can beat the data's distinguishability: the best possible decision error equals (1 − total variation distance)/2 (T15, agreement with simulation within 0.005 on every seed). The number of samples needed to detect a difference scales as (noise/difference)²: detecting a shift of 0.1 standard deviations with 5% error needs about 1,082 samples, and halving the detectable difference quadruples that. At a structural wall the distinguishability is zero and the error stays at one half for any budget. This is the most useful single number for an acquisition request: if a proposed purchase does not change the distinguishability of the decision quantity, its value is zero.

### 7.3 Estimating procedure for a pilot

Condensed from the GAO process to what a pilot needs:

1. State the purpose of the estimate and the decision it supports.
2. Define the business-as-usual baseline (P0) and at least one credible non-RG improvement option. The 2026 Green Book requires a clearly defined business-as-usual scenario so that the net effect of a proposal can be judged.
3. List the work items (the bundle's example input lists nine for P1–P2).
4. Give each item a low, most-likely and high value, with the basis written next to it.
5. Run a correlated Monte Carlo (`rg_cost_model.py`). Report P50 and P80. Never add most-likely values: in T18, ten skewed items whose most-likely values sum to 20 have a median total of 29.9, and the sum of most-likely values is met less than 0.1% of the time; correlation between items raises P80 from 32.9 to 37.6.
6. Compare the simulated range with the AACE class band. A Class 5 range narrower than about −20%/+30% needs justification.
7. Apply the optimism-bias uplift to the base estimate as a ceiling check, not on top of a Monte Carlo that already models the same risks.
8. Express benefits as distributions. A break-even number of avoided events computed at the average value per event is optimistic (T17: the naive figure is 7.3 events, the median 10.0 and the P80 19.6; with probability 0.65 more events than the naive figure are needed).
9. Document, have the estimate reviewed independently, and replace estimates with actual figures after the pilot. The pilot's recorded costs are the first historical data for the next estimate class.

### 7.4 Worked example with illustrative numbers

The bundle's example input describes a P1–P2 pilot on one measurement decision. **The numbers are placeholders chosen to demonstrate the method; they are not measured, benchmarked or recommended.** With nine items in person-days, within-group correlation 0.6 and between-group correlation 0.2, the tool reports:

| Output | Value (illustrative) |
|---|---|
| Sum of most-likely values | 38.0 person-days, met with probability 0.05 |
| P10 / P50 / P80 / P90 | 41.4 / 57.3 / 69.8 / 76.4 person-days |
| Optimism-bias ceiling (base × 3.0) | 114 person-days |
| Simulated range around P50 | −28% / +33%, inside the Class 5 band |
| Break-even avoided repeat acquisitions | naive 7.1; P50 9.5; P80 19.0 |

The example's lesson is the shape, not the numbers: a planner who adds the most-likely values presents 38 person-days and is very likely to overrun; a planner who reports P50 and P80 presents 57 and 70 and an explicit ceiling.

### 7.5 Buying measurements: the value-of-information rule

Never pay more for a measurement than the most it could ever be worth to the decision. Three cases from T21:

- If the decision is the same across the whole plausible range of the quantity, the value of any further information is zero. Stop.
- If the plausible range straddles the decision threshold, perfect information is worth about 0.40 of the quantity's prior standard deviation (in the decision's units) and a realistic noisy measurement about 0.36. That is the budget ceiling for the measurement.
- A measurement that is blind to the decision quantity is worth exactly zero however precise it is. In T21 a measurement of t1 + t2 with noise 0.1 is worthless for a decision on t1 − t2.

The third case is where RG pays for itself in procurement: the identifiability check of Section 5 tells you, before purchase, whether the proposed measurement is blind to the decision quantity.

The rule "buy the missing information, not more measurements" holds only after the null has been classified (T02). For a precision-limited null, more repeats do help, and ten times the data cuts the variance bound exactly tenfold. For a structural null, nothing but a new channel or reference helps. For a representation-limited null, a better model helps with no new data at all (in T02 a linear model reads a quadratic signal as noise, R² 0.000, while a quadratic model recovers it, R² 0.999).

### 7.6 Choosing between packages

Two practices that look reasonable fail on explicit counterexamples:

- **Ranking packages one by one by benefit/cost.** In T19 the target is a physical distance from echo delays. A thermometer alone and a counted clock alone are each worthless for distance; together they are the cheapest fix (cost 4, against 5 for a rod). A one-by-one ranking with a budget of 4 buys nothing useful. Small candidate sets should be enumerated in combination.
- **A single weighted score.** In T20 an option meeting both the cost floor and the assurance floor is nondominated but is never selected by any of 1,001 weightings of cost against assurance. Apply floors first, then present the whole nondominated set. This is consistent with the 2026 Green Book, which no longer treats a single benefit–cost ratio as a pass/fail test.

### 7.7 Funding gates

| Funding decision | Evidence to request | Reasons not to proceed |
|---|---|---|
| Scoped pilot | Named decision, current failure mode, baseline, capped budget, an independent reference that exists | No decision consumer; no measurable gap; no independent evaluation route |
| Deeper analysis | The pilot found a material ambiguity or an uncertainty defect | Relabelling only; no change to what the data support |
| New instrument or reference | A specific channel predicted to make the decision quantity distinguishable; value of information above cost | More precision along a direction already resolved; a shared calibration left unaddressed |
| Integration into operations | Frozen validation met the agreed floor; positive case at P80 | Integration cost above demonstrated reuse; unresolved release risk |
| Reuse across sites | Several validated consumers share a genuinely compatible component; calibration cycles closed | Reuse needs incompatible assumptions or a silently widened domain |

---

## 8. Assurance: identify the defect that actually limits the result

| Defect | Diagnostic | Remedy | Misleading substitute | Test |
|---|---|---|---|---|
| Random noise | Repeated independent-reference residuals | Replication or better acquisition | More output digits | T02, T15 |
| Shared systematic error | Independent reference or mechanism-specific cross-check | Different reference chain | Agreement between channels sharing a calibration (T07: 64 such channels agree to 0.10 while the error stays 0.50) | T07 |
| Exact ambiguity (product, fold) | Equality-of-law search; null classification | Different protocol, independent reference, or a set-valued result | More repeats of the same protocol | T01, T03, T04 |
| Silently fixed nuisance | Efficient information with the nuisance free | Declare and estimate or anchor the nuisance | A narrower interval (T01: a gain fixed 1% off gives 95% intervals that cover the truth 91% of the time at 10 repeats, 55% at 100 and never at 10,000) | T01 |
| Weak information in declared units | Per-target information against the required tolerance | Better geometry, operating restriction or wider uncertainty | A good condition number or high R² (T08: R² 0.98 with the target entirely unsupported) | T08 |
| Wrong noise model | Full residual covariance against the declared one | Revise the noise model | Agreement of parameter uncertainties (T09: two different noise models with identical information matrices) | T09 |
| Inconsistent calibration chain | Loop closure around every cycle | Global adjustment or restricted scope | Passing pairwise checks (T11: every step agrees to 0.999998; the loop returns rotated by 1.14 rad) | T11 |
| Assumption standing in for data | Check whether the output moves with the prior | Label as assumption-driven, or acquire an anchor | Large sample size (T16: posterior width unchanged from 1,000 to 10 million samples; shifting the prior by 0.5 shifts the answer by 0.5) | T16 |
| Leakage and reuse of holdouts | Count of candidates evaluated on the holdout | Fresh evaluation source | A good holdout score (T12: best of 100 skill-free models scores 0.59 on a reused holdout, 0.50 on a fresh one) | T12 |
| Mis-specified acceptance logic | All-gates-required acceptance; gate attempts logged | Fix the dispatcher | A majority vote of checks (T13: false acceptance 0.37 against a budget of 0.10) or retrying a check until it passes (0.096 against a claimed 0.02) | T13, T14 |

An RG analysis does not necessarily repair the defect. Its first useful output may be a defensible diagnosis and a refusal to report a unique answer. An honest analysis can widen uncertainty or increase abstention while improving assurance; report that as an assurance gain, not as an accuracy gain.

Two corrections to common practice deserve emphasis. First, certificates must be stated in declared precision units: a scale-free (normalized) rank test is blind to a wall where all information collapses uniformly, which is exactly how the first closure run in `RG - Wall-Tap Delay - Global.pdf` produced false LOCKs before the certificate was corrected. Second, instruments that "can both measure it" are not interchangeable: two instruments that identify a quantity equally can have Bayes errors of 0.10 and 0.30 (T10); compare instruments by decision risk or by efficient information in declared units.

---

## 9. Adoption packages, stages and organizational profiles

### 9.1 Packages

Packages are composable choices, not a maturity ladder every task must climb, and they are not RG statuses or certification grades.

| Package | Contents | Suitable use | Explicit limit |
|---|---|---|---|
| P0 Baseline | Record the current task, costs and assurance; keep the workflow | Low-consequence tasks already meeting requirements | No claim of RG use |
| P1 Evidence wrapper | Task contract, influence-quantity register, lineage, status and window, reproducible baseline | First pilot; cross-team handoff | Does not establish any ambiguity map |
| P2 Diagnostic check | P1 plus equality-of-law search, null classification, efficient information in units, full residual check, frozen held-out comparison | Inverse problems, sensor fusion, recurring model disputes, model review | Results are scoped to the declared domain and model |
| P3 Targeted protocol change | P2 plus a new reference, independent channel, changed excitation or geometry | A material structural ambiguity that reanalysis cannot resolve | The new channel must change distinguishability of the decision quantity (value of information above cost) |
| P4 Operational integration | Selected P1–P3 artifacts in release, monitoring, invalidation and fallback | Repeated consequential decisions | Automation does not turn assumptions into evidence |
| P5 Specialist module | Symmetry, stratification, transport, quantum, dynamics or gravitation modules with specialist review | Research or critical tasks that genuinely need them | Not an enterprise default |

### 9.2 Stage plan

Stages are triggered by deliverables, not by dates.

| Stage | Work | Go condition | Stop or reverse |
|---|---|---|---|
| E0 Select and baseline | Choose one decision; inventory evidence and costs | A plausible value question exists; consequence is explicit | Keep P0 if no material gap |
| E1 Wrap | P1 | Every artifact has a named consumer | Remove unused forms |
| E2 Diagnose | P2 offline | A decision-relevant defect is found, or the baseline is shown sufficient (both are successes) | Stop if the baseline suffices |
| E3 Design evidence | Compare reanalysis, task restriction and acquisition by value of information | A package exceeds its cost at P80 | Restrict the claim or keep the set-valued result |
| E4 Frozen validation | Independent reference; fresh holdout | Agreed floor met; costs measured | Reject, revise as a new version, or abstain |
| E5 Controlled integration | Interfaces, monitoring, fallback | Release authority accepts scope and rollback | Return to offline use |
| E6 Selective reuse | Only compatible components; cycles closed | Reuse benefit remains positive | Split incompatible scopes |

**Stopping rule.** Stop expanding when the current package meets the required tolerance and assurance floor and no further package has a value of information above its full cost at P80. If the floor cannot be met, restrict the claim, change the task, acquire different evidence or abstain. Never lower the floor to produce a positive business case.

### 9.3 Organizational profiles

**Lean laboratory or engineering team.** One task owner, one combined modelling and data role, and an independent reviewer for consequential claims. Start with P1 and a focused P2 check on the measurement that most often triggers repeat acquisitions. Avoid a new committee or platform before a task benefit is shown.

**Large or multisite organization.** Central ownership of status semantics, schemas and shared validation utilities; local ownership of physical assumptions, tolerances and calibrations. Before sharing any certificate across sites, close every calibration cycle (T11). Count platform work separately from scientific validation.

**Public, procedure-heavy or externally reviewed institution.** Map each new artifact to an existing required evidence item and keep all existing approvals. Start with P1 evidence mapping and a narrow pilot with no change to authorized outputs. The 2026 Green Book's emphasis on proportionality and a clearly defined business-as-usual baseline fits this pattern directly.

**Financial institution, insurer or funding body.** The decision owner and model developer define the task; independent validation challenges alternatives and leakage; finance evaluates lifecycle economics; the authorized compliance function decides which obligations apply. In the United States the supervisory reference for banks is the revised interagency model-risk guidance of 17 April 2026, which replaced SR 11-7 and emphasizes proportionality to model materiality; other jurisdictions apply their own expectations. Use P2 to report the range of the decision-relevant output over a declared set of competing models, and P4 only for recurring consequential outputs. Physical timing, cone or gravitation modules have no role in a financial model.

**Sensitive equipment and high-consequence physical systems.** Metrology and domain specialists own physical adequacy; controls and operations own safe integration; independent validation challenges references and common-cause errors. Use P2–P3 on the limiting measurement defect and P4 only after controlled validation. No RG output is permission to retune an instrument, drive an actuator or relax an interlock.

---

## 10. Worked examples

### 10.1 Polarimetry: a product ambiguity every laboratory already handles

Optical rotation follows α = k L c (specific rotation, path length, concentration). From the rotation alone, the three unknowns have a two-dimensional exact null: only the product is identified. `RG - Admissible Observable Sectors in Optical Measurement Systems.pdf` finds exactly this two-dimensional null in public undergraduate optics data, removed only by external controls; T01 reproduces it. Laboratories already solve it with a calibrated k and a measured cell length. The RG contribution is that the same check runs automatically on models where the product structure is not obvious, and that it refuses to report c if a control is missing.

### 10.2 Echo amplitude: gain and reflectivity

In the wall–tap protocol, the echo amplitude depends on gain times reflectivity. Without a reference reflector the information on reflectivity is zero; with one it is restored (`RG - Wall-Tap Delay - Global.pdf`, WT08; in T01 the reference restores 76% of the information that a perfectly known gain would give). A certificate that silently fixes the gain produces false LOCKs. Before buying a lower-noise amplitude channel, ask whether the decision needs absolute reflectivity or only the product. If the product suffices, keep the narrow task. If absolute reflectivity is needed, buy the independent gain reference, not repeats.

### 10.3 A multisite calibration chain

Three or more sites cross-calibrate pairwise. Every pairwise comparison passes. `RG - Necessity.pdf` shows that when the observation geometry is curved, a reference carried around the cycle returns changed by 37–84% even though every adjacent check agrees to 0.9999; T11 reproduces the effect (adjacent agreement 0.999998, loop defect 1.13–1.24 rad across seeds) and a flat control that closes exactly. The adoption step is cheap: close every cycle and record its defect before treating the chain as one calibration.

### 10.4 A funding decision on a model-based proposal

A proposal's predicted benefit comes from one fitted model. A P2 review reports the range of the benefit over a declared set of plausible models, checks whether any part of the predicted benefit rests on a prior along a direction the data cannot see (T16), and computes the value of the one observation that would most change the funding decision (T21). If the range crosses the funding threshold and no affordable observation narrows it, stage or decline the commitment rather than manufacture a point estimate. The gain to demonstrate is better decision support or lower review effort, not a narrower interval.

---

## 11. The adversarial battery: what it tested and what it changed

### 11.1 Design

The bundle contains 23 attacks (T01–T23) and four meta-tests (M01–M04). Each attack first targets the claim as drafted and must find a counterexample if one exists; otherwise the attack has no teeth and the test fails. It then checks the corrected claim on the same seed. Verdicts:

- **HOLDS:** an identity or floor; no counterexample can exist, verified numerically.
- **CONFIRMED:** the draft claim survived; the battery exhibits and quantifies the counterexample to the practice the claim warns against.
- **SHARPENED:** the draft claim was too broad, incomplete or missing; an explicit counterexample forced the corrected claim used in this report.

Meta-tests attack the report itself: every claim in the register must be backed by a test that passed on every seed (M01); every corpus file cited here must exist in the corpus manifest, every test must be referenced, and stale or over-claiming wording must be absent (M02); results must be reproducible under a fixed seed (M03); every SHARPENED or CONFIRMED verdict must carry a counterexample (M04).

A separate mutation check attacks the battery itself. It deliberately breaks four core ingredients the way a careless implementation would: efficient information computed without accounting for nuisances, an identifiability test that always says yes, chained alignments treated as exact, and a rank test without declared units. Every mutant is caught (by T01, T19, T11 and T19 respectively).

### 11.2 Results

All 23 attacks passed on all five seeds (115 of 115), and all four meta-tests passed. Results below are for the primary seed 20261021; cross-seed ranges are quoted where they matter.

| Test | What was attacked | Verdict | Counterexample or measurement |
|---|---|---|---|
| T01 | Product ambiguity, repeats, silent nuisance | CONFIRMED | Gain fixed 1% off: coverage 0.91 at N=10, 0.55 at N=100, 0.00 at N=10,000 while intervals narrowed 32×; silent certifier LOCKED, honest one returned CORRESPONDENCE |
| T02 | "Buy information, not measurements" | SHARPENED | Precision null: 10× data gives exactly 10× smaller variance bound; representation null: R² 0.000 vs 0.999 |
| T03 | Full rank versus identification | CONFIRMED | Local 95% intervals cover 0.24–0.27; set-valued report covers 0.94–0.97 |
| T04 | Completeness of the ambiguity register | SHARPENED | Declared 101 classes; true 73; search found all 28 hidden pairs |
| T05 | "The exact layer is cheap" | SHARPENED | True for exact tables (exponent 1.26–1.32); with estimated laws, equality is a non-transitive test |
| T06 | Adding channels | SHARPENED | Nullity never rose on fixed parameters (0 of 2,000) and efficient information never fell (0 of 2,000); a channel with two new nuisances raised nullity 0→1 |
| T07 | Correlated channels | CONFIRMED | 64 channels sharing a calibration: spread 0.10, error 0.50; independent calibrations: error 0.063 |
| T08 | Conditioning margins | SHARPENED | Normalized condition number 2.47 at every scale while the uncertainty bound reached 5.5 against a 0.1 target; R² 0.98 with an unsupported target |
| T09 | Uncertainty check via information | SHARPENED | Two noise models, identical information (difference 6×10⁻¹⁷); full residuals separate them (z ≈ 30–35) |
| T10 | Instrument comparison | SHARPENED | Equal identifiability, Bayes error 0.10 vs 0.30 |
| T11 | Copying certificates across sites | SHARPENED | Adjacent agreement 0.999998; loop defect 1.14 rad; flat control 0 |
| T12 | Holdout reuse | CONFIRMED | Best of 100 skill-free models: 0.59 reused, 0.50 fresh; best of 1,000: 0.61 |
| T13 | False-acceptance budget | SHARPENED | All-gates acceptance within budget 0.10 (0.009; correlated 0.016); majority vote 0.37; retry-until-pass 0.096 against a claimed 0.02 |
| T14 | Threshold monotonicity | SHARPENED | 0 violations with fixed data; a stringency-dependent trimming rule fails at 1.5 and passes at 3.0 |
| T15 | Decision-error floor and sample cost | HOLDS | Simulation within 0.005 of (1 − TV)/2; halving the difference ×4 samples; error 0.50 at a wall |
| T16 | Priors along blind directions | SHARPENED | Posterior width 0.200 at N=10³ and at N=10⁷; answer shifts one-for-one with the prior |
| T17 | Break-even formula | SHARPENED | Naive 7.3 events; median 10.0; P80 19.6 |
| T18 | Adding cost items | SHARPENED | Sum of most-likely 20; P50 29.9; P80 37.6 with correlation |
| T19 | One-by-one package ranking | CONFIRMED | Budget 4: ranking buys nothing useful; the jointly optimal pair costs 4 |
| T20 | Single weighted score | CONFIRMED | The only option meeting both floors is never chosen by 1,001 weightings |
| T21 | Stopping rule | CONFIRMED | Settled decision: value 0; straddling: 0.40 (perfect) and 0.36 (noisy); blind measurement: 0 |
| T22 | Chaotic forecasting cost | CONFIRMED | 1.00 forecast steps per bit of initial precision; a float64 reference trajectory itself fails at step 52 |
| T23 | Minimal measurement panel | CONFIRMED | Greedy 4 tests; exhaustive optimum 3 |

### 11.3 Running and extending the battery

```
pip install numpy scipy mpmath
python run_battery.py              # five seeds, all attacks, meta-tests
python run_battery.py --quick      # primary seed only
python mutation_check.py           # the battery must fail on every deliberately broken mutant
python rg_cost_model.py inputs/example_pilot_inputs.json --out results/cost_summary_example.md
```

Results are written to `results/battery_results.json` and `results/summary.md`. To attack a local claim, add a function to `rg_attacks.py` that returns the same record format, register it in `ALL_TESTS`, and add the claim to `claims_register.json`. A clean failure under the stated hypotheses is a contribution; please report it.

---

## 12. Non-claims and open items

- The battery tests implementations, assumptions and the boundaries of claims. It does not prove theorems; theorem status comes from the corpus proofs.
- No result here establishes economic benefit. Economic benefit requires a pilot against a real baseline.
- The readiness levels are self-assessed and must be confirmed independently.
- The seven statuses are not recognized by any external body; their mapping to decision-rule statements must be agreed with the relevant accreditation body or customer.
- Open rungs of the ladder (R14, R21, R27) and the physical scale anchor remain open or external. Nothing in packages P0–P4 depends on them.
- Evidence still needed to turn this design into an empirical adoption report: real baselines and costs, named tolerances and independent references, the applicable acceptance requirements, pilot results against credible non-RG alternatives, and lifecycle maintenance records.

---

## Appendix A. Glossary

- **Anchor:** an independent reference fixing a scale or nuisance.
- **Efficient information:** information about the target after nuisances have been accounted for; its inverse bounds the achievable variance.
- **Equality-of-law search:** a search for different situations that produce the same distribution of data.
- **Held-out anchor:** an independent reference used only for final validation.
- **Loop closure:** the check that chained cross-calibrations return to their starting point.
- **Nuisance:** an unknown that matters for the data but not for the decision.
- **Null (structural, precision, representation):** a direction the analysis cannot currently resolve, typed by what would resolve it.
- **P50, P80:** the values that the total cost stays below with 50% and 80% probability.
- **Value of information:** the most a measurement could be worth to a specific decision.
- **Wall:** a place where information about some direction drops to zero; reported with the resolution at which it was reached.

## Appendix B. References

**Corpus (cited by standard title).** `RG - Corpus.md`; `RG - Theorem Ladder.pdf`; `Resolution Geometry.pdf`; `RG - Axioms.pdf`; `RG - Fibrewise Characterization.pdf`; `RG - Elimination.pdf`; `RG - Made Visible.pdf`; `RG - Observation Limit.pdf`; `RG - Null Resolution Depth.pdf`; `RG - Ignorance.pdf`; `RG - Uncertainty Geometry.pdf`; `RG - Reconstruction Identifiability.pdf`; `RG - Covariance Geometry.pdf`; `RG - Blind Sector.pdf`; `RG - The Groupoid of Resolution.pdf`; `RG - Atlas Globalization.pdf`; `Global RG.md`; `RG - Necessity.pdf`; `RG - Hole Rejection.pdf`; `RG - Wall-Tap Delay.pdf`; `RG - Wall-Tap Delay - Global.pdf`; `RG - Admissible Observable Sectors in Optical Measurement Systems.pdf`; `RG - OMNI Necessity.pdf`; `RG - Seismic Necessity.pdf`; `RG - GR Placement Bridge - Gauge-Aware Wobs H1-L1.pdf`; `RG - GR Placement Bridge - Gauge-Aware Wobs ECG.pdf`; `RG - Chaos Is Not Hidden.pdf`; `RG - Gravitation.pdf`.

**External.**
- U.S. Government Accountability Office, *Cost Estimating and Assessment Guide*, GAO-20-195G, March 2020. https://www.gao.gov/products/GAO-20-195G
- U.S. Government Accountability Office, *Technology Readiness Assessment Guide*, GAO-20-48G, January 2020.
- NASA Earth Science Technology Office, *Technology Readiness Levels* (hardware and software descriptions). https://esto.nasa.gov/trl/
- AACE International, Recommended Practice 18R-97, *Cost Estimate Classification System – As Applied in Engineering, Procurement, and Construction for the Process Industries*.
- HM Treasury, *The Green Book* (2026 edition); *Supplementary Green Book Guidance: Optimism Bias*. https://www.gov.uk/government/publications/green-book-supplementary-guidance-optimism-bias
- Joint Committee for Guides in Metrology: JCGM 100:2008; JCGM 101:2008; JCGM GUM-1:2023; JCGM GUM-6:2020; JCGM 106:2012; JCGM 200:2012. https://www.bipm.org/en/publications/guides
- ISO/IEC 17025:2017, *General requirements for the competence of testing and calibration laboratories* (check for a later edition at the time of use); ILAC G8:09/2019, *Guidelines on Decision Rules and Statements of Conformity*.
- ASME V&V 40-2018, *Assessing Credibility of Computational Modeling through Verification and Validation: Application to Medical Devices*; NASA-STD-7009B, *Standard for Models and Simulations* (2024).
- OCC, Board of Governors of the Federal Reserve System and FDIC, *Revised Guidance on Model Risk Management*, 17 April 2026 (OCC Bulletin 2026-13; SR 26-2), rescinding SR 11-7.
- L. Le Cam, *Asymptotic Methods in Statistical Decision Theory*, Springer, 1986.
- H. Raiffa and R. Schlaifer, *Applied Statistical Decision Theory*, Harvard, 1961.
- M. R. Garey and D. S. Johnson, *Computers and Intractability*, Freeman, 1979 (Minimum Test Collection).
