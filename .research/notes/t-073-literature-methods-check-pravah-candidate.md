---
title: 'T-073 literature methods check: PRAVAH candidate contributions C1-C5'
pinned: false
links: []
created: '2026-10-13T22:50:52-04:00'
---

# T-073 literature methods check: PRAVAH candidate contributions C1-C5

T-073 literature methods check. Evidence: methods of SRC-001, 006, 007, 008, 009 read via arXiv/ar5iv HTML through a fetch tool that returns model-summarised text (not verbatim); appendices and some later sections unread. SRC-005 (Lin 2026, Elsevier) NOT ACCESSIBLE: only a catalogue listing and an unverified aggregator claim ("~25% improvement") were seen; nothing about it is asserted. SRC-002/003/004 not read, not marked.

TABLE (closest work | already shows | PRAVAH differs | experiment | verdict)

C1 TV nonneg batch source-history inversion, local+boundary sources.
 Closest: SRC-006 (nonneg elastic-net, sparse sources, but steady Gaussian plume, fixed emissions, interior sources); SRC-009/007 (time-dependent advection-diffusion inversion, but unknown = initial condition, Gaussian prior, zero Dirichlet inflow, no boundary source). TV/fused lasso (SRC-003, unread) is a standard regulariser.
 Differs: time-varying per-source profiles plus unknown boundary inflow, TV+nonneg (ADMM). Each ingredient is known; only the combination and the boundary-inflow unknown are absent from the sources read.
 Experiment: M2(TV) vs M1(ridge) vs M0 on matched synthetic truths incl. step/shutdown, fresh seeds, frozen hyperparameters; add a plain fused-lasso baseline on the same operator.
 Verdict: UNCERTAIN (novelty weak: combination only). Internal F-037 is preliminary and reviewer-disputed; M2 not uniformly better.

C2 Population-weighted dynamic placement vs source-ID placement, and their conflict.
 Closest: SRC-007/008/009 goal-oriented OED (QoI = linear functional of parameter; population exposure is such a QoI, so formally an instance). SRC-006 targets emission rates only; SRC-001 minimises a user-weighted filter-covariance trace.
 Already shows: QoI-oriented designs differ from parameter-oriented ones (SRC-007 premise, brute-force comparison). A quantified conflict between two objectives was not seen.
 Differs: population as QoI and the exposure-vs-source-ID trade-off under time-varying wind. Method is goal-oriented OED applied; only the conflict characterisation is a candidate contribution.
 Experiment: Pareto of J_pop vs source error over layouts (greedy and brute force on small candidate sets), several wind regimes, held-out scenarios; test whether the conflict exceeds between-scenario spread.
 Verdict: UNCERTAIN; method novelty NOT supported; the empirical conflict is untested in the literature read.

C3 Separate local emission changes from transboundary inflow under changing wind.
 Closest: none found among sources read. SRC-009 has inflow only as zero Dirichlet; SRC-006 wind is a random input, no inflow; SRC-001 periodic domain.
 Differs: identifiability of two source classes whose footprints change with wind. Absence in 5 papers is not absence in the literature (boundary/background-inversion literature not searched).
 Experiment: separability vs wind-direction diversity and sensor count; local-vs-boundary attribution confusion matrix; null with constant wind (separation should fail).
 Verdict: UNCERTAIN (plausible gap, not established; internal boundary-gain result unresolved).

C4 Design-scenario sufficiency (how many independent episodes for a stable design).
 Closest: SRC-006 sample-average rSAA/SBA (N~=5 or 100 scenarios, K=250 repeats, averaging layouts; Theorem 1 convergence bound) but no scenario-count stability study seen in text read. SRC-001 notes local optima and initialisation dependence.
 Differs: empirical layout variability vs number of design episodes.
 Experiment: layout displacement/regret vs number of episodes against a replicate-pair baseline (same logic as D-012).
 Verdict: UNCERTAIN (generic SAA theory exists; novelty at best an empirical characterisation).

C5 Discretisation-induced instability of sensor decisions.
 Closest: SRC-001 Table 1: filter MSE not monotone in basis size K (2.23e-2 at K=21 vs 2.67e-2 at K=37): MSE, not layout. SRC-007/008/009: no mesh-refinement study found. SRC-008 shows eigenvalue decay independent of parameter/candidate counts, which is not a mesh claim.
 Differs: sensitivity of the DECISION (layout, regret) to mesh and dt against a resampling baseline.
 Experiment: EXP-027 as registered; a void result is a registered stop. F-035: H1 untested, baseline not commensurable.
 Verdict: UNCERTAIN, leaning gap; internal evidence neither supports nor contradicts yet.

Overall: none CONTRADICTED, none SUPPORTED; sources show absence of coverage, not presence of the PRAVAH effect. C1+C2 are combinations of known tools. Most distinct: C3; C4/C5 as empirical decision-stability studies.

REUSABLE METHODS/CODE (read, not copied)
- SRC-006: bilevel design, rSAA/SBA, A-optimal initial layout (Prop 3). Repo github.com/lxc95/Optimal-Experimental-Design: licence NOT checked; paper CC BY-NC-ND. Do not copy; reimplement from equations only if needed.
- SRC-007: A/D-GOODE criteria, l1-weight relaxation + top-k thresholding; implementation via hIPPYlib/FEniCS (licences not verified here).
- SRC-008: offline-online decomposition (Thm 2) and swapping greedy (Alg 1): fits a linear-Gaussian dynamic response matrix with population QoI. Code licence not stated in text read.
- SRC-009: sensor-steering loop (infer, centre QoI, redesign, move), SVD ROM; code availability not stated.
- SRC-001: Kalman-Bucy/RML two-timescale idea; no code link on arXiv page.
Equations/algorithms can be reimplemented; verbatim code, figures and text are bound by each licence.

CLAIMS NOT VERIFIED
1 SRC-005 content entirely (incl. the 25% claim; whether it covers time-varying emissions, TV or boundary sources).
2 Whether SRC-006 appendices or unread last ~77k chars hold scenario-count or discretisation sensitivity.
3 Code availability/licence for SRC-007/008/009; SRC-006 repo licence.
4 Whether other literature (inflow/background control in 4D-Var, source-term change-point work) covers C3.
5 SRC-002/003/004 unread; TV/fused-lasso/changepoint relevance to C1 from general knowledge only.
6 Fetch returned model-summarised text; equation details and section numbers not confirmed verbatim.
