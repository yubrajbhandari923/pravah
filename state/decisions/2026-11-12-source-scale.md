# 2026-11-12 — Amendment A1: reference source rate a_ref = 1 g/s

**Original (draft protocol §4, benchmark-v1):** a_k ~ LogUniform(0.5, 5) × a_ref, a_ref = 100 µg/s.

**Evidence:** gate-00 B1 estimated per-sample SNR ~1e-3; E02 examples (peak anomaly ~1e-3 µg/m³ vs σ 0.25),
E03 (rank deficient in all 648 fits), E04 (practical rank 0/4 everywhere) and E05 (practical rank 0 in every cell)
confirmed it. Rescaling the E05 design matrices (σ held at 0.25) showed designs start to differ near 1 g/s
(48 % of cells with ≥1 detectable mode, 3 % fully identifiable), see state/experiments/e05.md.

**Change:** a_ref = 1e6 µg/s (1 g/s, 3.6 kg/h; strengths 0.5–5 g/s). Nothing else changes: noise stays
σ = 5 % of the median training truth at the candidate sites (it therefore grows with the plume), background,
K, λ, mixing depth, winds, seeds, splits and endpoints are unchanged.

**Decision:** recommended by the lead; approved by the user on 2026-11-12 ("Looks good get going" in reply to the
recommendation). Scope of the approval is B1 only; protocol approval/freeze (B2) and the numerical-diffusion question (B3) stay open.

**Implications:** all E02–E05 numbers made at 100 µg/s are superseded for interpretation; they stay in Git history
as the record of the unidentifiable regime. The E05 rescale analysis that motivated the value used the outcome data,
so 1 g/s was chosen after seeing results: E03–E05 reruns at this scale are exploratory, not confirmatory.
Confirmatory use starts with E07+ under a frozen protocol.
