# Decision: E07 confirmatory scope (gate-07 issue 1)

**Date:** 2026-11-13 · **Decided by:** user ("Yeah", replying to the lead's request for sign-off) · **Recorded by:** lead

The E07 confirmatory claim is stated as **"supported in the cells run"**: 2 training winds (veering, mixed) × K ∈ {2, 4} ×
k ∈ {8, 16}, 10 seeds per cell. Under protocol §12, the protocol's other k ≥ 8 cells (K = 8; k = 32; 20 seeds) are
**not tested** and count as unresolved; they are not claimed. K = 8 moves to E08.

Every statement of the result must also say that a locations-only control (source_near) beat dopt in 115/120 pairs.
The gain over random is attributable to knowing source locations, not to D-optimisation (gate-07).
