# Decision: E00 protocol approved (B2), with evidence-driven amendment allowed

**Date:** 2026-11-12 · **Decided by:** user (project lead) · **Recorded by:** lead agent

User's words: "I am happy with E00 but dont 'freeze' it in the sense that you force it even if evidence seems that
something might be not numerically possible, but current settings seem reasonable to proceed with."

## What this means
- `docs/research/preregistered-protocol.md` (with amendment A1, a_ref = 1 g/s) and `configs/benchmark-v1.yaml` are
  the **approved baseline** from this date. Gate-00 blocker B2 is resolved; B1 was resolved by A1.
- Results are scored against this baseline; endpoints, metrics, split rules and test-data rules are not changed silently.
- **Not a rigid freeze:** if evidence shows a setting is numerically or practically infeasible (e.g. B3, numerical
  diffusion/Courant at 64²), the agent amends it rather than forcing it. Each amendment gets a decision file stating the
  evidence, what changed, and which results it affects. Changes that alter the primary endpoint or the confirmatory claim
  still go to the user.
- Confirmatory claims start with experiments designed after this date (E07/E08 onward). E02–E06 stay exploratory.
