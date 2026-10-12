# Documentation index

Pick ONE row by the question you are answering; each doc states its own scope and links onward. Findings and their review status live in the Research
Panel (`.research/bin/research current`), not here.

| Question | Read |
|---|---|
| What is this project, how do I install and run it? | `../README.md` |
| What are the equations, units, boundary treatment, discretisation? | `specification.md` |
| How is the code organised (modules, data flow, parameter vector)? | `architecture.md` |
| How do I change a source / add a term / swap a solver / add an objective? | `extending.md` |
| What winds does the model accept, and what is the ERA5 ingestion? | `winds.md` |
| How do I route a different forward solver through everything? | `solver-interchangeability.md` |
| What did v0 verify and measure (tolerances, benchmarks, mismatch)? | `validation.md` |
| External vs local pollution, tags, bases, indoor box | `transboundary.md` |
| Real Belauri data: sources, CRS, boundary verification, relief, border | `geography.md` |
| Belauri forward scenarios and mesh / outer-domain sensitivity | `belauri-forward.md` |
| Geographic sensor design, robustness, exports | `geodesign.md` |
| Run the geodesign pipeline on another municipality (what to supply, what is still Belauri-specific) | `transfer.md` |
| What phase 2 established, what did not, known issues, requirement table | `phase2-validation.md` |
| Run on HPC: submit, monitor, retrieve, recover (Research OS path, verified) + set-up, order, EXP-015 stage 2, EXP-016 spec, registration, known gaps | `hpc-runbook.md` |
| Population data sources (WorldPop, GHS-POP): units, CRS, licences, regrid caveats, how to plug a raster in | `population-data.md` |
| Numerical verification and uncertainty methodology: definitions, scheme and convergence results, decision-stability metrics, interval wording | `numerical-methodology.md` |
| Phase 3 findings, limitations, what is still open, Phase 4 recommendations | `phase3-findings.md` |
| Why was it planned this way (design decisions, task map) | `implementation-plan.md`, `phase2-plan.md` |

Reading order for a new agent: `../AGENTS.md` -> `research current` -> this index -> one area doc -> `phase2-validation.md` "Wrap-up log" if you will change results.
