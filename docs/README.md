# Documentation index

Pick ONE row by the question you are answering; each doc states its own scope and links onward. Findings and their review status live in the Research
Panel (`.research/bin/research current`), not here.

| Question | Read |
|---|---|
| What is this project, how do I install and run it? | `../README.md` |
| What are the equations, units, boundary treatment, discretisation? | `specification.md` |
| How is the code organised (modules, data flow, parameter vector)? | `architecture.md` |
| How do I change a source / add a term / swap a solver / add an objective? | `extending.md` |
| How do I route a different forward solver through everything? | `solver-interchangeability.md` |
| What did v0 verify and measure (tolerances, benchmarks, mismatch)? | `validation.md` |
| External vs local pollution, tags, bases, indoor box | `transboundary.md` |
| Real Belauri data: sources, CRS, boundary verification, relief, border | `geography.md` |
| Belauri forward scenarios and mesh / outer-domain sensitivity | `belauri-forward.md` |
| Geographic sensor design, robustness, exports | `geodesign.md` |
| What phase 2 established, what did not, known issues, requirement table | `phase2-validation.md` |
| Why was it planned this way (design decisions, task map) | `implementation-plan.md`, `phase2-plan.md` |

Reading order for a new agent: `../AGENTS.md` -> `research current` -> this index -> one area doc -> `phase2-validation.md` "Wrap-up log" if you will change results.
