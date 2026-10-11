# Claude Code task: Build PRAVAH's Research Architecture Explorer

You are working INSIDE the existing PRAVAH repository. Implement a working, local-only research architecture explorer at `tools/architecture-explorer/`. Do not just propose a plan or produce mockups; inspect the repository, build the application, test it, and explain how to run it. Keep your final reply concise and explicit about what you verified.

## Mission

This is my **private scientific reasoning workspace**, not a public dashboard, dependency graph, repository browser, or marketing site. I want to understand the research system at multiple depths:

1. **What scientific problem is this subsystem solving?**
2. **How does it work mathematically, numerically, and computationally?**
3. **Why was this approach selected; what assumptions and trade-offs does it make?**
4. **Where is the implementation, configuration, verification, and experimental evidence?**
5. **What limitations, plausible alternatives, open questions, and research ideas does it suggest?**

The central organizing units must be *scientific concepts and architectural decisions*, not individual helper functions. I need to move from the system-level narrative to specific mechanisms and finally the source files worth reading. I should be able to formulate research ideas without losing them.

## Before implementing: inspect reality

Read `AGENTS.md` (and any applicable `CLAUDE.md`) first, then the actual repository structure. Read at least `README.md`, `docs/README.md`, `docs/architecture.md`, `docs/specification.md`, `docs/implementation-plan.md`, `docs/solver-interchangeability.md`, `docs/extending.md`, `docs/validation.md`, `docs/phase2-validation.md`, `docs/transboundary.md`, `docs/belauri-forward.md`, and `docs/geodesign.md` where they exist. Trace the relevant Python source in `src/pravah/`, representative configs, tests, and notebooks. Use grep/rg and targeted file reads; do not indiscriminately dump all code or notebook output into context.

**Documentation is a lead, not proof of implementation.** For each important assertion, find implementation and/or test evidence. If documentation describes a future capability, explicitly mark it as planned or unimplemented. Distinguish observed-in-code, asserted-in-docs, scientifically verified, and proposed. No fabricated source paths, exact line locations, tests, metrics, or algorithms. Missing or contradictory evidence must be surfaced, never smoothed over.

Do not rerun long production benchmarks or overwrite `results/`, `data/`, configs, notebooks, `.research/`, or any simulation code. The explorer should not import or execute the PRAVAH Python package at runtime. This first implementation is a read-only view of the scientific repository, except for files owned by the explorer that store research annotations.

## Minimum viable architecture

Use **React + TypeScript + Vite + React Flow** (`@xyflow/react`) inside `tools/architecture-explorer/`. Avoid Next.js, a database, a Python backend, auth, cloud services, AI APIs, or framework-heavy abstractions. Local browser access only (`127.0.0.1`). Keep dependencies minimal and code readable; apply Ponytail's YAGNI principles. Do not add GSAP, Three.js, or ELK unless the implemented interactions genuinely require them. CSS transitions are sufficient for simple animation.

Two switchable, *connected* modes:

### A. Scientific map (default)

Start with approximately 6–9 high-level concepts, chosen from the REAL repository. Candidate families (adapt after inspection):
- Physical transport model and assumptions (advection, diffusion, removal, boundary conditions, sources, mixing depth)
- Forward solution (FiPy, discretization, timestep, solver backends, truth vs inference)
- Observations / sensors (sampling/interpolation/noise)
- Inverse source estimation and identifiability (basis construction, constraints, Tikhonov alpha)
- Sensor-placement strategies and evaluation (random, spread, greedy, attribution-aware)
- External vs local attribution and one-way-coupled indoor box
- Belauri geography / scenario setup
- Benchmarks, verification, mismatch and robustness

These are suggestions, NOT claims of complete or current implementation. Map them to observed code and available evidence. For each concept show its scientific role, input/output, links to other concepts, core mathematical model (KaTeX only if needed), key assumptions, and implementation status.

### B. Software architecture map

Show the same system through major modules and dataflow: CLI/config -> scenario/data -> solver -> observations -> inverse/design -> metrics/experiments/provenance. Show major interfaces and module boundaries, not a function-call hairball. Every major software node must link back to the related scientific node(s) and vice versa. Annotate the actual entry point(s), implementations, tests, and representative config files.

### Progressive exploration (critical)

- Click a concept/module to EXPAND into a progressively more detailed diagram (not just display a wall of text). Collapsing must work. Support a stable breadcrumb/back path, pan/zoom/fit-to-view, and preserving the current context when switching science/software perspectives.
- Aim for **three useful levels**: overview -> mechanisms/decisions -> implementation/evidence. Stop there unless the code clearly warrants more. Avoid exposing every helper function.
- A compact detail area is allowed for explanations, equations, decisions, and evidence, but the PRIMARY exploration must be through progressively expanding graphs.
- Label edges by relationship (e.g., `provides forcing`, `generates observations`, `solves`, `estimates`, `selects`, `evaluates`). Do not make a misleading linear pipeline where there are cycles, interchangeable backends, or independent experiments.
- Search by scientific concept, method, module, assumption, and file path. Filters: implemented/planned, verified/documented/proposed, open concerns. Keyboard navigation, sufficient contrast, and a responsive layout are required.

## Content model and traceability

Maintain a small, human-editable, validated structured content model (JSON or TS data; choose ONE straightforward approach) for concepts, edges, decisions, issues/ideas, and evidence. Stable IDs and explicit cross-links must support both graph modes. Avoid generating descriptions from import ASTs at page load.

Each major concept or decision should have, where relevant:
- Plain-language purpose; short technical explanation; exact equation/method where supported
- Important inputs/outputs and related concepts
- Chosen approach; why (if known); alternatives and trade-offs
- Assumptions and limitations, with severity/impact clearly explained
- Lifecycle status: implemented / experimental / planned / unknown
- Evidence with repository-relative path, optional line range, evidence kind (`code`, `test`, `config`, `docs`, `notebook`, `result`), and a brief explanation of what it demonstrates
- An action to copy/open the relevant `path:line`, and ideally view a short highlighted excerpt locally when supported (do not promise editor deep links that cannot work)
- Review status: verified in current scan, documentation-only, or requiring inspection

For scientific claims do NOT use "verified" merely because there is a test file or a README statement. Label what was actually checked. State whether a validation result is a smoke test, synthetic matched model, numerical verification, mismatch test, or empirical measurement.

Seed substantive content from **actual repo inspection**. The starting explorer must not consist of placeholder nodes. In particular, investigate and make navigable these research decisions and caveats:
- The governing advection–diffusion–removal PDE, grid-normalized Gaussian sources, boundary conditions, and fixed mixing depth.
- First-order time stepping / numerical diffusion (README reports a significant effective-versus-nominal diffusivity discrepancy); distinguish measurement from theory and inspect supporting evidence.
- FiPy backend interchangeability, solver capabilities, and any supported alternatives.
- Inverse source amplitudes with known transport parameters, non-negativity, sensor interpolation/noise, identifiability, and rank/uncertainty diagnostics.
- How Tikhonov alpha is chosen: separate validation ensemble, frozen alpha, and fresh test ensemble. Mark exploratory benchmark runs as superseded. Do not silently merge exploratory and final results.
- Greedy sensor-placement heuristics versus optimality; design-stage versus held-out test performance.
- Matched-model inverse crime versus mismatch/robustness experiments.
- Belauri real geography with synthetic emissions/met; DEM as plot context, NOT a terrain-driven wind solver.
- External-versus-local attribution and what 'transboundary' does *not* establish.

Do not assert that any undocumented future extension (e.g. neural surrogate, chemistry, forecast, empirical calibration) exists.

## Research ideas workspace

I need to *write and save* ideas inside the browser. Implement a small, reliable local file-backed editor for explorer-owned files (for example `content/ideas.json` and `content/decisions.json`). Browser storage alone is NOT sufficient. Use the simplest safe solution: a tightly scoped local Vite development-server middleware/API or tiny Node server using built-in filesystem APIs; choose whichever requires less code. Bind to loopback; accept writes ONLY to explicit explorer-owned files, validate JSON/input, check request origin, and use atomic writes or equivalent protection against data loss. Never expose generic filesystem write endpoints.

Each idea has: title, question/hypothesis, related concept IDs, motivation, possible method/experiment, references/URLs, expected evidence, status (`idea`, `investigating`, `tested`, `accepted`, `rejected`), and free-form notes. Keep timestamps and stable IDs. I should be able to create, edit, delete with confirmation, filter, and find ideas from relevant diagram nodes. Decisions should record choice, rationale, alternatives, evidence, and date/status. Distinguish human-authored annotations from Claude-generated suggestions. Preserve my edits through any architecture refresh; never replace them wholesale.

Research ideas proposed by Claude must be visibly labeled as **proposals**, not findings or implemented solutions. Provide a small number of concrete, scientifically interesting starting prompts tied to observed limitations, but do not invent performance results or unsupported literature citations.

## Refresh and drift detection

The map is **curated with Claude-assisted refresh**, NOT continuously auto-generated. Implement `npm run check:evidence` or an equivalent small script that validates the content schema and cross-links; verifies cited repository paths and optional line ranges; records/checks source revision/hash changes; and prints actionable messages (missing reference, changed referenced file, stale path, invalid ID, unsupported claim requiring manual review). File changes are a *drift signal*, NOT proof that a claim became false. This check must not rewrite content or run simulations.

Write `tools/architecture-explorer/REFRESH_PROMPT.md` containing a short repeatable prompt for a future Claude Code session: inspect relevant code/doc/config/test changes since the last refresh; produce a human-reviewable proposed patch with evidence; update vetted map content only; never overwrite research ideas/decisions. Optional: a local project-specific Claude skill to run that workflow, but not necessary for MVP.

## Visual direction

Create an intentional, restrained **scientific systems map**: legible technical typography, spacious diagrams, meaningful semantic colors, clear graph topology, subtle animation, excellent hover/focus states. The UI should feel like a researcher's interactive notebook/architecture atlas, not a generic SaaS dashboard, flashy landing page, or wall of cards. Integrate the installed design skills if useful (Taste, Minimalist UI, Web Design Guidelines, UI/UX Pro Max, Impeccable); prioritize scientific usability and accessibility over design flourishes. Do not add charts or animations without an explanatory function.

Recommended layout: compact left navigation and perspective selector; large central expandable graph; context-sensitive mechanism/decision/evidence panel; integrated research ideas view. Make it work well on a laptop. Include an at-a-glance legend and indicate provenance/status directly on nodes. Avoid overwhelming the initial view.

## Implementation and verification

1. Audit the repo and briefly write down a factual concept map and important implementation decisions before coding. Adapt the structure to what you find.
2. Implement one *complete vertical slice first*: scientific node -> expand -> decision -> evidence path -> attach/persist idea. Prove persistence by reload. Then fill out the rest of the graph.
3. Add meaningful initial content for all major *implemented* subsystems and links across both perspectives. Optimize for explanation quality, not node count.
4. Add tests for graph expansion/collapse and cross-link integrity, content/evidence validation, missing-file/drift conditions, and research note persistence. Use the browser via Playwright MCP if available; test at laptop viewport and narrow viewport. Take screenshots to review usability.
5. Do not introduce a Python dependency or mutate PRAVAH's scientific pipeline. Existing `pytest`, `pyright`, notebook and benchmark results remain untouched.
6. Include an explorer README with exact one-time setup and local start command (`npm install`, `npm run dev`, listening on `127.0.0.1`), test/check commands, a content schema explanation, and how to refresh after PRAVAH changes.
7. Summarize: what files were created, what scientific subsystems were mapped, which claims are supported vs uncertain, tested interactions, remaining gaps, and the commands to run.

### Acceptance criteria

- I can open the local webpage and immediately see a meaningful PRAVAH scientific overview, not a generic generated repository graph.
- I can expand an important node at least twice to reach specific methods/decisions and actual source files.
- I can switch to a software map and find the code underlying the same concept without losing orientation.
- I can distinguish implementation facts, documented assertions, unimplemented plans, known scientific limitations, and my own speculative ideas.
- I can save an idea in the browser, reload, and see it persisted to a tracked explorer file.
- Evidence links resolve to real repository paths; missing/stale ones are reported by a command.
- A future Claude session can refresh curated content without deleting my notes.
- Local-only app; no changes to model code, tests, production results, or archived scientific output.

Start with repo inspection, then execute the implementation. If something is genuinely ambiguous, prefer a simple decision consistent with this brief rather than stopping for more planning. Do not add technology simply because a design skill mentions it.
