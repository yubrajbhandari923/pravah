# PRAVAH architecture explorer

A private, local-only scientific reasoning workspace for the PRAVAH repository: a **scientific map** (concepts → mechanisms/decisions → source files) and a **software map** (modules → components → source files), cross-linked, plus a research-ideas workspace saved to tracked files. React + TypeScript + Vite + React Flow; no backend, database, cloud or AI service. It is a read-only view of the repository except for the two annotation files it owns. It never imports or runs PRAVAH.

## Run

```bash
cd tools/architecture-explorer
npm install          # one time
npm run dev          # http://127.0.0.1:5174  (loopback only)
```

Other commands: `npm test` (vitest: layout/expand/collapse, cross-links, evidence/drift, persistence), `npm run typecheck`, `npm run build`, `npm run check:evidence` (read-only drift check), `npm run lock:evidence` (records the reviewed baseline; writes only `content/evidence.lock.json`).

## Using it

- Click a card for its purpose, equation, decision, limitations (▲ = high/medium impact) and evidence. **+** (or double-click / →) expands a node in place; − (or ←) collapses. Level 1 = mechanisms and decisions, level 2 = the source/test/config/doc/result files that evidence them. Evidence cards can copy `path:line` and show a highlighted local excerpt (served only for curated evidence ids).
- Panel links jump between maps (science ↔ software) and keep your place. `/` searches concepts, methods, assumptions and file paths. Filters dim non-matching nodes (lifecycle, review status, open concerns).
- **Ideas** / **Decisions** views create, edit, filter and delete (with confirmation) notes; link them to nodes; the node panel lists linked ideas. Items are marked *mine* or *Claude proposal*; proposals are suggestions tied to documented limitations, never findings.

## Content model (`content/`)

| File | Owner | Contents |
|---|---|---|
| `model.json` | curated (human + Claude refresh) | `nodes` (id, perspective, parent, kind, plain purpose, technical text, equation, inputs/outputs, chosen/why/alternatives, assumptions, limitations with severity+impact, `lifecycle`, `review`, `evidence` ids, `mirrors` ids, `tags`), `edges` (labelled relationship + kind), `evidence` (repo-relative `path`, optional `lines`, `kind` code/test/config/docs/notebook/result, `validation` kind, note). Decisions are nodes with `kind: "decision"`. |
| `ideas.json` | **you** | question/hypothesis, related node ids, motivation, method, references, expected evidence, status (idea/investigating/tested/accepted/rejected), notes, `origin` (`human`/`claude-proposal`), timestamps. |
| `decisions.json` | **you** | your own decisions: choice, rationale, alternatives, evidence, status, related ids. |
| `evidence.lock.json` | refresh | per-evidence hash of the cited lines at a git revision, for drift detection. |

Vocabulary: `lifecycle` = implemented / experimental / planned / unknown. `review` = `verified-scan` (implementing code/tests located and read when curated; **tests were not executed**), `docs-only`, `needs-inspection`. Evidence `validation` says what sort of check it is (smoke, synthetic-matched, numerical-verification, mismatch, empirical) — a test file alone is not scientific validation. Writes go through `PUT /api/{ideas|decisions}` only: loopback host/origin check, custom header, schema validation, revision check (409 if the file changed on disk), atomic temp-file + rename. No generic file endpoint exists.

## Refresh after PRAVAH changes

`npm run check:evidence` lists missing files, stale line ranges, invalid ids/links, unsupported claims, and drift since the recorded baseline (a drift is a signal, not proof). Then use `REFRESH_PROMPT.md` in a Claude Code session; it proposes a patch to `model.json` only and never touches your ideas/decisions.
