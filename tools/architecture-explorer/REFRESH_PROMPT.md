# Refresh prompt (paste into a fresh Claude Code session at the repository root)

> Refresh the PRAVAH architecture explorer map (`tools/architecture-explorer/content/model.json`).
>
> 1. Read `AGENTS.md`, then run `cd tools/architecture-explorer && npm run check:evidence`. Treat every `drift:` / `stale` line as a lead, not proof that a claim is false.
> 2. Run `git log --stat <lock revision>..HEAD -- src configs tests docs scripts` (revision is in `content/evidence.lock.json`) and read only the changed files that evidence entries cite, plus new modules/docs not yet mapped.
> 3. For each affected node: re-inspect the code/test/config, then update `content/model.json` — claims, line ranges, `lifecycle`, `review` (`verified-scan` only if you read implementing code/tests in this session; `docs-only` otherwise), limitations and cross-links (mirrors must stay reciprocal). Mark anything planned/unimplemented as such. Do not invent paths, lines, tests, metrics or algorithms; surface contradictions.
> 4. Propose the change as a reviewable patch (show the diff; explain each change with its evidence). Do NOT edit `content/ideas.json` or `content/decisions.json` — they hold the owner's notes and Claude proposals; you may only *report* ideas whose related node ids no longer exist. Do not run simulations, edit `results/`, `data/`, `configs/`, `notebooks/` or `src/`.
> 5. After the owner approves: `npm run check:evidence` (must show no errors), `npm test`, then `npm run lock:evidence` to record the new baseline.
