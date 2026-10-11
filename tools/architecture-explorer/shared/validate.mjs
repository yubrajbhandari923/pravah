// Shared by the UI, the dev-server middleware, the tests and scripts/check-evidence.mjs.
// Pure functions only: every validator returns a list of human-readable problems ([] = valid).

export const LIFECYCLE = ['implemented', 'experimental', 'planned', 'unknown'];
export const REVIEW = ['verified-scan', 'docs-only', 'needs-inspection'];
export const EVIDENCE_KINDS = ['code', 'test', 'config', 'docs', 'notebook', 'result'];
export const VALIDATION = ['none', 'smoke', 'synthetic-matched', 'numerical-verification', 'mismatch', 'empirical'];
export const NODE_KINDS = ['concept', 'mechanism', 'decision', 'module', 'component'];
export const EDGE_KINDS = ['data', 'control', 'evaluates', 'alternative', 'feedback'];
export const SEVERITY = ['high', 'medium', 'low'];
export const PERSPECTIVES = ['science', 'software'];
export const IDEA_STATUS = ['idea', 'investigating', 'tested', 'accepted', 'rejected'];
export const DECISION_STATUS = ['proposed', 'decided', 'superseded'];
export const ORIGIN = ['human', 'claude-proposal'];

const ID = /^[a-z][a-z0-9_.-]*$/;
const str = (v) => typeof v === 'string' && v.trim().length > 0;
const strs = (v) => Array.isArray(v) && v.every((x) => typeof x === 'string');

/** Validate the curated map (content/model.json). Cross-links and structure only; files are checked by check-evidence. */
export function validateModel(m) {
  const p = [];
  if (!m || typeof m !== 'object') return ['model is not an object'];
  for (const k of ['nodes', 'edges', 'evidence']) if (!Array.isArray(m[k])) p.push(`model.${k} must be an array`);
  if (p.length) return p;
  const byId = new Map();
  const seen = (kind, id) => {
    if (!str(id) || !ID.test(id)) return p.push(`${kind}: invalid id ${JSON.stringify(id)} (lowercase letters, digits, . _ -)`), false;
    if (byId.has(id)) return p.push(`duplicate id ${id}`), false;
    byId.set(id, kind);
    return true;
  };
  const ev = new Set();
  for (const e of m.evidence) {
    if (!seen('evidence', e.id)) continue;
    ev.add(e.id);
    if (!str(e.path) || e.path.startsWith('/') || e.path.includes('..')) p.push(`evidence ${e.id}: path must be repository-relative`);
    if (!EVIDENCE_KINDS.includes(e.kind)) p.push(`evidence ${e.id}: kind must be one of ${EVIDENCE_KINDS.join('|')}`);
    if (e.validation !== undefined && !VALIDATION.includes(e.validation)) p.push(`evidence ${e.id}: bad validation ${e.validation}`);
    if (!str(e.note)) p.push(`evidence ${e.id}: note (what it demonstrates) is required`);
    if (e.lines !== undefined && !(Array.isArray(e.lines) && e.lines.length === 2 && Number.isInteger(e.lines[0]) && Number.isInteger(e.lines[1]) && e.lines[0] >= 1 && e.lines[1] >= e.lines[0]))
      p.push(`evidence ${e.id}: lines must be [start, end] with 1 <= start <= end`);
  }
  const nodes = new Map();
  for (const n of m.nodes) {
    if (!seen('node', n.id)) continue;
    nodes.set(n.id, n);
    if (!PERSPECTIVES.includes(n.perspective)) p.push(`node ${n.id}: bad perspective`);
    if (!NODE_KINDS.includes(n.kind)) p.push(`node ${n.id}: bad kind`);
    if (!LIFECYCLE.includes(n.lifecycle)) p.push(`node ${n.id}: lifecycle must be ${LIFECYCLE.join('|')}`);
    if (!REVIEW.includes(n.review)) p.push(`node ${n.id}: review must be ${REVIEW.join('|')}`);
    for (const k of ['title', 'role', 'purpose']) if (!str(n[k])) p.push(`node ${n.id}: ${k} is required`);
    for (const k of ['inputs', 'outputs', 'assumptions', 'alternatives', 'tags']) if (n[k] !== undefined && !strs(n[k])) p.push(`node ${n.id}: ${k} must be a string array`);
    for (const l of n.limitations ?? []) if (!str(l.text) || !SEVERITY.includes(l.severity) || !str(l.impact)) p.push(`node ${n.id}: each limitation needs text, severity, impact`);
    for (const k of ['evidence', 'mirrors']) if (!strs(n[k] ?? [])) p.push(`node ${n.id}: ${k} must be a string array`);
  }
  for (const n of nodes.values()) {
    if (n.parent != null) {
      const par = nodes.get(n.parent);
      if (!par) p.push(`node ${n.id}: unknown parent ${n.parent}`);
      else if (par.perspective !== n.perspective) p.push(`node ${n.id}: parent is in another perspective`);
    }
    for (const id of n.evidence ?? []) if (!ev.has(id)) p.push(`node ${n.id}: unknown evidence ${id}`);
    for (const id of n.mirrors ?? []) {
      const o = nodes.get(id);
      if (!o) p.push(`node ${n.id}: unknown mirror ${id}`);
      else if (o.perspective === n.perspective) p.push(`node ${n.id}: mirror ${id} must be in the other perspective`);
      else if (!(o.mirrors ?? []).includes(n.id)) p.push(`cross-link ${n.id} -> ${id} is not reciprocal`);
    }
    if (n.review === 'verified-scan' && !(n.evidence ?? []).some((id) => ['code', 'test', 'config', 'result', 'notebook'].includes(m.evidence.find((e) => e.id === id)?.kind)))
      p.push(`node ${n.id}: review 'verified-scan' needs code/test/config/result evidence, not documentation alone`);
    if (n.lifecycle === 'planned' && n.review === 'verified-scan') p.push(`node ${n.id}: a planned capability cannot be 'verified-scan' (unsupported claim)`);
    if (n.lifecycle === 'implemented' && n.kind !== 'decision' && !(n.evidence ?? []).length && !m.nodes.some((c) => c.parent === n.id))
      p.push(`node ${n.id}: implemented but cites no evidence`);
  }
  const edgeIds = new Set();
  for (const e of m.edges) {
    if (!seen('edge', e.id)) continue;
    edgeIds.add(e.id);
    const a = nodes.get(e.from), b = nodes.get(e.to);
    if (!a || !b) { p.push(`edge ${e.id}: unknown endpoint`); continue; }
    if (a.perspective !== e.perspective || b.perspective !== e.perspective) p.push(`edge ${e.id}: endpoints must be in perspective ${e.perspective}`);
    if (!str(e.label)) p.push(`edge ${e.id}: label (relationship) is required`);
    if (!EDGE_KINDS.includes(e.kind)) p.push(`edge ${e.id}: bad kind`);
  }
  // acyclic containment
  for (const n of nodes.values()) {
    const chain = new Set([n.id]);
    for (let c = n.parent ? nodes.get(n.parent) : null; c; c = c.parent ? nodes.get(c.parent) : null) {
      if (chain.has(c.id)) { p.push(`node ${n.id}: parent cycle`); break; }
      chain.add(c.id);
    }
  }
  return p;
}

const common = (it, i, p, idPrefix) => {
  const w = `item ${i}${it && it.id ? ` (${it.id})` : ''}`;
  if (!it || typeof it !== 'object') return p.push(`${w}: not an object`), false;
  if (!str(it.id) || !ID.test(it.id)) p.push(`${w}: bad id`);
  if (!str(it.title)) p.push(`${w}: title is required`);
  if (!ORIGIN.includes(it.origin)) p.push(`${w}: origin must be ${ORIGIN.join('|')}`);
  for (const k of ['created', 'updated']) if (!str(it[k]) || Number.isNaN(Date.parse(it[k]))) p.push(`${w}: ${k} must be an ISO timestamp`);
  if (!strs(it.related ?? [])) p.push(`${w}: related must be a string array`);
  for (const k of ['title', 'notes']) if (typeof (it[k] ?? '') === 'string' && (it[k] ?? '').length > 20000) p.push(`${w}: ${k} too long`);
  return idPrefix;
};

export function validateIdeas(list) {
  const p = [];
  if (!Array.isArray(list)) return ['ideas must be an array'];
  const ids = new Set();
  list.forEach((it, i) => {
    common(it, i, p);
    if (!it || typeof it !== 'object') return;
    if (ids.has(it.id)) p.push(`duplicate idea id ${it.id}`);
    ids.add(it.id);
    if (!IDEA_STATUS.includes(it.status)) p.push(`idea ${it.id}: status must be ${IDEA_STATUS.join('|')}`);
    for (const k of ['question', 'motivation', 'method', 'expected', 'notes']) if (typeof (it[k] ?? '') !== 'string') p.push(`idea ${it.id}: ${k} must be text`);
    if (!strs(it.references ?? [])) p.push(`idea ${it.id}: references must be a string array`);
  });
  return p;
}

export function validateDecisions(list) {
  const p = [];
  if (!Array.isArray(list)) return ['decisions must be an array'];
  const ids = new Set();
  list.forEach((it, i) => {
    common(it, i, p);
    if (!it || typeof it !== 'object') return;
    if (ids.has(it.id)) p.push(`duplicate decision id ${it.id}`);
    ids.add(it.id);
    if (!DECISION_STATUS.includes(it.status)) p.push(`decision ${it.id}: status must be ${DECISION_STATUS.join('|')}`);
    for (const k of ['choice', 'rationale', 'evidence', 'notes']) if (typeof (it[k] ?? '') !== 'string') p.push(`decision ${it.id}: ${k} must be text`);
    if (!strs(it.alternatives ?? [])) p.push(`decision ${it.id}: alternatives must be a string array`);
  });
  return p;
}

/** Ideas/decisions may point at node ids that a later refresh removed; report, never delete. */
export function danglingRelated(items, model) {
  const ids = new Set(model.nodes.map((n) => n.id));
  return items.flatMap((it) => (it.related ?? []).filter((r) => !ids.has(r)).map((r) => `${it.id}: related node ${r} no longer exists in the map`));
}
