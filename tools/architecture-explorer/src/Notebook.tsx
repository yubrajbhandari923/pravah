import { useEffect, useMemo, useRef, useState } from 'react';
import { load, save, type Kind } from './api';
import type { Decision, Idea, MNode } from './types';

type Item = Idea | Decision;
export const useStore = <T extends Item>(kind: Kind) => {
  const [items, setItems] = useState<T[]>([]);
  const rev = useRef('');
  const [status, setStatus] = useState<{ ok: boolean; msg: string }>({ ok: true, msg: 'loading…' });
  const reload = () => load<T>(kind).then((r) => { setItems(r.items); rev.current = r.rev; setStatus({ ok: true, msg: `loaded content/${kind}.json` }); })
    .catch((e) => setStatus({ ok: false, msg: `cannot load: ${e.message}` }));
  useEffect(() => { reload(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const persist = async (next: T[]) => {
    try {
      const r = await save<T>(kind, next, rev.current);
      setItems(r.items); rev.current = r.rev;
      setStatus({ ok: true, msg: `saved to tools/architecture-explorer/content/${kind}.json at ${new Date().toLocaleTimeString()}` });
      return true;
    } catch (e: any) {
      setStatus({ ok: false, msg: e.status === 409 ? 'The file changed on disk since it was loaded. Nothing was saved; reload and re-apply your edit.' : `not saved: ${e.message}` });
      return false;
    }
  };
  return { items, status, persist, reload };
};

const now = () => new Date().toISOString();
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'untitled';
const lines = (s: string) => s.split('\n').map((x) => x.trim()).filter(Boolean);

export const blankIdea = (related: string[] = []): Idea => ({
  id: `idea.${slug('new')}-${Date.now().toString(36)}`, title: '', status: 'idea', question: '', related, motivation: '', method: '', references: [], expected: '', notes: '',
  origin: 'human', created: now(), updated: now(),
});
export const blankDecision = (related: string[] = []): Decision => ({
  id: `dec.${slug('new')}-${Date.now().toString(36)}`, title: '', status: 'proposed', choice: '', rationale: '', alternatives: [], evidence: '', related, notes: '',
  origin: 'human', created: now(), updated: now(),
});

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="field"><span>{label}</span>{children}</label>;
}

function RelatedPicker({ value, onChange, nodes }: { value: string[]; onChange: (v: string[]) => void; nodes: MNode[] }) {
  return (
    <div className="field"><span>Related concepts (links the idea to diagram nodes)</span>
      <div className="chips">{value.map((id) => (
        <span className="chip rel" key={id}>{nodes.find((n) => n.id === id)?.title ?? `${id} (missing)`}
          <button aria-label={`Remove link ${id}`} onClick={() => onChange(value.filter((v) => v !== id))}>×</button></span>))}
      </div>
      <select aria-label="Add related node" value="" onChange={(e) => e.target.value && onChange([...value, e.target.value])}>
        <option value="">+ link a node…</option>
        {(['science', 'software'] as const).map((p) => (<optgroup key={p} label={p}>{nodes.filter((n) => n.perspective === p && !value.includes(n.id)).map((n) => <option key={n.id} value={n.id}>{n.title}</option>)}</optgroup>))}
      </select>
    </div>
  );
}

const IDEA_STATUS = ['idea', 'investigating', 'tested', 'accepted', 'rejected'] as const;
const DEC_STATUS = ['proposed', 'decided', 'superseded'] as const;

export function Workspace({ kind, items, status, persist, nodes, focusId, onGo, draft, setDraft }: {
  kind: Kind; items: Item[]; status: { ok: boolean; msg: string }; persist: (n: any[]) => Promise<boolean>; nodes: MNode[];
  focusId: string | null; onGo: (nodeId: string) => void; draft: Item | null; setDraft: (d: Item | null) => void;
}) {
  const [q, setQ] = useState('');
  const [st, setSt] = useState('all');
  const [org, setOrg] = useState('all');
  const [nodeFilter, setNodeFilter] = useState('');
  const isIdea = kind === 'ideas';
  const list = useMemo(() => items.filter((i) => (st === 'all' || i.status === st) && (org === 'all' || i.origin === org) && (!nodeFilter || i.related.includes(nodeFilter))
    && (!q || JSON.stringify(i).toLowerCase().includes(q.toLowerCase()))), [items, q, st, org, nodeFilter]);
  const sel = draft;
  const isNew = sel ? !items.some((i) => i.id === sel.id) : false;
  const set = (patch: Record<string, unknown>) => setDraft({ ...(sel as Item), ...patch } as Item);
  const doSave = async () => {
    if (!sel || !sel.title.trim()) return;
    const next = { ...sel, updated: now() };
    await persist(isNew ? [next, ...items] : items.map((i) => (i.id === next.id ? next : i)));
    setDraft(next);
  };
  const doDelete = async () => {
    if (!sel || isNew) return setDraft(null);
    if (!window.confirm(`Delete “${sel.title}”? This removes it from content/${kind}.json.`)) return;
    if (await persist(items.filter((i) => i.id !== sel.id))) setDraft(null);
  };
  const stats = isIdea ? IDEA_STATUS : DEC_STATUS;
  return (
    <div className="workspace">
      <aside className="wlist" aria-label={`${kind} list`}>
        <div className="row spread"><h2>{isIdea ? 'Research ideas' : 'My decisions'}</h2>
          <button className="btn" data-testid="new-item" onClick={() => setDraft(isIdea ? blankIdea(focusId ? [focusId] : []) : blankDecision(focusId ? [focusId] : []))}>New</button></div>
        <input type="search" placeholder="Filter text…" aria-label="Filter by text" value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="row">
          <select aria-label="Filter by status" value={st} onChange={(e) => setSt(e.target.value)}><option value="all">any status</option>{stats.map((s) => <option key={s}>{s}</option>)}</select>
          <select aria-label="Filter by author" value={org} onChange={(e) => setOrg(e.target.value)}><option value="all">any author</option><option value="human">mine</option><option value="claude-proposal">Claude proposals</option></select>
        </div>
        <select aria-label="Filter by linked node" value={nodeFilter} onChange={(e) => setNodeFilter(e.target.value)}><option value="">any linked node</option>{nodes.map((n) => <option key={n.id} value={n.id}>{n.title}</option>)}</select>
        <p className={status.ok ? 'muted small-t' : 'warn'} role="status" data-testid="save-status">{status.msg}</p>
        <ul className="wl">{list.map((i) => (
          <li key={i.id}><button className={`wrow ${sel?.id === i.id ? 'on' : ''}`} onClick={() => setDraft({ ...i })} data-testid="item-row">
            <span>{i.title || '(untitled)'}</span><span className="chip">{i.status}</span>{i.origin === 'claude-proposal' && <span className="chip proposal">proposal</span>}</button></li>))}
        </ul>
        {list.length === 0 && <p className="muted">Nothing matches.</p>}
      </aside>
      <section className="wedit" aria-label="Editor">
        {!sel ? <p className="muted">Select an item or create a new one. Everything here is written to <code>content/{kind}.json</code>; Claude-generated items are labelled proposals and are never findings.</p> : (<>
          {sel.origin === 'claude-proposal' && (
            <div className="proposal-banner" role="note"><b>Proposal generated by Claude.</b> A suggestion tied to a documented limitation — not a finding, not implemented, and no result is claimed.{' '}
              <button className="link" onClick={() => set({ origin: 'human' })}>Adopt as my own</button></div>)}
          <Field label="Title"><input value={sel.title} data-testid="f-title" onChange={(e) => set({ title: e.target.value })} /></Field>
          <div className="row">
            <Field label="Status"><select value={sel.status} onChange={(e) => set({ status: e.target.value as any })}>{stats.map((s) => <option key={s}>{s}</option>)}</select></Field>
            <div className="field"><span>Written</span><small>{sel.origin === 'human' ? 'by me' : 'by Claude (proposal)'} · created {sel.created.slice(0, 10)} · updated {sel.updated.slice(0, 16).replace('T', ' ')}</small></div>
          </div>
          {isIdea ? (<>
            <Field label="Question / hypothesis"><textarea rows={3} value={(sel as Idea).question} data-testid="f-question" onChange={(e) => set({ question: e.target.value })} /></Field>
            <RelatedPicker value={sel.related} nodes={nodes} onChange={(v) => set({ related: v })} />
            <Field label="Motivation"><textarea rows={3} value={(sel as Idea).motivation} onChange={(e) => set({ motivation: e.target.value })} /></Field>
            <Field label="Possible method / experiment"><textarea rows={3} value={(sel as Idea).method} onChange={(e) => set({ method: e.target.value })} /></Field>
            <Field label="Expected evidence"><textarea rows={2} value={(sel as Idea).expected} onChange={(e) => set({ expected: e.target.value })} /></Field>
            <Field label="References / URLs (one per line)"><textarea rows={2} value={(sel as Idea).references.join('\n')} onChange={(e) => set({ references: lines(e.target.value) })} /></Field>
          </>) : (<>
            <RelatedPicker value={sel.related} nodes={nodes} onChange={(v) => set({ related: v })} />
            <Field label="Choice"><textarea rows={2} value={(sel as Decision).choice} onChange={(e) => set({ choice: e.target.value })} /></Field>
            <Field label="Rationale"><textarea rows={3} value={(sel as Decision).rationale} onChange={(e) => set({ rationale: e.target.value })} /></Field>
            <Field label="Alternatives (one per line)"><textarea rows={3} value={(sel as Decision).alternatives.join('\n')} onChange={(e) => set({ alternatives: lines(e.target.value) })} /></Field>
            <Field label="Evidence (paths, run ids, findings)"><textarea rows={2} value={(sel as Decision).evidence} onChange={(e) => set({ evidence: e.target.value })} /></Field>
          </>)}
          <Field label="Notes"><textarea rows={4} value={sel.notes} data-testid="f-notes" onChange={(e) => set({ notes: e.target.value })} /></Field>
          <div className="row spread">
            <div className="row"><button className="btn primary" data-testid="save" disabled={!sel.title.trim()} onClick={doSave}>Save to file</button>
              <button className="btn ghost" onClick={() => setDraft(null)}>Close</button></div>
            <div className="row">{sel.related.map((r) => <button key={r} className="link" onClick={() => onGo(r)}>show {nodes.find((n) => n.id === r)?.title ?? r} in map</button>)}
              <button className="btn danger" data-testid="delete" onClick={doDelete}>{isNew ? 'Discard' : 'Delete…'}</button></div>
          </div>
        </>)}
      </section>
    </div>
  );
}
