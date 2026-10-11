import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ancestors, buildIndex, isLeaf, matchSet, reveal, search, toggle as toggleSet } from './layout';
import { GraphView, type Ctx } from './Graph';
import { blankIdea, useStore, Workspace } from './Notebook';
import { LeafDetail, NodeDetail } from './Panel';
import type { Decision, Idea, Lifecycle, MNode, Model, Persp, Review } from './types';

type View = 'map' | 'ideas' | 'decisions';
const KEY = 'pravah-explorer-state-v1';
const read = () => { try { return JSON.parse(localStorage.getItem(KEY) ?? '{}'); } catch { return {}; } };
const write = (v: unknown) => { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* storage unavailable: state just is not remembered */ } };

const LIFE: Lifecycle[] = ['implemented', 'experimental', 'planned', 'unknown'];
const REV: Review[] = ['verified-scan', 'docs-only', 'needs-inspection'];
const concernsOf = (n: MNode) => (n.limitations ?? []).filter((l) => l.severity !== 'low').length;

function Chip({ on, onClick, children, cls = '' }: { on: boolean; onClick: () => void; children: React.ReactNode; cls?: string }) {
  return <button className={`fchip ${cls} ${on ? 'on' : ''}`} aria-pressed={on} onClick={onClick}>{children}</button>;
}

export function App({ model }: { model: Model }) {
  const ix = useMemo(() => buildIndex(model), [model]);
  const saved = useMemo(read, []);
  const [persp, setPersp] = useState<Persp>(saved.persp === 'software' ? 'software' : 'science');
  const [exp, setExp] = useState<Record<Persp, string[]>>({ science: [], software: [], ...(saved.exp ?? {}) });
  const [sel, setSel] = useState<string | null>(saved.sel && (ix.nodes.has(saved.sel) || isLeaf(saved.sel)) ? saved.sel : null);
  const [view, setView] = useState<View>('map');
  const [focus, setFocus] = useState<{ id: string; tick: number } | null>(null);
  const [q, setQ] = useState('');
  const [life, setLife] = useState<Set<Lifecycle>>(new Set());
  const [rev, setRev] = useState<Set<Review>>(new Set());
  const [onlyConcerns, setOnlyConcerns] = useState(false);
  const [menu, setMenu] = useState(false);
  const [draft, setDraft] = useState<Idea | Decision | null>(null);
  const ideas = useStore<Idea>('ideas');
  const decisions = useStore<Decision>('decisions');
  const tick = useRef(0);

  useEffect(() => write({ persp, exp, sel }), [persp, exp, sel]);
  const expanded = useMemo(() => new Set(exp[persp]), [exp, persp]);
  const setExpanded = useCallback((p: Persp, s: Set<string>) => setExp((e) => ({ ...e, [p]: [...s] })), []);

  const openIdeas = useCallback((id: string) => ideas.items.filter((i) => i.related.includes(id) && (i.status === 'idea' || i.status === 'investigating')).length, [ideas.items]);
  const filtering = life.size > 0 || rev.size > 0 || onlyConcerns;
  const lit = useMemo(() => matchSet(ix, (n) => (!life.size || life.has(n.lifecycle)) && (!rev.size || rev.has(n.review)) && (!onlyConcerns || concernsOf(n) > 0)), [ix, life, rev, onlyConcerns]);

  const nodeOf = (id: string | null) => (id == null ? null : isLeaf(id) ? ix.nodes.get(id.split(':')[1]) ?? null : ix.nodes.get(id) ?? null);
  const selNode = nodeOf(sel);

  const goTo = useCallback((id: string) => {
    const n = nodeOf(id);
    if (!n) return;
    setView('map'); setPersp(n.perspective); setExpanded(n.perspective, reveal(ix, new Set(exp[n.perspective]), id)); setSel(id);
    setFocus({ id, tick: ++tick.current });
  }, [ix, exp, setExpanded]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = useCallback((id: string) => {
    setExpanded(persp, toggleSet(ix, expanded, id));
    setFocus({ id, tick: ++tick.current });
  }, [ix, expanded, persp, setExpanded]);

  const switchPersp = (p: Persp) => {
    if (p === persp) return;
    const m = selNode?.mirrors?.map((i) => ix.nodes.get(i)).filter((n): n is MNode => !!n && n.perspective === p)[0];
    if (m) goTo(m.id); else { setPersp(p); setSel(null); setFocus({ id: '', tick: ++tick.current }); }
  };

  const linked = useMemo(() => new Set(selNode?.mirrors ?? []), [selNode]);
  const ctx: Ctx = { ix, selected: sel, expanded, dim: (id) => filtering && !lit.has(id), linked, concerns: (id) => concernsOf(ix.nodes.get(id)!), ideas: openIdeas, select: setSel, toggle };
  const hits = useMemo(() => search(ix, q), [ix, q]);
  const crumbs = sel ? [...ancestors(ix, nodeOf(sel)?.id ?? sel).reverse(), nodeOf(sel)!.id] : [];

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (e.key === '/' && tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') { e.preventDefault(); document.getElementById('search')?.focus(); }
      if (e.key === 'Escape' && tag !== 'INPUT') setSel(null);
    };
    window.addEventListener('keydown', k); return () => window.removeEventListener('keydown', k);
  }, []);

  const newIdeaFor = (id: string) => { setView('ideas'); setDraft(blankIdea([id])); };
  const openIdeas2 = (id: string) => {
    setView('ideas');
    const item = ideas.items.find((i) => i.id === id);
    setDraft(item ? { ...item } : null);
  };
  const toggleIn = <T,>(s: Set<T>, v: T, set: (n: Set<T>) => void) => { const n = new Set(s); n.has(v) ? n.delete(v) : n.add(v); set(n); };

  return (
    <div className={`app view-${view}`}>
      <nav className={`nav ${menu ? 'open' : ''}`} aria-label="Atlas navigation">
        <div className="brand"><h1>PRAVAH <small>research atlas</small></h1><button className="btn small menu" aria-expanded={menu} onClick={() => setMenu(!menu)}>Menu</button></div>
        <div className="navbody">
          <div className="seg" role="group" aria-label="Perspective">
            <button className={persp === 'science' ? 'on' : ''} aria-pressed={persp === 'science'} data-testid="persp-science" onClick={() => { setView('map'); switchPersp('science'); }}>Scientific map</button>
            <button className={persp === 'software' ? 'on' : ''} aria-pressed={persp === 'software'} data-testid="persp-software" onClick={() => { setView('map'); switchPersp('software'); }}>Software map</button>
          </div>
          <div className="seg" role="group" aria-label="Workspace">
            <button className={view === 'map' ? 'on' : ''} onClick={() => setView('map')}>Map</button>
            <button className={view === 'ideas' ? 'on' : ''} data-testid="view-ideas" onClick={() => { setView('ideas'); setDraft(null); }}>Ideas ({ideas.items.length})</button>
            <button className={view === 'decisions' ? 'on' : ''} onClick={() => { setView('decisions'); setDraft(null); }}>Decisions ({decisions.items.length})</button>
          </div>
          <div className="search">
            <label htmlFor="search" className="sr">Search concepts, methods, assumptions, files</label>
            <input id="search" type="search" placeholder="Search concept, method, file…  ( / )" value={q} onChange={(e) => setQ(e.target.value)} />
            {hits.length > 0 && <ul className="hits" aria-label="Search results">{hits.map((h) => (
              <li key={h.id}><button onClick={() => { goTo(h.id); setQ(''); }}><b>{h.title}</b><span className="muted">{h.persp} · {h.where}</span></button></li>))}</ul>}
            {q.trim().length >= 2 && hits.length === 0 && <p className="muted small-t">No match.</p>}
          </div>
          <details open className="filters"><summary>Filters <span className="muted">(dim non-matching)</span></summary>
            <div className="chips">{LIFE.map((l) => <Chip key={l} cls={`life-${l}`} on={life.has(l)} onClick={() => toggleIn(life, l, setLife)}>{l}</Chip>)}</div>
            <div className="chips">{REV.map((r) => <Chip key={r} on={rev.has(r)} onClick={() => toggleIn(rev, r, setRev)}>{r}</Chip>)}</div>
            <div className="chips"><Chip on={onlyConcerns} onClick={() => setOnlyConcerns(!onlyConcerns)}>open concerns ▲</Chip>
              {filtering && <button className="link" onClick={() => { setLife(new Set()); setRev(new Set()); setOnlyConcerns(false); }}>clear</button>}</div>
          </details>
          <details className="legend"><summary>Legend</summary>
            <ul>
              <li><span className="sw life-implemented" /> implemented — code located</li>
              <li><span className="sw life-experimental" /> experimental</li>
              <li><span className="sw life-planned" /> planned / proposed, not implemented</li>
              <li><span className="sw life-unknown" /> unknown</li>
              <li>● code read in scan · ◐ docs only · ○ needs inspection</li>
              <li>▲ high/medium limitation · ✎ open ideas</li>
              <li><span className="ln data" /> data flows · <span className="ln control" /> controls · <span className="ln evaluates" /> evaluates · <span className="ln feedback" /> feedback · <span className="ln alternative" /> alternative/replaces</li>
              <li><b>+</b> expands: overview → mechanisms &amp; decisions → source files. Double-click or → expands, ← collapses.</li>
            </ul>
            <p className="muted small-t">“Code read” means the implementation and tests were located and read when the map was curated; tests were not executed by this tool. Run <code>npm run check:evidence</code> for drift.</p>
          </details>
        </div>
      </nav>

      <main>
        {view === 'map' ? (<>
          <div className="crumbs" aria-label="Breadcrumb">
            <button className="link" onClick={() => setSel(null)}>{persp === 'science' ? 'Scientific map' : 'Software map'}</button>
            {crumbs.map((c) => (<span key={c}> › <button className="link" onClick={() => goTo(c)}>{ix.nodes.get(c)?.title}</button></span>))}
            {isLeaf(sel ?? '') && <span> › <code>{ix.ev.get(sel!.split(':')[2])?.path.split('/').pop()}</code></span>}
            <span className="spacer" />
            <button className="btn small" onClick={() => { setExpanded(persp, new Set()); setFocus({ id: '', tick: ++tick.current }); }}>Collapse all</button>
          </div>
          <div className="canvas"><GraphView ctx={ctx} persp={persp} focus={focus?.id ? focus : focus ? { id: '', tick: focus.tick } : null} /></div>
        </>) : (
          <Workspace key={view} kind={view} items={view === 'ideas' ? ideas.items : decisions.items} status={view === 'ideas' ? ideas.status : decisions.status}
            persist={view === 'ideas' ? ideas.persist : decisions.persist} nodes={model.nodes} focusId={selNode?.id ?? null} onGo={goTo} draft={draft} setDraft={setDraft} />
        )}
      </main>

      {view === 'map' && (
        <aside className="panel" aria-label="Details" data-testid="panel">
          {!selNode ? (
            <div className="detail intro">
              <h2>{persp === 'science' ? 'The scientific system' : 'The software system'}</h2>
              <p>{persp === 'science'
                ? 'Eight concepts, from physical transport to validation. Click a card for its purpose; press + to expand it into mechanisms and decisions, then again into the source files that implement it.'
                : 'The same system as modules and data flow. Each module links back to the scientific concepts it implements.'}</p>
              <p className="muted">Everything quantitative here is synthetic; only the Belauri geometry is real. Nothing in this map is a claim about measurements.</p>
              <ul className="links"><li>Open concerns ▲ mark documented limitations of high or medium impact.</li><li>Press <kbd>/</kbd> to search concepts, assumptions and file paths.</li></ul>
            </div>
          ) : isLeaf(sel!) ? <LeafDetail e={ix.ev.get(sel!.split(':')[2])!} node={selNode} onGo={goTo} />
            : <NodeDetail n={selNode} ix={ix} model={model} onGo={goTo} ideas={ideas.items} onNewIdea={newIdeaFor} onOpenIdeas={openIdeas2} />}
        </aside>
      )}
    </div>
  );
}
