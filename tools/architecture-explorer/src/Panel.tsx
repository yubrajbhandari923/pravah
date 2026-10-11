import { useEffect, useState } from 'react';
import { excerpt, type Excerpt } from './api';
import type { Idx } from './layout';
import { Badges } from './Graph';
import type { Evidence, Idea, MNode, Model } from './types';

export const whereOf = (e: Evidence) => (e.lines ? `${e.path}:${e.lines[0]}${e.lines[1] !== e.lines[0] ? '-' + e.lines[1] : ''}` : e.path);

function List({ title, items }: { title: string; items?: string[] }) {
  if (!items?.length) return null;
  return (<section><h4>{title}</h4><ul>{items.map((x, i) => <li key={i}>{x}</li>)}</ul></section>);
}

function CopyBtn({ text, label = 'Copy path:line' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button className="btn small" onClick={() => { navigator.clipboard?.writeText(text).then(() => { setDone(true); setTimeout(() => setDone(false), 1400); }).catch(() => undefined); }}>
      {done ? 'Copied' : label}
    </button>
  );
}

export function EvidenceCard({ e, defaultOpen = false }: { e: Evidence; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const [x, setX] = useState<Excerpt | null>(null);
  const [err, setErr] = useState('');
  useEffect(() => {
    if (open && !x) excerpt(e.id).then(setX).catch((er) => setErr(String(er.message ?? er)));
  }, [open, e.id, x]);
  return (
    <li className={`ev ek-${e.kind}`}>
      <div className="evtop">
        <span className="chip">{e.kind}</span>
        {e.validation && e.validation !== 'none' && <span className="chip val" title="what kind of check this evidence is">{e.validation}</span>}
        <code className="path">{whereOf(e)}</code>
      </div>
      <p>{e.note}</p>
      <div className="row">
        <CopyBtn text={whereOf(e)} />
        <button className="btn small" aria-expanded={open} onClick={() => setOpen(!open)}>{open ? 'Hide excerpt' : 'View excerpt'}</button>
      </div>
      {open && (err ? <p className="warn">Excerpt unavailable: {err}</p> : !x ? <p className="muted">Loading…</p> : x.missing ? <p className="warn">File not found in the repository: {x.path}</p> : (
        <pre className="excerpt" aria-label={`Excerpt of ${whereOf(e)}`}>{x.text!.split('\n').map((l, i) => {
          const n = (x.start ?? 1) + i, hl = n >= (x.hlStart ?? 0) && n <= (x.hlEnd ?? -1);
          return <div key={i} className={hl ? 'hl' : ''}><span className="ln">{n}</span>{l || ' '}</div>;
        })}</pre>
      ))}
    </li>
  );
}

export function NodeDetail({ n, ix, model, onGo, ideas, onNewIdea, onOpenIdeas }: {
  n: MNode; ix: Idx; model: Model; onGo: (id: string) => void; ideas: Idea[]; onNewIdea: (id: string) => void; onOpenIdeas: (id: string) => void;
}) {
  const [tab, setTab] = useState<'detail' | 'evidence' | 'ideas'>('detail');
  useEffect(() => setTab('detail'), [n.id]);
  const ev = (n.evidence ?? []).map((id) => ix.ev.get(id)!).filter(Boolean);
  const nbrs = model.edges.filter((e) => e.from === n.id || e.to === n.id);
  const mine = ideas.filter((i) => i.related.includes(n.id));
  const other = n.perspective === 'science' ? 'software' : 'science';
  const concerns = (n.limitations ?? []).filter((l) => l.severity !== 'low').length;
  return (
    <div className="detail">
      <header>
        <div className="kindtag">{n.perspective} · {n.kind} · <code>{n.id}</code></div>
        <h2>{n.title}</h2>
        <Badges n={n} concerns={concerns} ideas={mine.filter((i) => i.status === 'idea' || i.status === 'investigating').length} />
      </header>
      <div role="tablist" className="tabs" aria-label="Detail sections">
        {(['detail', 'evidence', 'ideas'] as const).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'on' : ''} onClick={() => setTab(t)}>
            {t === 'detail' ? 'Mechanism' : t === 'evidence' ? `Evidence (${ev.length})` : `Ideas (${mine.length})`}
          </button>
        ))}
      </div>
      {tab === 'detail' && (<div role="tabpanel">
        <p className="lead">{n.purpose}</p>
        {n.technical && <p>{n.technical}</p>}
        {n.equation && <pre className="eq" aria-label="Equation">{n.equation}</pre>}
        <List title="Inputs" items={n.inputs} /><List title="Outputs" items={n.outputs} />
        {(n.chosen || n.why || n.alternatives?.length) && (
          <section className="decision"><h4>Decision</h4>
            {n.chosen && <p><b>Chosen:</b> {n.chosen}</p>}
            {n.why && <p><b>Why:</b> {n.why}</p>}
            {n.alternatives?.length ? <><b>Alternatives and trade-offs</b><ul>{n.alternatives.map((a, i) => <li key={i}>{a}</li>)}</ul></> : null}
          </section>
        )}
        <List title="Assumptions" items={n.assumptions} />
        {n.limitations?.length ? (<section><h4>Limitations and impact</h4><ul className="lims">{n.limitations.map((l, i) => (
          <li key={i} className={`sev-${l.severity}`}><span className="sev">{l.severity}</span> {l.text}<div className="impact">Impact: {l.impact}</div></li>))}</ul></section>) : null}
        {n.reviewNote && <p className="note"><b>Review note.</b> {n.reviewNote}</p>}
        <p className="muted small-t"><b>{n.review}:</b> {model.reviewLegend[n.review]}</p>
        {n.mirrors?.length ? (<section><h4>{n.perspective === 'science' ? 'Implemented in (software map)' : 'Scientific purpose (science map)'}</h4>
          <ul className="links">{n.mirrors.map((m) => (<li key={m}><button className="link" data-testid={`mirror-${m}`} onClick={() => onGo(m)}>{ix.nodes.get(m)?.title} <span className="muted">→ {other}</span></button></li>))}</ul></section>)
          : n.perspective === 'science' ? <p className="muted small-t">No software node implements this ({n.lifecycle}).</p> : null}
        {nbrs.length ? (<section><h4>Relationships</h4><ul className="links">{nbrs.map((e) => {
          const out = e.from === n.id, o = out ? e.to : e.from;
          return <li key={e.id}><span className="muted">{out ? 'this' : ix.nodes.get(o)?.title} </span><i>{e.label}</i><span className="muted"> {out ? ix.nodes.get(o)?.title : 'this'}</span> <button className="link" onClick={() => onGo(o)}>go</button></li>;
        })}</ul></section>) : null}
      </div>)}
      {tab === 'evidence' && (<div role="tabpanel">
        <p className="muted small-t">Evidence kind says where it lives; the second chip says what sort of check it is. A test file is not a validation of the science: read the note.</p>
        <ul className="evs">{ev.map((e) => <EvidenceCard key={e.id} e={e} />)}</ul>
      </div>)}
      {tab === 'ideas' && (<div role="tabpanel">
        <div className="row"><button className="btn" data-testid="new-idea-for-node" onClick={() => onNewIdea(n.id)}>New idea for this node</button>
          <button className="btn ghost" onClick={() => onOpenIdeas(n.id)}>Open workspace</button></div>
        {mine.length === 0 ? <p className="muted">No ideas linked yet.</p> : <ul className="ideas-mini">{mine.map((i) => (
          <li key={i.id}><button className="link" onClick={() => onOpenIdeas(i.id)}>{i.title}</button>{' '}
            <span className="chip">{i.status}</span>{i.origin === 'claude-proposal' && <span className="chip proposal">proposal</span>}</li>))}</ul>}
      </div>)}
    </div>
  );
}

export function LeafDetail({ e, node, onGo }: { e: Evidence; node: MNode; onGo: (id: string) => void }) {
  return (
    <div className="detail">
      <header><div className="kindtag">implementation evidence · <code>{e.id}</code></div><h2>{whereOf(e)}</h2></header>
      <ul className="evs"><EvidenceCard key={e.id} e={e} defaultOpen /></ul>
      <p className="muted small-t">Cited by <button className="link" onClick={() => onGo(node.id)}>{node.title}</button>.</p>
    </div>
  );
}
