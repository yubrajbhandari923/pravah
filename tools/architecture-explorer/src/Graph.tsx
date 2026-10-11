import { createContext, useContext, useEffect, useMemo } from 'react';
import { Background, BackgroundVariant, Controls, Handle, MarkerType, Position, ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type Node, type NodeProps } from '@xyflow/react';
import { ancestors, hasKids, isLeaf, layout, type Idx } from './layout';
import type { MNode, Persp } from './types';

export interface Ctx {
  ix: Idx; selected: string | null; expanded: Set<string>; dim: (id: string) => boolean; linked: Set<string>;
  concerns: (id: string) => number; ideas: (id: string) => number; select: (id: string) => void; toggle: (id: string) => void;
}
export const GraphCtx = createContext<Ctx>(null as unknown as Ctx);

const REVIEW_GLYPH = { 'verified-scan': '●', 'docs-only': '◐', 'needs-inspection': '○' } as const;
const REVIEW_TEXT = { 'verified-scan': 'code read', 'docs-only': 'docs only', 'needs-inspection': 'needs inspection' } as const;
const KIND_LABEL: Record<string, string> = { concept: 'concept', mechanism: 'mechanism', decision: 'decision', module: 'module', component: 'component' };

export function Badges({ n, concerns, ideas }: { n: MNode; concerns: number; ideas: number }) {
  return (
    <div className="badges">
      <span className={`pill life-${n.lifecycle}`}>{n.lifecycle}</span>
      <span className={`pill rev-${n.review}`} title={n.review}><span aria-hidden>{REVIEW_GLYPH[n.review]}</span> {REVIEW_TEXT[n.review]}</span>
      {concerns > 0 && <span className="pill concern" title={`${concerns} high/medium limitation(s)`}>▲ {concerns}</span>}
      {ideas > 0 && <span className="pill idea" title={`${ideas} open idea(s) linked`}>✎ {ideas}</span>}
    </div>
  );
}

function Handles() {
  return (<>
    <Handle id="tl" type="target" position={Position.Left} isConnectable={false} />
    <Handle id="sr" type="source" position={Position.Right} isConnectable={false} />
    <Handle id="tt" type="target" position={Position.Top} isConnectable={false} />
    <Handle id="tb" type="target" position={Position.Bottom} isConnectable={false} />
    <Handle id="st" type="source" position={Position.Top} isConnectable={false} />
    <Handle id="sb" type="source" position={Position.Bottom} isConnectable={false} />
    <Handle id="sl" type="source" position={Position.Left} isConnectable={false} />
    <Handle id="tr" type="target" position={Position.Right} isConnectable={false} />
  </>);
}

function keys(c: Ctx, id: string, open: boolean, kids: boolean) {
  return (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') { e.preventDefault(); c.select(id); }
    else if ((e.key === 'ArrowRight' || e.key === ' ') && kids && !open) { e.preventDefault(); c.toggle(id); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); if (open) c.toggle(id); else { const p = ancestors(c.ix, id)[0]; if (p) c.select(p); } }
  };
}

function ModelNode({ data }: NodeProps) {
  const id = (data as { id: string }).id;
  const c = useContext(GraphCtx);
  const n = c.ix.nodes.get(id)!;
  const open = c.expanded.has(id) && hasKids(c.ix, id);
  const kids = hasKids(c.ix, id);
  const cls = ['mnode', open ? 'is-group' : 'is-card', `kind-${n.kind}`, `life-${n.lifecycle}`, c.selected === id ? 'sel' : '', c.dim(id) ? 'dim' : '', c.linked.has(id) ? 'linked' : ''].join(' ');
  return (
    <div className={cls} tabIndex={0} role="button" aria-expanded={kids ? open : undefined} data-testid={`node-${id}`} data-expanded={open}
      aria-label={`${KIND_LABEL[n.kind]}: ${n.title}. ${n.lifecycle}, ${REVIEW_TEXT[n.review]}.${kids ? (open ? ' Expanded.' : ' Collapsed.') : ''}`}
      onClick={() => c.select(id)} onDoubleClick={() => kids && c.toggle(id)} onKeyDown={keys(c, id, open, kids)}>
      <Handles />
      <div className="mhead">
        <div className="mtitle">
          <span className="kindtag">{KIND_LABEL[n.kind]}</span>
          <strong>{n.title}</strong>
        </div>
        {kids && (
          <button className="chev" aria-label={open ? `Collapse ${n.title}` : `Expand ${n.title}`} data-testid={`toggle-${id}`} title={open ? 'Collapse' : 'Expand'}
            onClick={(e) => { e.stopPropagation(); c.toggle(id); }}>{open ? '−' : '+'}</button>
        )}
      </div>
      {!open && <p className="mrole">{n.role}</p>}
      <Badges n={n} concerns={c.concerns(id)} ideas={c.ideas(id)} />
    </div>
  );
}

function LeafNode({ data }: NodeProps) {
  const id = (data as { id: string }).id;
  const c = useContext(GraphCtx);
  const [, nid, eid] = id.split(':');
  const e = c.ix.ev.get(eid)!;
  const where = e.lines ? `${e.path}:${e.lines[0]}${e.lines[1] !== e.lines[0] ? '–' + e.lines[1] : ''}` : e.path;
  return (
    <div className={`leaf ek-${e.kind} ${c.selected === id ? 'sel' : ''} ${c.dim(nid) ? 'dim' : ''}`} tabIndex={0} role="button" data-testid={`leaf-${id}`}
      aria-label={`${e.kind} evidence ${where}`} onClick={() => c.select(id)} onKeyDown={(ev) => { if (ev.key === 'Enter') c.select(id); else if (ev.key === 'ArrowLeft') c.select(nid); }}>
      <Handles />
      <span className="ekind">{e.kind}{e.validation && e.validation !== 'none' ? ` · ${e.validation}` : ''}</span>
      <code title={where}>{where}</code>
    </div>
  );
}

const nodeTypes = { model: ModelNode, leaf: LeafNode };

function Fit({ focus, persp }: { focus: { id: string; tick: number } | null; persp: Persp }) {
  const rf = useReactFlow();
  useEffect(() => {
    const t = setTimeout(() => rf.fitView({ duration: 280, padding: 0.12, nodes: focus ? [{ id: focus.id }] : undefined, maxZoom: 1.1 }), 60);
    return () => clearTimeout(t);
  }, [focus?.tick, persp]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

export function GraphView({ ctx, persp, focus }: { ctx: Ctx; persp: Persp; focus: { id: string; tick: number } | null }) {
  const lay = useMemo(() => layout(ctx.ix, persp, ctx.expanded), [ctx.ix, persp, ctx.expanded]);
  const nodes: Node[] = lay.nodes.map((n) => ({
    id: n.id, type: n.leaf ? 'leaf' : 'model', position: { x: n.x, y: n.y }, parentId: n.parentId, data: { id: n.id },
    style: { width: n.w, height: n.h }, zIndex: n.expanded ? 0 : 2, draggable: false, selectable: false, focusable: false,
  }));
  const sel = ctx.selected;
  const edges: Edge[] = lay.edges.map((e) => ({
    id: e.id, source: e.source, target: e.target, sourceHandle: e.sh, targetHandle: e.th, label: e.label, type: 'default', zIndex: 1,
    className: `ek-${e.kind} ${e.lifted ? 'lifted' : ''} ${sel && (e.source === sel || e.target === sel) ? 'hot' : ''}`,
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 }, focusable: false,
    labelBgPadding: [6, 3], labelBgBorderRadius: 3, data: { kind: e.kind },
  }));
  return (
    <GraphCtx.Provider value={ctx}>
      <ReactFlowProvider>
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView minZoom={0.15} maxZoom={1.6} nodesDraggable={false} nodesConnectable={false}
          elementsSelectable={false} nodesFocusable={false} edgesFocusable={false} panOnScroll proOptions={{ hideAttribution: false }} data-testid="graph">
          <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="var(--dot)" />
          <Controls showInteractive={false} aria-label="Zoom and fit" />
          <Fit focus={focus} persp={persp} />
        </ReactFlow>
      </ReactFlowProvider>
    </GraphCtx.Provider>
  );
}

export { isLeaf };
