import { useRef, useEffect, useState, useMemo } from 'react'
import { gtc, tagStyle, computeRelationships, computeClusters } from '../utils'

export default function GraphView({ papers, onSelect }) {
  const withSummary = useMemo(() => papers.filter(p => p.summary), [papers])
  const rels = useMemo(() => computeRelationships(withSummary), [withSummary])
  const { clusters, bridges } = useMemo(() => withSummary.length >= 2 ? computeClusters(withSummary, rels) : { clusters: [], bridges: [] }, [withSummary, rels])
  const [activeTab, setActiveTab] = useState('clusters')
  const COLORS = useMemo(() => ['#5b8a72', '#7eb8da', '#9070c4', '#c49070', '#c4c470', '#c47070', '#7ec4c4', '#c490d1'], [])

  if (withSummary.length < 2) {
    return (
      <div style={{ padding: '20px 32px', textAlign: 'center', paddingTop: 60, color: 'var(--text-faint)', fontFamily: 'var(--mono)' }}>
        Add 2+ papers with summaries to see relationships
      </div>
    )
  }

  const tabStyle = (active) => ({
    padding: '8px 16px', fontSize: 12, fontFamily: 'var(--mono)', cursor: 'pointer',
    background: active ? '#1a2535' : 'transparent', color: active ? 'var(--accent-green-light)' : 'var(--text-faint)',
    border: '1px solid ' + (active ? 'var(--border)' : 'transparent'), borderRadius: 6,
  })

  return (
    <div style={{ padding: '20px 32px' }}>
      <h2 style={{ fontSize: 18, color: 'var(--text-primary)', fontWeight: 400, margin: '0 0 16px' }}>Paper Relationships</h2>
      <div style={{ display: 'flex', gap: 6, marginBottom: 20 }}>
        <span onClick={() => setActiveTab('clusters')} style={tabStyle(activeTab === 'clusters')}>Clusters ({clusters.length})</span>
        <span onClick={() => setActiveTab('connections')} style={tabStyle(activeTab === 'connections')}>Connections ({rels.length})</span>
        <span onClick={() => setActiveTab('graph')} style={tabStyle(activeTab === 'graph')}>Interactive Graph</span>
      </div>

      {activeTab === 'clusters' && <ClustersTab clusters={clusters} bridges={bridges} withSummary={withSummary} onSelect={onSelect} colors={COLORS} />}
      {activeTab === 'connections' && <ConnectionsTab rels={rels} onSelect={onSelect} />}
      {activeTab === 'graph' && <InteractiveGraph papers={withSummary} clusters={clusters} rels={rels} onSelect={onSelect} colors={COLORS} />}
    </div>
  )
}

// ── Clusters Tab ─────────────────────────────────────────────────────────
function ClustersTab({ clusters, bridges, withSummary, onSelect, colors }) {
  if (clusters.length === 0) {
    return <div style={{ fontSize: 13, color: 'var(--text-faint)', fontFamily: 'var(--mono)', padding: 40, textAlign: 'center' }}>No clusters detected yet.</div>
  }
  const clusteredIds = new Set(clusters.flatMap(c => c.papers.map(p => p.id)))
  const unclustered = withSummary.filter(p => !clusteredIds.has(p.id))

  return (
    <div>
      {clusters.map((cluster, ci) => (
        <div key={ci} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10, padding: '18px 22px', marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
            <div style={{ width: 12, height: 12, borderRadius: '50%', background: colors[ci % colors.length], flexShrink: 0 }} />
            <h3 style={{ fontSize: 15, color: 'var(--text-primary)', fontWeight: 400, margin: 0, flex: 1 }}>{cluster.name}</h3>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>{cluster.size} papers</span>
          </div>
          {cluster.tags.length > 0 && <div style={{ marginBottom: 10 }}>{cluster.tags.map(t => <span key={t} style={tagStyle(t)}>{t}</span>)}</div>}
          {cluster.reasons.length > 0 && <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 12, padding: '6px 10px', background: 'var(--bg-primary)', borderRadius: 4 }}>Grouped by: {cluster.reasons.slice(0, 4).join(' · ')}</div>}
          {cluster.papers.map(p => (
            <div key={p.id} onClick={() => onSelect(p.id)} style={{ padding: '8px 12px', marginBottom: 4, borderRadius: 6, cursor: 'pointer', borderLeft: '3px solid ' + colors[ci % colors.length] + '44' }}
              onMouseEnter={e => e.currentTarget.style.background = '#1a2030'} onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
              <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.3 }}>{p.starred && <span style={{ color: '#c4c470', marginRight: 4 }}>★</span>}{p.title}</div>
              <div style={{ fontSize: 11, color: 'var(--text-faint)', fontFamily: 'var(--mono)', marginTop: 2 }}>{(p.authors || []).slice(0, 2).join(', ')}{(p.authors || []).length > 2 ? ' et al.' : ''} · {p.published}</div>
            </div>
          ))}
        </div>
      ))}
      {bridges.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: 14, color: 'var(--accent-purple)', fontWeight: 400, margin: '0 0 12px', fontFamily: 'var(--mono)' }}>Bridge Papers</h3>
          {bridges.map((b, i) => (
            <div key={i} onClick={() => onSelect(b.paper.id)} style={{ background: 'var(--bg-secondary)', border: '1px solid #2a2535', borderRadius: 8, padding: '10px 14px', marginBottom: 6, cursor: 'pointer' }}>
              <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>{b.paper.title}</div>
              <div style={{ fontSize: 11, color: '#9070c4', fontFamily: 'var(--mono)', marginTop: 4 }}>Connects: {b.clusters.join(' ↔ ')}</div>
            </div>
          ))}
        </div>
      )}
      {unclustered.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: 14, color: 'var(--text-muted)', fontWeight: 400, margin: '0 0 12px', fontFamily: 'var(--mono)' }}>Unclustered ({unclustered.length})</h3>
          {unclustered.map(p => (
            <div key={p.id} onClick={() => onSelect(p.id)} style={{ padding: '6px 12px', fontSize: 13, color: 'var(--text-faint)', cursor: 'pointer', marginBottom: 4 }}
              onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'} onMouseLeave={e => e.currentTarget.style.color = 'var(--text-faint)'}>{p.title}</div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Connections Tab ───────────────────────────────────────────────────────
function ConnectionsTab({ rels, onSelect }) {
  if (rels.length === 0) return <div style={{ fontSize: 13, color: 'var(--text-faint)', fontFamily: 'var(--mono)', padding: 40, textAlign: 'center' }}>No connections found.</div>
  return (
    <div>
      {rels.slice(0, 30).map((r, i) => (
        <div key={i} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 8, padding: '12px 16px', marginBottom: 8 }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontSize: 13, color: 'var(--text-primary)', flex: 1, cursor: 'pointer' }} onClick={() => onSelect(r.a)}>{r.aTitle?.length > 50 ? r.aTitle.slice(0, 50) + '...' : r.aTitle}</span>
            <span style={{ fontSize: 11, color: 'var(--accent-green)', fontFamily: 'var(--mono)', flexShrink: 0 }}>↔</span>
            <span style={{ fontSize: 13, color: 'var(--text-primary)', flex: 1, cursor: 'pointer', textAlign: 'right' }} onClick={() => onSelect(r.b)}>{r.bTitle?.length > 50 ? r.bTitle.slice(0, 50) + '...' : r.bTitle}</span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>{r.reasons.join(' · ')}</div>
        </div>
      ))}
    </div>
  )
}

// ── Interactive Graph with Zoom + Pan + Tooltips ─────────────────────────
function InteractiveGraph({ papers, clusters, rels, onSelect, colors }) {
  const canvasRef = useRef(null)
  const tooltipRef = useRef(null)
  const nodesRef = useRef([])
  const edgesRef = useRef([])
  const viewRef = useRef({ x: 0, y: 0, scale: 1 })
  const dragRef = useRef({ dragging: false, startX: 0, startY: 0, startViewX: 0, startViewY: 0 })
  const hoveredRef = useRef(null)

  // Build cluster color map (only when clusters change)
  const clusterMap = useMemo(() => {
    const m = {}
    clusters.forEach((c, ci) => { c.papers.forEach(p => { m[p.id] = ci }) })
    return m
  }, [clusters])

  // Initialize nodes and run force simulation
  useEffect(() => {
    const nodes = papers.map((p, i) => {
      const angle = (2 * Math.PI * i) / papers.length
      const r = Math.min(280, papers.length * 20)
      const ci = clusterMap[p.id]
      return { id: p.id, title: p.title, color: ci !== undefined ? colors[ci % colors.length] : '#3a4555', x: 400 + r * Math.cos(angle), y: 320 + r * Math.sin(angle), vx: 0, vy: 0 }
    })
    // Build edge lookup map for O(1) access
    const edgeList = rels.filter(r => r.strength >= 1).map(r => ({ source: r.a, target: r.b, strength: r.strength, reasons: r.reasons }))
    const nodeMap = {}
    nodes.forEach(n => { nodeMap[n.id] = n })

    // Force simulation — use nodeMap for O(1) lookups
    for (let iter = 0; iter < 100; iter++) {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
          const f = 1000 / (dist * dist)
          nodes[i].vx -= (dx / dist) * f; nodes[i].vy -= (dy / dist) * f
          nodes[j].vx += (dx / dist) * f; nodes[j].vy += (dy / dist) * f
        }
      }
      for (const e of edgeList) {
        const s = nodeMap[e.source], t = nodeMap[e.target]
        if (!s || !t) continue
        const dx = t.x - s.x, dy = t.y - s.y, dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
        const f = (dist - 160) * 0.01 * e.strength
        s.vx += (dx / dist) * f; s.vy += (dy / dist) * f
        t.vx -= (dx / dist) * f; t.vy -= (dy / dist) * f
      }
      for (const n of nodes) {
        n.vx += (400 - n.x) * 0.008; n.vy += (320 - n.y) * 0.008
        n.x += n.vx * 0.3; n.y += n.vy * 0.3; n.vx *= 0.75; n.vy *= 0.75
      }
    }
    nodesRef.current = nodes; edgesRef.current = edgeList
    viewRef.current = { x: 0, y: 0, scale: 1 }
    draw()
  }, [papers, clusterMap, rels])

  function draw() {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = 800, H = 640
    const { x: vx, y: vy, scale } = viewRef.current
    const nodes = nodesRef.current, edges = edgesRef.current
    const hid = hoveredRef.current

    ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#0a0e13'; ctx.fillRect(0, 0, W, H)
    ctx.save(); ctx.translate(vx, vy); ctx.scale(scale, scale)

    // Build connected set for hovered node (once per draw, not per edge)
    const connectedIds = new Set()
    if (hid) { edges.forEach(e => { if (e.source === hid) connectedIds.add(e.target); if (e.target === hid) connectedIds.add(e.source) }) }

    for (const e of edges) {
      const s = nodes.find(n => n.id === e.source), t = nodes.find(n => n.id === e.target)
      if (!s || !t) continue
      const isHL = hid && (e.source === hid || e.target === hid)
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y)
      ctx.strokeStyle = isHL ? '#4a8a5a' : (e.strength >= 3 ? '#2a5a3a' : e.strength >= 2 ? '#1e3a2e' : '#162520')
      ctx.lineWidth = isHL ? Math.min(e.strength + 1, 4) : Math.min(e.strength, 3); ctx.stroke()
      if (isHL && e.reasons) {
        const mx = (s.x + t.x) / 2, my = (s.y + t.y) / 2
        ctx.fillStyle = '#6a8a7a'; ctx.font = '8px Arial'; ctx.textAlign = 'center'
        ctx.fillText(e.reasons[0] || '', mx, my - 4)
      }
    }

    for (const n of nodes) {
      const isH = hid === n.id
      const isConn = hid ? connectedIds.has(n.id) : false
      const radius = isH ? 14 : (isConn ? 11 : 8)
      ctx.globalAlpha = hid ? (isH || isConn ? 1 : 0.3) : 1
      if (isH) { ctx.beginPath(); ctx.arc(n.x, n.y, 20, 0, Math.PI * 2); ctx.fillStyle = n.color + '30'; ctx.fill() }
      ctx.beginPath(); ctx.arc(n.x, n.y, radius, 0, Math.PI * 2)
      ctx.fillStyle = n.color; ctx.fill()
      ctx.strokeStyle = isH ? '#ffffff' : '#c0c0c0'; ctx.lineWidth = isH ? 2.5 : 1; ctx.stroke()
      ctx.fillStyle = isH || isConn ? '#d4dae0' : '#6a7b8f'
      ctx.font = (isH ? 'bold 11px' : '10px') + ' Arial'; ctx.textAlign = 'center'
      ctx.fillText(n.title.length > 32 ? n.title.slice(0, 32) + '...' : n.title, n.x, n.y + radius + 14)
      ctx.globalAlpha = 1
    }

    ctx.restore()
    ctx.fillStyle = '#3a4555'; ctx.font = '10px monospace'; ctx.textAlign = 'right'
    ctx.fillText('Zoom: ' + Math.round(scale * 100) + '%  |  Scroll to zoom  |  Drag to pan', W - 10, H - 8)
  }

  function screenToGraph(sx, sy) {
    const rect = canvasRef.current.getBoundingClientRect()
    const cx = (sx - rect.left) * (800 / rect.width), cy = (sy - rect.top) * (640 / rect.height)
    const { x: vx, y: vy, scale } = viewRef.current
    return { x: (cx - vx) / scale, y: (cy - vy) / scale }
  }

  function findNode(e) {
    if (!canvasRef.current) return null
    const { x, y } = screenToGraph(e.clientX, e.clientY)
    return nodesRef.current.find(n => Math.sqrt((n.x - x) ** 2 + (n.y - y) ** 2) < 16)
  }

  function updateTooltip(e, node) {
    const el = tooltipRef.current; if (!el) return
    if (!node) { el.style.display = 'none'; return }
    const rect = canvasRef.current.getBoundingClientRect()
    const tx = Math.min(e.clientX - rect.left + 15, 520)
    const ty = e.clientY - rect.top - 10
    const connected = edgesRef.current.filter(ed => ed.source === node.id || ed.target === node.id)
    let html = '<div style="font-size:13px;color:#e8e8e8;font-weight:bold;margin-bottom:6px;line-height:1.3">' + node.title + '</div>'
    if (connected.length > 0) {
      html += '<div style="font-size:10px;color:#7ec49e;font-family:monospace;margin-bottom:4px;text-transform:uppercase">Connected to:</div>'
      connected.slice(0, 5).forEach(c => {
        const otherId = c.source === node.id ? c.target : c.source
        const other = nodesRef.current.find(nd => nd.id === otherId)
        const title = other ? (other.title.length > 45 ? other.title.slice(0, 45) + '...' : other.title) : '?'
        html += '<div style="margin-bottom:4px;padding-left:8px;border-left:2px solid #2a4a39"><div style="font-size:11px;color:#b0bac5;line-height:1.2">' + title + '</div><div style="font-size:9px;color:#5a6a7a;font-family:monospace">' + (c.reasons || []).slice(0, 2).join(' · ') + '</div></div>'
      })
      if (connected.length > 5) html += '<div style="font-size:10px;color:#4a5a6a;font-family:monospace">+' + (connected.length - 5) + ' more</div>'
    } else {
      html += '<div style="font-size:11px;color:#4a5a6a;font-family:monospace">No direct connections</div>'
    }
    el.innerHTML = html
    el.style.display = 'block'; el.style.left = tx + 'px'; el.style.top = ty + 'px'
  }

  function handleMouseMove(e) {
    if (dragRef.current.dragging) {
      viewRef.current.x = dragRef.current.startViewX + (e.clientX - dragRef.current.startX)
      viewRef.current.y = dragRef.current.startViewY + (e.clientY - dragRef.current.startY)
      draw(); return
    }
    const n = findNode(e)
    const newId = n ? n.id : null
    if (newId !== hoveredRef.current) { hoveredRef.current = newId; draw() }
    if (canvasRef.current) canvasRef.current.style.cursor = n ? 'pointer' : 'grab'
    updateTooltip(e, n)
  }

  function handleMouseDown(e) {
    if (!findNode(e)) {
      dragRef.current = { dragging: true, startX: e.clientX, startY: e.clientY, startViewX: viewRef.current.x, startViewY: viewRef.current.y }
      if (canvasRef.current) canvasRef.current.style.cursor = 'grabbing'
    }
  }

  function handleMouseUp(e) {
    if (!dragRef.current.dragging) { const n = findNode(e); if (n) onSelect(n.id) }
    dragRef.current.dragging = false
    if (canvasRef.current) canvasRef.current.style.cursor = 'grab'
  }

  function handleWheel(e) {
    e.preventDefault()
    const rect = canvasRef.current.getBoundingClientRect()
    const mx = (e.clientX - rect.left) * (800 / rect.width), my = (e.clientY - rect.top) * (640 / rect.height)
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    const newScale = Math.max(0.3, Math.min(4, viewRef.current.scale * delta))
    viewRef.current.x = mx - (mx - viewRef.current.x) * (newScale / viewRef.current.scale)
    viewRef.current.y = my - (my - viewRef.current.y) * (newScale / viewRef.current.scale)
    viewRef.current.scale = newScale; draw()
  }

  return (
    <div style={{ position: 'relative' }}>
      <canvas ref={canvasRef} width={800} height={640}
        onMouseMove={handleMouseMove} onMouseDown={handleMouseDown} onMouseUp={handleMouseUp}
        onMouseLeave={() => { dragRef.current.dragging = false; hoveredRef.current = null; draw(); if (tooltipRef.current) tooltipRef.current.style.display = 'none' }}
        onWheel={handleWheel}
        style={{ width: '100%', maxWidth: 800, height: 'auto', borderRadius: 8, border: '1px solid var(--border)', cursor: 'grab' }} />
      {/* Tooltip — updated via DOM ref, no React re-renders */}
      <div ref={tooltipRef} style={{ display: 'none', position: 'absolute', background: '#151d28', border: '1px solid #2a3545', borderRadius: 8, padding: '10px 14px', maxWidth: 300, pointerEvents: 'none', zIndex: 10, boxShadow: '0 4px 20px rgba(0,0,0,0.5)' }} />
      {clusters.length > 0 && (
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text-faint)' }}>
          {clusters.map((c, i) => <span key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: '50%', background: colors[i % colors.length], display: 'inline-block' }} />{c.name}</span>)}
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}><span style={{ width: 8, height: 8, borderRadius: '50%', background: '#3a4555', display: 'inline-block' }} />Unclustered</span>
        </div>
      )}
    </div>
  )
}
