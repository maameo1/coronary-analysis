import { useRef, useEffect, useState } from 'react'
import { gtc, tagStyle, computeRelationships, computeClusters } from '../utils'

export default function GraphView({ papers, onSelect }) {
  const withSummary = papers.filter(p => p.summary)
  const rels = computeRelationships(withSummary)
  const { clusters, bridges } = withSummary.length >= 2 ? computeClusters(withSummary, rels) : { clusters: [], bridges: [] }
  const [activeTab, setActiveTab] = useState('clusters') // clusters | connections | graph

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

  const CLUSTER_COLORS = ['#5b8a72', '#7eb8da', '#9070c4', '#c49070', '#c4c470', '#c47070', '#7ec4c4', '#c490d1']

  return (
    <div style={{ padding: '20px 32px' }}>
      <h2 style={{ fontSize: 18, color: 'var(--text-primary)', fontWeight: 400, margin: '0 0 16px' }}>Paper Relationships</h2>

      {/* Sub-tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20 }}>
        <span onClick={() => setActiveTab('clusters')} style={tabStyle(activeTab === 'clusters')}>Clusters ({clusters.length})</span>
        <span onClick={() => setActiveTab('connections')} style={tabStyle(activeTab === 'connections')}>Connections ({rels.length})</span>
        <span onClick={() => setActiveTab('graph')} style={tabStyle(activeTab === 'graph')}>Visual Graph</span>
      </div>

      {/* Clusters Tab */}
      {activeTab === 'clusters' && (
        <div>
          {clusters.length === 0 ? (
            <div style={{ fontSize: 13, color: 'var(--text-faint)', fontFamily: 'var(--mono)', padding: 40, textAlign: 'center' }}>No clusters detected. Papers may need more overlapping tags or methods.</div>
          ) : (
            <>
              {clusters.map((cluster, ci) => (
                <div key={ci} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 10, padding: '18px 22px', marginBottom: 16 }}>
                  {/* Cluster header */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                    <div style={{ width: 12, height: 12, borderRadius: '50%', background: CLUSTER_COLORS[ci % CLUSTER_COLORS.length], flexShrink: 0 }} />
                    <h3 style={{ fontSize: 15, color: 'var(--text-primary)', fontWeight: 400, margin: 0, flex: 1 }}>{cluster.name}</h3>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>{cluster.size} papers</span>
                  </div>

                  {/* Tags */}
                  {cluster.tags.length > 0 && (
                    <div style={{ marginBottom: 10 }}>
                      {cluster.tags.map(t => <span key={t} style={tagStyle(t)}>{t}</span>)}
                    </div>
                  )}

                  {/* Why grouped */}
                  {cluster.reasons.length > 0 && (
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 12, padding: '6px 10px', background: 'var(--bg-primary)', borderRadius: 4 }}>
                      Grouped by: {cluster.reasons.slice(0, 4).join(' · ')}
                    </div>
                  )}

                  {/* Papers in cluster */}
                  {cluster.papers.map(p => (
                    <div key={p.id} onClick={() => onSelect(p.id)} style={{ padding: '8px 12px', marginBottom: 4, borderRadius: 6, cursor: 'pointer', borderLeft: '3px solid ' + CLUSTER_COLORS[ci % CLUSTER_COLORS.length] + '44', transition: 'background 0.15s' }}
                      onMouseEnter={e => e.currentTarget.style.background = '#1a2030'}
                      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.3 }}>
                        {p.starred && <span style={{ color: '#c4c470', marginRight: 4 }}>★</span>}
                        {p.title}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-faint)', fontFamily: 'var(--mono)', marginTop: 2 }}>
                        {(p.authors || []).slice(0, 2).join(', ')}{(p.authors || []).length > 2 ? ' et al.' : ''} · {p.published}
                      </div>
                    </div>
                  ))}
                </div>
              ))}

              {/* Bridge papers */}
              {bridges.length > 0 && (
                <div style={{ marginTop: 20 }}>
                  <h3 style={{ fontSize: 14, color: 'var(--accent-purple)', fontWeight: 400, margin: '0 0 12px', fontFamily: 'var(--mono)' }}>Bridge Papers (connecting multiple clusters)</h3>
                  {bridges.map((b, i) => (
                    <div key={i} onClick={() => onSelect(b.paper.id)} style={{ background: 'var(--bg-secondary)', border: '1px solid #2a2535', borderRadius: 8, padding: '10px 14px', marginBottom: 6, cursor: 'pointer' }}>
                      <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>{b.paper.title}</div>
                      <div style={{ fontSize: 11, color: '#9070c4', fontFamily: 'var(--mono)', marginTop: 4 }}>Connects: {b.clusters.join(' ↔ ')}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Unclustered papers */}
              {(() => {
                const clusteredIds = new Set(clusters.flatMap(c => c.papers.map(p => p.id)))
                const unclustered = withSummary.filter(p => !clusteredIds.has(p.id))
                if (unclustered.length === 0) return null
                return (
                  <div style={{ marginTop: 20 }}>
                    <h3 style={{ fontSize: 14, color: 'var(--text-muted)', fontWeight: 400, margin: '0 0 12px', fontFamily: 'var(--mono)' }}>Unclustered ({unclustered.length})</h3>
                    {unclustered.map(p => (
                      <div key={p.id} onClick={() => onSelect(p.id)} style={{ padding: '6px 12px', fontSize: 13, color: 'var(--text-faint)', cursor: 'pointer', marginBottom: 4 }}
                        onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
                        onMouseLeave={e => e.currentTarget.style.color = 'var(--text-faint)'}>
                        {p.title}
                      </div>
                    ))}
                  </div>
                )
              })()}
            </>
          )}
        </div>
      )}

      {/* Connections Tab */}
      {activeTab === 'connections' && (
        <div>
          {rels.length === 0 && <div style={{ fontSize: 13, color: 'var(--text-faint)', fontFamily: 'var(--mono)', padding: 40, textAlign: 'center' }}>No connections found.</div>}
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
      )}

      {/* Graph Tab */}
      {activeTab === 'graph' && (
        <RelCanvas papers={withSummary} clusters={clusters} onSelect={onSelect} />
      )}
    </div>
  )
}

function RelCanvas({ papers, clusters, onSelect }) {
  const canvasRef = useRef(null)
  const nodesRef = useRef([])
  const edgesRef = useRef([])
  const [hovered, setHovered] = useState(null)
  const COLORS = ['#5b8a72', '#7eb8da', '#9070c4', '#c49070', '#c4c470', '#c47070', '#7ec4c4', '#c490d1']

  useEffect(() => {
    const rels = computeRelationships(papers)
    // Assign cluster colors
    const clusterMap = {}
    clusters.forEach((c, ci) => { c.papers.forEach(p => { clusterMap[p.id] = ci }) })

    const nodes = papers.map((p, i) => {
      const angle = (2 * Math.PI * i) / papers.length
      const r = Math.min(250, papers.length * 18)
      const ci = clusterMap[p.id]
      return { id: p.id, title: p.title, color: ci !== undefined ? COLORS[ci % COLORS.length] : '#3a4555', x: 400 + r * Math.cos(angle), y: 300 + r * Math.sin(angle), vx: 0, vy: 0 }
    })
    const edges = rels.filter(r => r.strength >= 1).map(r => ({ source: r.a, target: r.b, strength: r.strength }))

    for (let iter = 0; iter < 120; iter++) {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y
          const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
          const f = 800 / (dist * dist)
          nodes[i].vx -= (dx / dist) * f; nodes[i].vy -= (dy / dist) * f
          nodes[j].vx += (dx / dist) * f; nodes[j].vy += (dy / dist) * f
        }
      }
      for (const e of edges) {
        const s = nodes.find(n => n.id === e.source), t = nodes.find(n => n.id === e.target)
        if (!s || !t) continue
        const dx = t.x - s.x, dy = t.y - s.y, dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
        const f = (dist - 150) * 0.01 * e.strength
        s.vx += (dx / dist) * f; s.vy += (dy / dist) * f
        t.vx -= (dx / dist) * f; t.vy -= (dy / dist) * f
      }
      for (const n of nodes) {
        n.vx += (400 - n.x) * 0.01; n.vy += (300 - n.y) * 0.01
        n.x += n.vx * 0.3; n.y += n.vy * 0.3; n.vx *= 0.8; n.vy *= 0.8
        n.x = Math.max(40, Math.min(760, n.x)); n.y = Math.max(40, Math.min(560, n.y))
      }
    }
    nodesRef.current = nodes; edgesRef.current = edges; draw()
  }, [papers, clusters])

  function draw() {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d')
    const nodes = nodesRef.current, edges = edgesRef.current
    ctx.clearRect(0, 0, 800, 600); ctx.fillStyle = '#0a0e13'; ctx.fillRect(0, 0, 800, 600)
    for (const e of edges) {
      const s = nodes.find(n => n.id === e.source), t = nodes.find(n => n.id === e.target)
      if (!s || !t) continue
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y)
      ctx.strokeStyle = e.strength >= 3 ? '#2a5a3a' : e.strength >= 2 ? '#1e3a2e' : '#162520'
      ctx.lineWidth = Math.min(e.strength, 3); ctx.stroke()
    }
    for (const n of nodes) {
      const isH = hovered === n.id
      ctx.beginPath(); ctx.arc(n.x, n.y, isH ? 12 : 8, 0, Math.PI * 2)
      ctx.fillStyle = n.color; ctx.fill()
      ctx.strokeStyle = isH ? '#ffffff' : '#e8e8e8'; ctx.lineWidth = isH ? 2 : 1; ctx.stroke()
      ctx.fillStyle = '#8a9bb5'; ctx.font = '10px monospace'; ctx.textAlign = 'center'
      ctx.fillText(n.title.length > 28 ? n.title.slice(0, 28) + '...' : n.title, n.x, n.y + 20)
    }
  }

  useEffect(() => { draw() }, [hovered])

  function findNode(e) {
    if (!canvasRef.current) return null
    const rect = canvasRef.current.getBoundingClientRect()
    const mx = (e.clientX - rect.left) * (800 / rect.width), my = (e.clientY - rect.top) * (600 / rect.height)
    return nodesRef.current.find(n => Math.sqrt((n.x - mx) ** 2 + (n.y - my) ** 2) < 14)
  }

  return <canvas ref={canvasRef} width={800} height={600}
    onMouseMove={e => { const n = findNode(e); setHovered(n ? n.id : null); if (canvasRef.current) canvasRef.current.style.cursor = n ? 'pointer' : 'default' }}
    onClick={e => { const n = findNode(e); if (n) onSelect(n.id) }}
    style={{ width: '100%', maxWidth: 800, height: 'auto', borderRadius: 8, border: '1px solid var(--border)' }} />
}
