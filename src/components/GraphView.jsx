import { useRef, useEffect, useState } from 'react'
import { gtc, computeRelationships } from '../utils'

export default function GraphView({ papers, onSelect }) {
  const rels = computeRelationships(papers.filter(p => p.summary))
  const hasSummaries = papers.filter(p => p.summary).length >= 2

  return (
    <div style={{ padding: '20px 32px' }}>
      <h2 style={{ fontSize: 18, color: 'var(--text-primary)', fontWeight: 400, margin: '0 0 16px' }}>Paper Relationships</h2>
      {!hasSummaries ? (
        <div style={{ textAlign: 'center', padding: 60, color: 'var(--text-faint)', fontFamily: 'var(--mono)' }}>Add 2+ papers with summaries to see relationships</div>
      ) : (
        <>
          <RelCanvas papers={papers.filter(p => p.summary)} onSelect={onSelect} />
          <div style={{ marginTop: 20 }}>
            <h3 style={{ fontSize: 14, color: 'var(--text-primary)', fontWeight: 400, margin: '0 0 12px', fontFamily: 'var(--mono)' }}>Connections ({rels.length})</h3>
            {rels.length === 0 && <div style={{ fontSize: 13, color: 'var(--text-faint)', fontFamily: 'var(--mono)' }}>No connections found. Papers may need summaries to detect relationships.</div>}
            {rels.slice(0, 25).map((r, i) => (
              <div key={i} style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 8, padding: '12px 16px', marginBottom: 8 }}>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
                  <span style={{ fontSize: 13, color: 'var(--text-primary)', flex: 1, cursor: 'pointer' }} onClick={() => onSelect(r.a)}>{r.aTitle?.length > 50 ? r.aTitle.slice(0, 50) + '…' : r.aTitle}</span>
                  <span style={{ fontSize: 11, color: 'var(--accent-green)', fontFamily: 'var(--mono)', flexShrink: 0 }}>↔</span>
                  <span style={{ fontSize: 13, color: 'var(--text-primary)', flex: 1, cursor: 'pointer', textAlign: 'right' }} onClick={() => onSelect(r.b)}>{r.bTitle?.length > 50 ? r.bTitle.slice(0, 50) + '…' : r.bTitle}</span>
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>{r.reasons.join(' · ')}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function RelCanvas({ papers, onSelect }) {
  const canvasRef = useRef(null)
  const nodesRef = useRef([])
  const edgesRef = useRef([])
  const [hovered, setHovered] = useState(null)

  useEffect(() => {
    const rels = computeRelationships(papers)
    const nodes = papers.map((p, i) => {
      const angle = (2 * Math.PI * i) / papers.length
      const r = Math.min(250, papers.length * 18)
      return { id: p.id, title: p.title, tags: p.summary?.tags || [], x: 400 + r * Math.cos(angle), y: 300 + r * Math.sin(angle), vx: 0, vy: 0 }
    })
    const edges = rels.filter(r => r.strength >= 1).map(r => ({ source: r.a, target: r.b, strength: r.strength }))
    // Force simulation
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
        n.x += n.vx * 0.3; n.y += n.vy * 0.3
        n.vx *= 0.8; n.vy *= 0.8
        n.x = Math.max(40, Math.min(760, n.x)); n.y = Math.max(40, Math.min(560, n.y))
      }
    }
    nodesRef.current = nodes; edgesRef.current = edges
    draw()
  }, [papers])

  function draw() {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d')
    const nodes = nodesRef.current, edges = edgesRef.current
    ctx.clearRect(0, 0, 800, 600)
    ctx.fillStyle = '#0a0e13'; ctx.fillRect(0, 0, 800, 600)
    for (const e of edges) {
      const s = nodes.find(n => n.id === e.source), t = nodes.find(n => n.id === e.target)
      if (!s || !t) continue
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(t.x, t.y)
      ctx.strokeStyle = e.strength >= 3 ? '#2a5a3a' : e.strength >= 2 ? '#1e3a2e' : '#162520'
      ctx.lineWidth = Math.min(e.strength, 3); ctx.stroke()
    }
    for (const n of nodes) {
      const col = n.tags[0] ? gtc(n.tags[0]).fg : '#5b8a72'
      const isH = hovered === n.id
      ctx.beginPath(); ctx.arc(n.x, n.y, isH ? 12 : 8, 0, Math.PI * 2)
      ctx.fillStyle = col; ctx.fill()
      ctx.strokeStyle = isH ? '#ffffff' : '#e8e8e8'; ctx.lineWidth = isH ? 2 : 1; ctx.stroke()
      ctx.fillStyle = '#8a9bb5'; ctx.font = '10px monospace'; ctx.textAlign = 'center'
      ctx.fillText(n.title.length > 28 ? n.title.slice(0, 28) + '…' : n.title, n.x, n.y + 20)
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
