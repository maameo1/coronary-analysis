import { useState, useRef } from 'react'
import { tagStyle, getPaperUrl, getPdfUrl, genSchematic } from '../utils'

export default function DetailView({ paper, apiKey, onBack, onDelete, onToggleRead, onToggleStar, onUpdateNotes, onUpdateFigure, onUpdateSchematic, speaking, onSpeak }) {
  const p = paper, s = p.summary || {}
  const pUrl = getPaperUrl(p), pdfUrl = getPdfUrl(p)
  const figRef = useRef(null)
  const [genL, setGenL] = useState(false)

  async function handleGenSchematic() {
    if (!apiKey) { alert('Add API key in ⚙ Settings first.'); return }
    setGenL(true)
    try {
      const svg = await genSchematic(apiKey, p)
      onUpdateSchematic(p.id, svg)
    } catch (e) { alert('Schematic generation failed: ' + e.message) }
    finally { setGenL(false) }
  }

  function handleFigUpload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => { onUpdateFigure(p.id, ev.target.result) }
    reader.readAsDataURL(file)
  }

  return (
    <div style={{ padding: '20px 32px' }}>
      <button onClick={onBack} className="btn-sec" style={{ marginBottom: 20 }}>← Back</button>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
        <div style={{ flex: 1 }}>
          <h2 style={{ fontSize: 22, color: 'var(--text-primary)', fontWeight: 400, lineHeight: 1.3, margin: '0 0 8px' }}>{p.title}</h2>
          <div style={{ fontSize: 13, color: 'var(--accent-green)', fontFamily: 'var(--mono)', marginBottom: 6 }}>{(p.authors || []).join(', ')}</div>
          <div style={{ fontSize: 12, color: 'var(--text-faint)', fontFamily: 'var(--mono)', marginBottom: 8 }}>{p.published} · {p.source}{p.venue ? ' · ' + p.venue : ''}</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {pUrl && <a href={pUrl} target="_blank" rel="noopener" className="btn-sec" style={{ textDecoration: 'none', padding: '6px 12px', fontSize: 12, color: 'var(--accent-blue)' }}>🔗 View Paper</a>}
            {pdfUrl && <a href={pdfUrl} target="_blank" rel="noopener" className="btn-sec" style={{ textDecoration: 'none', padding: '6px 12px', fontSize: 12, color: 'var(--accent-green-light)' }}>📄 PDF</a>}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={() => onToggleStar(p.id)} className="btn-sec" style={{ color: p.starred ? '#c4c470' : 'var(--text-muted)', fontSize: 16, padding: '8px 12px' }}>{p.starred ? '★' : '☆'}</button>
          <button onClick={() => onSpeak(p)} className="btn-primary">{speaking ? '■ Stop' : '▶ Listen'}</button>
          <button onClick={() => onToggleRead(p.id)} className="btn-sec">{p.readStatus === 'read' ? '✓ Read' : '○ Unread'}</button>
          <button onClick={() => onDelete(p.id)} className="btn-sec" style={{ color: 'var(--accent-red)' }}>Delete</button>
        </div>
      </div>

      <div style={{ marginTop: 24 }}>
        {/* TLDR */}
        {s.tldr ? (
          <div className="section-box" style={{ background: '#0f1a15', borderColor: '#1a3329', marginBottom: 16 }}>
            <div className="section-label" style={{ color: 'var(--accent-green)' }}>TLDR</div>
            <div style={{ fontSize: 15, color: '#c4d4cc', lineHeight: 1.5 }}>{s.tldr}</div>
          </div>
        ) : <div className="section-box" style={{ background: '#1a1a15', borderColor: '#3a3329', marginBottom: 16, color: 'var(--accent-orange)', fontFamily: 'var(--mono)', fontSize: 13 }}>No summary yet. ⚡ Summarize All in library.</div>}

        {s.tags?.length > 0 && <div style={{ marginBottom: 16 }}>{s.tags.map(t => <span key={t} style={tagStyle(t)}>{t}</span>)}</div>}

        {/* Key Figure / Schematic */}
        <div className="section-box" style={{ marginBottom: 16 }}>
          <div className="section-label" style={{ color: 'var(--accent-blue)' }}>Visual Overview</div>

          {/* AI Schematic */}
          {p.schematic && p.schematic.trim().startsWith('<svg') ? (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 6 }}>AI-Generated Schematic</div>
              <div style={{ background: '#111822', borderRadius: 6, padding: 8, overflow: 'auto' }} dangerouslySetInnerHTML={{ __html: p.schematic }} />
              <button onClick={handleGenSchematic} disabled={genL} className="btn-sec" style={{ marginTop: 8, fontSize: 11, padding: '4px 10px' }}>{genL ? 'Regenerating...' : '↻ Regenerate'}</button>
            </div>
          ) : p.schematic ? (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 11, color: 'var(--accent-orange)', fontFamily: 'var(--mono)', marginBottom: 6 }}>Schematic generation returned invalid SVG.</div>
              <button onClick={handleGenSchematic} disabled={genL} className="btn-primary" style={{ fontSize: 12 }}>{genL ? 'Regenerating...' : '↻ Try Again'}</button>
            </div>
          ) : (
            <button onClick={handleGenSchematic} disabled={genL || !s.tldr} className="btn-primary" style={{ marginBottom: 12, fontSize: 12, opacity: !s.tldr ? 0.4 : 1 }}>{genL ? 'Generating...' : '🎨 Generate Method Schematic'}</button>
          )}

          {/* User-uploaded figure */}
          {p.figure ? (
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 6 }}>Key Figure</div>
              <img src={p.figure} alt="Key figure" style={{ maxWidth: '100%', borderRadius: 6, border: '1px solid var(--border)' }} />
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button onClick={() => figRef.current?.click()} className="btn-sec" style={{ fontSize: 11, padding: '4px 10px' }}>Replace</button>
                <button onClick={() => onUpdateFigure(p.id, null)} className="btn-sec" style={{ fontSize: 11, padding: '4px 10px', color: 'var(--accent-red)' }}>Remove</button>
              </div>
            </div>
          ) : (
            <div>
              <button onClick={() => figRef.current?.click()} className="btn-sec" style={{ fontSize: 12 }}>📷 Upload Key Figure</button>
              <span style={{ fontSize: 11, color: 'var(--text-faint)', fontFamily: 'var(--mono)', marginLeft: 8 }}>Screenshot or save a key figure from the paper</span>
            </div>
          )}
          <input ref={figRef} type="file" accept="image/*" onChange={handleFigUpload} style={{ display: 'none' }} />
        </div>

        {/* Info grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {[{ l: 'Key Contributions', i: s.key_contributions, c: 'var(--accent-green)' }, { l: 'Methods', i: s.methods, c: 'var(--accent-blue)' }, { l: 'Limitations', i: s.limitations, c: 'var(--accent-orange)' }, { l: 'Open Questions', i: s.open_questions, c: 'var(--accent-yellow)' }].map(x => x.i?.length > 0 ? (
            <div key={x.l} className="section-box"><div className="section-label" style={{ color: x.c }}>{x.l}</div>{x.i.map((it, j) => <div key={j} style={{ fontSize: 13, color: '#a0aab5', lineHeight: 1.5, marginBottom: 4, paddingLeft: 10, borderLeft: '2px solid ' + x.c + '33' }}>{it}</div>)}</div>
          ) : null)}
        </div>

        {/* Citations */}
        {s.key_citations_to_follow?.length > 0 && (
          <div className="section-box" style={{ marginTop: 12 }}>
            <div className="section-label" style={{ color: 'var(--accent-purple)' }}>Citations to Follow</div>
            {s.key_citations_to_follow.map((c, i) => { const ct = typeof c === 'object' && c.citation ? c.citation : String(c); const re = typeof c === 'object' ? c.reason : null; const sq = encodeURIComponent(ct.replace(/\d{4}/g, '').replace(/et al/g, '').trim()); return (
              <div key={i} style={{ marginBottom: 10, paddingLeft: 10, borderLeft: '2px solid #9070c433' }}>
                <a href={'https://scholar.google.com/scholar?q=' + sq} target="_blank" rel="noopener" style={{ fontSize: 13, color: '#b0a5d0', textDecoration: 'none', borderBottom: '1px dashed #9070c444' }}>{ct}</a>
                {re && <div style={{ fontSize: 11, color: '#6a6080', marginTop: 2, fontFamily: 'var(--mono)' }}>{re}</div>}
              </div>
            ) })}
          </div>
        )}

        {s.relevance_to_medical_imaging && <div className="section-box" style={{ marginTop: 12, background: '#1a1822', borderColor: '#2a2535' }}><div className="section-label" style={{ color: 'var(--accent-purple)' }}>Relevance to Your Research</div><div style={{ fontSize: 13, color: '#b0a5c0', lineHeight: 1.5 }}>{s.relevance_to_medical_imaging}</div></div>}

        {p.abstract && <details style={{ marginTop: 16 }}><summary style={{ fontSize: 13, color: 'var(--text-muted)', cursor: 'pointer', fontFamily: 'var(--mono)' }}>Full Abstract</summary><div style={{ marginTop: 8, fontSize: 13, color: '#8a95a5', lineHeight: 1.6, padding: '12px 16px', background: '#0e1318', borderRadius: 6 }}>{p.abstract}</div></details>}

        <div style={{ marginTop: 20 }}>
          <div className="section-label">Your Notes</div>
          <textarea value={p.notes || ''} onChange={e => onUpdateNotes(p.id, e.target.value)} placeholder="Add thoughts, connections, TODOs..." style={{ width: '100%', minHeight: 100, padding: '12px 14px', background: '#0e1318', border: '1px solid var(--border)', borderRadius: 6, color: '#b0bac5', fontSize: 13, fontFamily: 'var(--mono)', lineHeight: 1.5, resize: 'vertical', outline: 'none', boxSizing: 'border-box' }} />
        </div>
      </div>
    </div>
  )
}
