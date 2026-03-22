export default function GapView({ papers, gap, gapL, onRunGap }) {
  const sections = [
    { l: 'Themes', i: gap?.themes, c: 'var(--accent-green)' },
    { l: 'Gaps in Literature', i: gap?.gaps, c: 'var(--accent-red)' },
    { l: 'Contradictions', i: gap?.contradictions, c: 'var(--accent-orange)' },
    { l: 'Suggested Directions', i: gap?.suggested_directions, c: 'var(--accent-blue)' },
    { l: 'Missing Baselines', i: gap?.missing_baselines, c: 'var(--accent-purple)' },
    { l: 'Search Queries to Try', i: gap?.suggested_search_queries, c: '#5bc4b0' },
  ]

  return (
    <div style={{ padding: '20px 32px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2 style={{ fontSize: 18, color: 'var(--text-primary)', fontWeight: 400, margin: 0 }}>Gap Analysis</h2>
        <button onClick={onRunGap} disabled={gapL || papers.length < 2} className="btn-primary" style={{ opacity: papers.length < 2 ? 0.4 : 1 }}>{gapL ? 'Analyzing...' : '⚡ Run Gap Analysis'}</button>
      </div>

      {!gap ? (
        <div style={{ textAlign: 'center', padding: 60, color: 'var(--text-faint)', fontFamily: 'var(--mono)' }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>◇</div>
          Click "⚡ Run Gap Analysis" to identify themes, gaps, and research directions across your {papers.length} papers.
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          {sections.map(x => x.i?.length > 0 ? (
            <div key={x.l} className="section-box">
              <div className="section-label" style={{ color: x.c }}>{x.l}</div>
              {x.i.map((it, j) => (
                <div key={j} style={{ fontSize: 13, color: '#a0aab5', lineHeight: 1.5, marginBottom: 6, paddingLeft: 10, borderLeft: '2px solid ' + x.c + '44' }}>{it}</div>
              ))}
            </div>
          ) : null)}
        </div>
      )}
    </div>
  )
}
